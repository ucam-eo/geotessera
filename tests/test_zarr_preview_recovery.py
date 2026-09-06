"""Concurrent and interrupted creation of preview metadata."""

import errno

import numpy as np
import pytest
import zarr

import geotessera.zarr as build


@pytest.fixture(params=[False, True], ids=["local", "file-url"])
def preview_store(tmp_path, monkeypatch, request):
    monkeypatch.setattr(build, "GLOBAL_LEVEL0_H", 32)
    monkeypatch.setattr(build, "GLOBAL_LEVEL0_W", 64)
    monkeypatch.setattr(build, "GLOBAL_CHUNK", 4)
    path = tmp_path / "preview.zarr"
    store = build.StoreLocation(path.as_uri() if request.param else str(path))
    store.open_group(mode="w", zarr_format=3)
    return store


def _assert_ready(store, levels):
    group = store.open_group(mode="r", path="global_rgb")
    layout = group.attrs["multiscales"]["layout"]
    assert [entry["asset"] for entry in layout] == [str(i) for i in range(levels)]
    for level in range(levels):
        assert layout[level]["spatial:shape"] == [32 >> level, 64 >> level]
        assert group[f"{level}/rgb"].shape == (32 >> level, 64 >> level, 4)
        np.testing.assert_array_equal(group[f"{level}/band"][:], np.arange(4))


def test_preview_resumes_incomplete_levels(preview_store, monkeypatch):
    require_array = build._require_array

    def fail_level(parent, name, **kwargs):
        if parent.path == "global_rgb/1" and name == "rgb":
            raise OSError("interrupted level creation")
        return require_array(parent, name, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(build, "_require_array", fail_level)
        with pytest.raises(OSError, match="interrupted level creation"):
            build._ensure_global_store(preview_store, 3)
    assert "global_rgb/0/rgb" in preview_store.open_group(mode="r")
    assert (
        "multiscales" not in preview_store.open_group(mode="r", path="global_rgb").attrs
    )

    build._ensure_global_store(preview_store, 3)
    _assert_ready(preview_store, 3)


def test_preview_repairs_unwritten_band_coordinates(preview_store):
    build._ensure_global_store(preview_store, 3)
    group = preview_store.open_group(mode="r+", path="global_rgb")
    group["1/band"][:] = 0

    build._ensure_global_store(preview_store, 3)
    _assert_ready(preview_store, 3)


def test_preview_extends_without_losing_pixels(preview_store):
    build._ensure_global_store(preview_store, 2)
    group = preview_store.open_group(mode="r+", path="global_rgb")
    group["0/rgb"][0, 0] = [1, 2, 3, 4]
    build._ensure_global_store(preview_store, 4)
    _assert_ready(preview_store, 4)
    np.testing.assert_array_equal(group["0/rgb"][0, 0], [1, 2, 3, 4])

    build._ensure_global_store(preview_store, 2)
    _assert_ready(preview_store, 4)


@pytest.mark.parametrize("level", [0, 1])
def test_preview_preserves_incompatible_pyramid(preview_store, level):
    build._ensure_global_store(preview_store, 3)
    group = preview_store.open_group(mode="r+", path=f"global_rgb/{level}")
    del group["rgb"]
    rgb = group.create_array("rgb", shape=(1, 1, 4), dtype=np.uint8)
    rgb[:] = 42

    with pytest.raises(ValueError, match=f"global_rgb/{level}/rgb with shape"):
        build._ensure_global_store(preview_store, 3)
    np.testing.assert_array_equal(group["rgb"][:], np.full((1, 1, 4), 42))
    assert "global_rgb/2/rgb" in preview_store.open_group(mode="r")


def test_preview_publishes_complete_attributes(preview_store, monkeypatch):
    update = zarr.Group.update_attributes
    snapshots = []

    def observe(self, attributes):
        result = update(self, attributes)
        if self.path == "global_rgb":
            snapshots.append(
                dict(preview_store.open_group(mode="r", path=self.path).attrs)
            )
        return result

    monkeypatch.setattr(zarr.Group, "update_attributes", observe)
    build._ensure_global_store(preview_store, 3)
    _assert_ready(preview_store, 3)
    assert snapshots
    for snapshot in snapshots:
        assert snapshot == snapshots[-1]
        assert "multiscales" in snapshot
        assert {c["name"] for c in snapshot["zarr_conventions"]} == {
            "spatial:",
            "proj:",
            "multiscales",
        }


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_preview_retries_windows_sharing_conflicts(
    preview_store, monkeypatch, winerror
):
    update = zarr.Group.update_attributes
    attempts = 0

    def conflict(self, attributes):
        nonlocal attempts
        if self.path == "global_rgb":
            attempts += 1
            if attempts == 1:
                error = PermissionError(errno.EACCES, "Windows sharing conflict")
                error.winerror = winerror
                raise error
        return update(self, attributes)

    monkeypatch.setattr(zarr.Group, "update_attributes", conflict)
    build._ensure_global_store(preview_store, 3)
    assert attempts == 2
    _assert_ready(preview_store, 3)


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("transport")])
def test_preview_preserves_publish_errors_and_recovers(
    preview_store, monkeypatch, error
):
    attempts = 0

    def fail(self, attributes):
        nonlocal attempts
        attempts += 1
        raise error

    with monkeypatch.context() as failure:
        failure.setattr(zarr.Group, "update_attributes", fail)
        with pytest.raises(type(error), match=str(error)):
            build._ensure_global_store(preview_store, 3)
    assert attempts == 1
    build._ensure_global_store(preview_store, 3)
    _assert_ready(preview_store, 3)
