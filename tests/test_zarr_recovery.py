"""Recovery after failed Zarr writes, using small real stores."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from rasterio.transform import from_origin

import geotessera.zarr as build


@pytest.fixture(params=[False, True], ids=["local", "file-url"])
def tiny_store(tmp_path, monkeypatch, request):
    monkeypatch.setattr(build, "SHARD_SIZE", 4)
    path = tmp_path / "store.zarr"
    location = build.StoreLocation(path.as_uri() if request.param else str(path))
    root = location.open_group(mode="w", zarr_format=3)
    group = root.create_group("utm31")
    transform = from_origin(500_000, 5_600_000, 10, 10)
    group.attrs.update(
        {
            "spatial:transform": list(transform)[:6],
            "spatial:shape": [4, 8],
            "proj:code": "EPSG:32631",
        }
    )
    group.create_array(
        "embeddings",
        shape=(1, 128, 4, 8),
        chunks=(1, 128, 4, 4),
        dtype="i1",
        fill_value=0,
    )
    group.create_array(
        "scales",
        shape=(1, 4, 8),
        chunks=(1, 4, 4),
        dtype="f4",
        fill_value=np.inf,
    )
    group.create_array("time", data=np.array([2024], dtype="i4"), chunks=(1,))
    build.create_stretch_arrays(group, 1, 32, 1, 2)
    embeddings = tmp_path / "embeddings.npy"
    scales = tmp_path / "scales.npy"
    np.save(embeddings, np.ones((4, 8, 128), dtype="i1"))
    np.save(scales, np.full((4, 8), 0.25, dtype="f4"))
    tile = build.TileInfo(
        3.05,
        50.55,
        2024,
        32631,
        transform,
        4,
        8,
        "",
        str(embeddings),
        str(scales),
    )
    grid = build.UnifiedZoneGrid(31, [2024], 32631, 500_000, 5_600_000, 8, 4)
    specs = build.build_shard_index([tile], grid, 0)
    monkeypatch.setattr(build, "store_uses_landmask", lambda store: False)
    monkeypatch.setattr(
        build, "gather_tile_infos", lambda *args, **kwargs: {31: [tile]}
    )

    def serial_writes(**kwargs):
        """Use the real workers while replacing only process scheduling."""
        written, failed, statistics = 0, set(), []
        for spec in kwargs["shard_specs"]:
            try:
                result = build._write_one_shard(
                    spec,
                    group,
                    sample_cap=kwargs["sample_cap"],
                    depths=kwargs["depths"],
                    use_landmask=False,
                )
                written += bool(result)
                if isinstance(result, dict):
                    statistics.append(result)
            except OSError:
                failed.add((spec.sr, spec.sc))
        monkeypatch.setattr(build, "_worker_store", group)
        monkeypatch.setattr(build, "_worker_time_index", kwargs["time_index"])
        monkeypatch.setattr(build, "_worker_sample_cap", kwargs["sample_cap"])
        for coord in kwargs["stats_coords"]:
            try:
                statistics.append(build._stats_catchup_worker(coord))
            except OSError:
                failed.add(coord)
        return written, failed, statistics

    monkeypatch.setattr(build, "_write_shards", serial_writes)
    return SimpleNamespace(
        location=location,
        group=group,
        specs=specs,
        embeddings=embeddings,
        scales=scales,
    )


def _fill(store, **kwargs):
    return build.fill_store(
        object(), store.location, year=2024, zones=[31], workers=1, **kwargs
    )


def _fail_array_write(monkeypatch, array_name):
    original = zarr.Array.__setitem__

    def fail(self, selection, value):
        if self.path == f"utm31/{array_name}":
            raise OSError(f"injected {array_name} failure")
        return original(self, selection, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", fail)


@pytest.mark.parametrize("array_name", ["scales", "embeddings"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["initial", "rewrite"])
def test_failed_shard_write_is_retried(tiny_store, monkeypatch, array_name, rewrite):
    if rewrite:
        assert _fill(tiny_store) == 2
        np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    with monkeypatch.context() as failure:
        _fail_array_write(failure, array_name)
        with pytest.raises(RuntimeError, match="2 shard\\(s\\) failed"):
            _fill(tiny_store, skip_existing_shards=not rewrite)
    assert (
        build._existing_shards(tiny_store.location, "utm31", 0, {(0, 0), (0, 1)})
        == set()
    )
    assert _fill(tiny_store) == 2
    np.testing.assert_array_equal(
        tiny_store.group["scales"][:], 0.5 if rewrite else 0.25
    )
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_resume_repairs_legacy_embedding_without_scales(tiny_store):
    tiny_store.group["embeddings"][0, :, :, :4] = 1
    assert build._existing_shards(tiny_store.location, "utm31", 0, {(0, 0)}) == set()
    assert _fill(tiny_store) == 2
    assert np.isfinite(tiny_store.group["scales"][:]).all()


def test_resume_checks_both_arrays_without_listing(tiny_store, monkeypatch):
    _fill(tiny_store)
    monkeypatch.setattr(
        tiny_store.location,
        "listdir",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("no listing")),
    )
    assert build._existing_shards(
        tiny_store.location, "utm31", 0, {(0, 0), (0, 1)}
    ) == {(0, 0), (0, 1)}


def test_extend_recovers_failed_coordinate_write(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        _fail_array_write(failure, "time")
        with pytest.raises(OSError, match="time failure"):
            build.extend_store(tiny_store.location, [2025, 2026], consolidate=False)
    assert tiny_store.group["time"][:].tolist() == [2024, 0, 0]
    with pytest.raises(ValueError, match="re-run zarr-extend"):
        _fill(tiny_store)
    assert build.extend_store(tiny_store.location, [2025, 2026], consolidate=False) == 1
    assert tiny_store.group["time"][:].tolist() == [2024, 2025, 2026]
    assert build._PENDING_TIME_ATTR not in tiny_store.group.attrs
    for name in ("embeddings", "scales", *build.STRETCH_ARRAY_NAMES):
        assert tiny_store.group[name].shape[0] == 3


def test_extend_repairs_legacy_trailing_zeros(tiny_store):
    tiny_store.group["time"].resize((2,))
    build.extend_store(tiny_store.location, [2025], consolidate=False)
    assert tiny_store.group["time"][:].tolist() == [2024, 2025]


def test_extend_retries_failed_consolidation(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        failure.setattr(
            build,
            "consolidate_store",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("consolidation")),
        )
        with pytest.raises(OSError, match="consolidation"):
            build.extend_store(tiny_store.location, [2025])
    assert build.extend_store(tiny_store.location, [2025]) == 0
    root = zarr.open_group(
        tiny_store.location.as_zarr_store(), mode="r", use_consolidated=True
    )
    assert root["utm31/embeddings"].shape[0] == 2


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), OSError("transport"), ValueError("metadata")]
)
def test_fill_propagates_zone_access_errors(tiny_store, monkeypatch, error):
    original = tiny_store.location.open_group

    def fail_zone(*args, **kwargs):
        if kwargs.get("path") == "utm31":
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_store.location, "open_group", fail_zone)
    with pytest.raises(type(error), match=str(error)):
        _fill(tiny_store)


def test_rewritten_shards_replace_stretch_statistics(tiny_store):
    assert _fill(tiny_store) == 2
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    assert _fill(tiny_store, skip_existing_shards=False) == 2
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 16.0)
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_prod"][0], 8.0)
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_rewrite_without_stats_invalidates_then_rebuilds(tiny_store):
    _fill(tiny_store)
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    _fill(tiny_store, skip_existing_shards=False, collect_stretch_stats=False)
    assert int(tiny_store.group["stretch_stats_count"][0]) == -1
    with pytest.raises(RuntimeError, match="statistics are incomplete"):
        build.compute_stretch_from_stats(tiny_store.location, 2024, persist=False)
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 16.0)


def test_failed_stats_update_rebuilds_on_retry(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        _fail_array_write(failure, "stretch_stats_sum")
        with pytest.raises(OSError, match="stretch_stats_sum failure"):
            _fill(tiny_store)
    assert int(tiny_store.group["stretch_stats_count"][0]) == -1
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 8.0)


def test_failed_stats_catchup_invalidates_then_rebuilds(tiny_store, monkeypatch):
    _fill(tiny_store, collect_stretch_stats=False)
    with monkeypatch.context() as failure:

        def fail_catchup(coord):
            raise OSError("injected statistics read failure")

        failure.setattr(build, "_stats_catchup_worker", fail_catchup)
        with pytest.raises(RuntimeError, match="2 shard\\(s\\) failed"):
            _fill(tiny_store)
    assert int(tiny_store.group["stretch_stats_count"][0]) == -1
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_worker_pool_reports_failed_stats_catchup(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    def thread_pool(**kwargs):
        return ThreadPoolExecutor(max_workers=1)

    def fail_catchup(coord):
        raise OSError("injected statistics read failure")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", thread_pool)
    monkeypatch.setattr(build, "_stats_catchup_worker", fail_catchup)
    written, failed, statistics = build._write_shards(
        build.StoreLocation(str(tmp_path)),
        "utm31",
        [],
        workers=1,
        source_options=None,
        label="test",
        console=None,
        stats_coords=[(0, 0)],
    )
    assert (written, failed, statistics) == (0, {(0, 0)}, [])
