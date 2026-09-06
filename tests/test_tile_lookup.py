"""Regression tests for scoped tile lookup and download error handling."""

import errno
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geotessera.registry import (
    EMBEDDINGS_DIR_NAME,
    LANDMASKS_DIR_NAME,
    tile_to_embedding_paths,
    tile_to_landmask_filename,
)


def make_tile(base: Path, lon: float, lat: float, year: int, landmask=True):
    """Write a minimal but valid NPY tile (+ landmask) into *base*."""
    emb_rel, scales_rel = tile_to_embedding_paths(lon, lat, year)
    emb = base / EMBEDDINGS_DIR_NAME / emb_rel
    emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb, np.zeros((4, 4, 2), dtype=np.int8))
    np.save(base / EMBEDDINGS_DIR_NAME / scales_rel, np.ones(2, dtype=np.float32))
    if landmask:
        make_landmask(base, lon, lat)


def make_landmask(base: Path, lon: float, lat: float):
    lm = base / LANDMASKS_DIR_NAME / tile_to_landmask_filename(lon, lat)
    lm.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        lm,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(lon - 0.05, lat + 0.05, 0.025, 0.025),
    ) as dst:
        dst.write(np.ones((4, 4), dtype="uint8"), 1)


def test_lookup_is_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A 2-coord request must not touch the other 200 tiles on disk."""
    from geotessera import tiles as tiles_mod

    base = tmp_path / "mirror"
    wanted = [(0.15, 52.05), (0.25, 52.15)]
    for lon, lat in wanted:
        make_tile(base, lon, lat, 2024)
    # Decoys: same year, and a second year, that must never be opened.
    decoys = [(round(-8.0 + i * 0.1, 2), 54.05) for i in range(100)]
    for lon, lat in decoys:
        make_tile(base, lon, lat, 2024)
        make_tile(base, lon, lat, 2023)

    opened = []
    real_open = rasterio.open

    def counting_open(path, *a, **kw):
        opened.append(str(path))
        return real_open(path, *a, **kw)

    monkeypatch.setattr(rasterio, "open", counting_open)
    found = tiles_mod.tiles_for_coords(base, set(wanted), 2024)

    assert set(found) == set(wanted)
    assert len(opened) == 2, f"opened {len(opened)} files; a full scan would open 202"
    assert not any("54.05" in path for path in opened)

    # A coord with no embedding on disk must simply be absent, not raise.
    missing = tiles_mod.tiles_for_coords(base, {(99.95, 12.05)}, 2024)
    assert missing == {}

    # An embedding whose landmask is absent has no spatial metadata, so it
    # is not usable and must not be returned (matching discover_npy_tiles).
    make_tile(base, 1.15, 51.05, 2024, landmask=False)
    partial = tiles_mod.tiles_for_coords(base, {(1.15, 51.05)}, 2024)
    assert partial == {}

    # Scales missing is likewise incomplete.
    make_tile(base, 2.15, 51.05, 2024)
    (
        base / EMBEDDINGS_DIR_NAME / tile_to_embedding_paths(2.15, 51.05, 2024)[1]
    ).unlink()
    assert tiles_mod.tiles_for_coords(base, {(2.15, 51.05)}, 2024) == {}


def test_ensure_tiles_available_is_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """_ensure_tiles_available must not enumerate the mirror."""
    from geotessera import tiles as tiles_mod
    from geotessera.core import GeoTessera

    base = tmp_path / "mirror2"
    make_tile(base, 0.15, 52.05, 2024)
    for i in range(50):
        make_tile(base, round(-8.0 + i * 0.1, 2), 54.05, 2024)

    scanned = []

    def trap(directory):
        scanned.append(directory)
        return {}

    monkeypatch.setattr(tiles_mod, "discover_tiles", trap)
    gt = GeoTessera(embeddings_dir=str(base))
    tile_map = gt._ensure_tiles_available(
        required_coords={(0.15, 52.05)}, year=2024, auto_download=False
    )

    assert set(tile_map) == {(0.15, 52.05)}
    assert scanned == []


def test_permanent_write_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch):
    """A read-only destination must fail immediately, not after 4 retries."""
    from geotessera import registry as reg

    calls = []
    slept = []

    def boom(url, progress_callback, cache_path):
        calls.append(url)
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(reg, "_download_once", boom)
    monkeypatch.setattr(reg.time, "sleep", slept.append)
    with pytest.raises(OSError, match="Read-only file system"):
        reg.download_file_to_temp("https://example.invalid/x.npy")

    assert len(calls) == 1
    assert slept == []

    # A genuinely transient OSError must still be retried.
    calls.clear()
    slept.clear()

    def flaky(url, progress_callback, cache_path):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("connection reset midway")
        return "/tmp/ok"

    monkeypatch.setattr(reg, "_download_once", flaky)
    out = reg.download_file_to_temp("https://example.invalid/x.npy")
    assert out == "/tmp/ok"
    assert len(calls) == 3


def test_preflight_rejects_unwritable_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unwritable destination must be rejected before any request."""
    from geotessera.registry import UnwritableDestination, _preflight_destination

    blocked_parent = tmp_path / "readonly" / "sub"
    real_mkdir = Path.mkdir

    def controlled_mkdir(path, *args, **kwargs):
        if path == blocked_parent:
            raise OSError(errno.EACCES, "Permission denied")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", controlled_mkdir)
    with pytest.raises(UnwritableDestination):
        _preflight_destination(blocked_parent / "grid.npy")

    dest = tmp_path / "writable" / "deep" / "grid.npy"
    _preflight_destination(dest)
    assert dest.parent.is_dir()


def test_ensure_tiles_available_stops_on_unwritable_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """One unwritable-dir failure must stop the batch, not repeat N times."""
    from geotessera.core import GeoTessera
    from geotessera.registry import UnwritableDestination

    base = tmp_path / "mirror3"
    make_tile(base, 0.15, 52.05, 2024)  # so the dir is a valid mirror

    gt = GeoTessera(embeddings_dir=str(base))
    attempts = []

    def unwritable(lon, lat, year, *a, **kw):
        attempts.append((lon, lat))
        raise UnwritableDestination(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(gt, "download_tile", unwritable)
    wanted = {(round(-8.0 + i * 0.1, 2), 54.05) for i in range(20)}
    with pytest.raises(UnwritableDestination) as exc_info:
        gt._ensure_tiles_available(
            required_coords=wanted, year=2024, auto_download=True
        )
    message = str(exc_info.value)
    assert "20 tiles are missing" in message
    assert "geotessera download" in message
    assert len(attempts) == 1
