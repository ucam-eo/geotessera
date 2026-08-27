"""Checks for issue #384: scoped tile lookup and non-retryable write errors.

Run via ``uv run python tests/tile_lookup_check.py``; prints one ``ok:``
line per check and exits non-zero on the first failure.
"""

import errno
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geotessera.registry import (
    EMBEDDINGS_DIR_NAME,
    LANDMASKS_DIR_NAME,
    tile_to_embedding_paths,
    tile_to_landmask_filename,
)

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"ok: {label}")
    else:
        print(f"FAIL: {label} {detail}")
        FAILURES.append(label)


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


def test_lookup_is_scoped(tmp: Path):
    """A 2-coord request must not touch the other 200 tiles on disk."""
    from geotessera import tiles as tiles_mod

    base = tmp / "mirror"
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

    rasterio.open = counting_open
    try:
        found = tiles_mod.tiles_for_coords(base, set(wanted), 2024)
    finally:
        rasterio.open = real_open

    check("scoped lookup finds both requested tiles", set(found) == set(wanted))
    check(
        "scoped lookup opens only the requested landmasks",
        len(opened) == 2,
        f"(opened {len(opened)} files; a full scan would open 202)",
    )
    check(
        "scoped lookup touches no decoy tile",
        not any("54.05" in p for p in opened),
        f"({[p for p in opened if '54.05' in p][:3]})",
    )

    # A coord with no embedding on disk must simply be absent, not raise.
    missing = tiles_mod.tiles_for_coords(base, {(99.95, 12.05)}, 2024)
    check("absent coord yields no tile", missing == {})

    # An embedding whose landmask is absent has no spatial metadata, so it
    # is not usable and must not be returned (matching discover_npy_tiles).
    make_tile(base, 1.15, 51.05, 2024, landmask=False)
    partial = tiles_mod.tiles_for_coords(base, {(1.15, 51.05)}, 2024)
    check("embedding without landmask is not returned", partial == {})

    # Scales missing is likewise incomplete.
    make_tile(base, 2.15, 51.05, 2024)
    (
        base / EMBEDDINGS_DIR_NAME / tile_to_embedding_paths(2.15, 51.05, 2024)[1]
    ).unlink()
    check(
        "embedding without scales is not returned",
        tiles_mod.tiles_for_coords(base, {(2.15, 51.05)}, 2024) == {},
    )


def test_ensure_tiles_available_is_scoped(tmp: Path):
    """_ensure_tiles_available must not enumerate the mirror."""
    from geotessera import tiles as tiles_mod
    from geotessera.core import GeoTessera

    base = tmp / "mirror2"
    make_tile(base, 0.15, 52.05, 2024)
    for i in range(50):
        make_tile(base, round(-8.0 + i * 0.1, 2), 54.05, 2024)

    scanned = []
    real_discover = tiles_mod.discover_tiles

    def trap(directory):
        scanned.append(directory)
        return real_discover(directory)

    tiles_mod.discover_tiles = trap
    try:
        gt = GeoTessera(embeddings_dir=str(base))
        tile_map = gt._ensure_tiles_available(
            required_coords={(0.15, 52.05)}, year=2024, auto_download=False
        )
    finally:
        tiles_mod.discover_tiles = real_discover

    check(
        "_ensure_tiles_available resolves the requested tile",
        set(tile_map) == {(0.15, 52.05)},
    )
    check(
        "_ensure_tiles_available performs no full-directory scan",
        scanned == [],
        f"(discover_tiles called {len(scanned)}x)",
    )


def test_permanent_write_errors_are_not_retried(tmp: Path):
    """A read-only destination must fail immediately, not after 4 retries."""
    from geotessera import registry as reg

    calls = []
    slept = []

    def boom(url, progress_callback, cache_path):
        calls.append(url)
        raise OSError(errno.EROFS, "Read-only file system")

    real_once, real_sleep = reg._download_once, reg.time.sleep
    reg._download_once = boom
    reg.time.sleep = lambda s: slept.append(s)
    try:
        try:
            reg.download_file_to_temp("https://example.invalid/x.npy")
        except OSError as e:
            raised = e
        else:
            raised = None
    finally:
        reg._download_once, reg.time.sleep = real_once, real_sleep

    check("read-only destination raises", isinstance(raised, OSError))
    check(
        "read-only destination is attempted once, not four times",
        len(calls) == 1,
        f"(attempted {len(calls)}x)",
    )
    check("read-only destination sleeps no backoff", slept == [], f"(slept {slept})")

    # A genuinely transient OSError must still be retried.
    calls.clear()
    slept.clear()

    def flaky(url, progress_callback, cache_path):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("connection reset midway")
        return "/tmp/ok"

    reg._download_once = flaky
    reg.time.sleep = lambda s: slept.append(s)
    try:
        out = reg.download_file_to_temp("https://example.invalid/x.npy")
    finally:
        reg._download_once, reg.time.sleep = real_once, real_sleep

    check(
        "transient OSError is still retried",
        out == "/tmp/ok" and len(calls) == 3,
        f"(attempts={len(calls)})",
    )


def test_preflight_rejects_unwritable_dir(tmp: Path):
    """An unwritable destination must be rejected before any request."""
    import os
    import stat

    from geotessera.registry import UnwritableDestination, _preflight_destination

    ro = tmp / "readonly"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        # Any other exception type propagates and fails the run.
        raised = None
        try:
            _preflight_destination(ro / "sub" / "grid.npy")
        except UnwritableDestination as e:
            raised = e
        check("preflight rejects an uncreatable destination dir", raised is not None)

        # A writable destination must pass and have its dirs created.
        dest = tmp / "writable" / "deep" / "grid.npy"
        _preflight_destination(dest)
        check("preflight creates a writable destination dir", dest.parent.is_dir())
    finally:
        os.chmod(ro, 0o755)


def test_ensure_tiles_available_stops_on_unwritable_dir(tmp: Path):
    """One unwritable-dir failure must stop the batch, not repeat N times."""
    from geotessera.core import GeoTessera
    from geotessera.registry import UnwritableDestination

    base = tmp / "mirror3"
    make_tile(base, 0.15, 52.05, 2024)  # so the dir is a valid mirror

    gt = GeoTessera(embeddings_dir=str(base))
    attempts = []

    def unwritable(lon, lat, year, *a, **kw):
        attempts.append((lon, lat))
        raise UnwritableDestination(errno.EROFS, "Read-only file system")

    gt.download_tile = unwritable
    wanted = {(round(-8.0 + i * 0.1, 2), 54.05) for i in range(20)}
    try:
        gt._ensure_tiles_available(
            required_coords=wanted, year=2024, auto_download=True
        )
    except UnwritableDestination as e:
        stopped = True
        message = str(e)
    else:
        stopped, message = False, ""

    check("unwritable embeddings_dir raises UnwritableDestination", stopped)
    check(
        "the error names the tile count and how to populate the directory",
        "20 tiles are missing" in message and "geotessera download" in message,
        f"({message!r})",
    )
    check(
        "unwritable embeddings_dir stops after the first tile",
        len(attempts) == 1,
        f"(attempted {len(attempts)} of {len(wanted)} tiles)",
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_lookup_is_scoped(tmp)
        test_ensure_tiles_available_is_scoped(tmp)
        test_permanent_write_errors_are_not_retried(tmp)
        test_preflight_rejects_unwritable_dir(tmp)
        test_ensure_tiles_available_stops_on_unwritable_dir(tmp)
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
