"""Checks for location-transparent I/O and parallel-safe zarr fill state.

Run by tests/zarr.t.  Every check works on small in-memory arrays and a
temporary directory, using ``file://`` URLs to drive the same fsspec code
path a remote ``s3://`` store takes — so the suite stays offline and fast
while still covering the byte-range reader, the per-zone tracking files and
the advisory locks.

Prints one ``ok - <name>`` line per check and exits non-zero on the first
failure.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from geotessera import remote
from geotessera.zarr import (
    LOCK_DIR_NAME,
    REGISTRY_DIR_NAME,
    StoreLocation,
    TileInfo,
    TileSource,
    UnifiedZoneGrid,
    _acquire_zone_lock,
    _get_written_tiles,
    _record_written_tiles,
    _release_zone_lock,
    build_shard_index,
    merge_tile_registry,
    shard_coords_for_tiles,
)

TMP = Path(tempfile.mkdtemp(prefix="gt-zarr-check-"))
FAILED = []


def check(name, condition):
    if condition:
        print(f"ok - {name}")
    else:
        print(f"FAIL - {name}")
        FAILED.append(name)


def url(path):
    return f"file://{Path(path).resolve()}"


# ---------------------------------------------------------------------------
# Byte-range .npy reads
# ---------------------------------------------------------------------------

emb = np.random.default_rng(0).integers(-128, 127, size=(40, 30, 8), dtype=np.int8)
scales = np.random.default_rng(1).random((40, 30)).astype(np.float32)
np.save(TMP / "emb.npy", emb)
np.save(TMP / "scales.npy", scales)

windows = [(0, 40, 0, 30), (5, 12, 0, 30), (7, 33, 4, 19), (38, 40, 29, 30)]
for r0, r1, c0, c1 in windows:
    local = remote.read_npy_window(TMP / "emb.npy", r0, r1, c0, c1)
    over_url = remote.read_npy_window(url(TMP / "emb.npy"), r0, r1, c0, c1)
    check(
        f"npy window {r0}:{r1},{c0}:{c1} matches source",
        np.array_equal(local, emb[r0:r1, c0:c1]),
    )
    check(
        f"npy window {r0}:{r1},{c0}:{c1} identical over url",
        np.array_equal(local, over_url),
    )

check(
    "2-D npy window (scales) reads over url",
    np.array_equal(
        remote.read_npy_window(url(TMP / "scales.npy"), 3, 9, 2, 11),
        scales[3:9, 2:11],
    ),
)

# A window past the end of the array clamps rather than over-reading.
clamped = remote.read_npy_window(url(TMP / "emb.npy"), 35, 60, 0, 30)
check("npy window past EOF clamps to array height", clamped.shape[0] == 5)

# ---------------------------------------------------------------------------
# GeoTIFF window reads
# ---------------------------------------------------------------------------

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

mask = np.ones((40, 30), dtype=np.uint8)
mask[:6, :6] = 0
with rasterio.open(
    TMP / "lm.tiff",
    "w",
    driver="GTiff",
    height=40,
    width=30,
    count=1,
    dtype="uint8",
    crs="EPSG:4326",
    transform=from_origin(0, 52, 0.01, 0.01),
) as dst:
    dst.write(mask, 1)

check(
    "tiff window identical local and over url",
    np.array_equal(
        remote.read_tiff_window(TMP / "lm.tiff", 2, 10, 1, 9),
        remote.read_tiff_window(url(TMP / "lm.tiff"), 2, 10, 1, 9),
    ),
)
check(
    "tiff window matches source",
    np.array_equal(
        remote.read_tiff_window(url(TMP / "lm.tiff"), 0, 6, 0, 6),
        np.zeros((6, 6), dtype=np.uint8),
    ),
)

# ---------------------------------------------------------------------------
# StoreLocation basics
# ---------------------------------------------------------------------------

for label, loc in [
    ("local", StoreLocation(str(TMP / "store_local"))),
    ("url", StoreLocation(url(TMP / "store_url"))),
]:
    check(f"{label} store reports remote correctly", loc.is_remote == (label == "url"))
    check(f"{label} store missing object absent", not loc.exists("nope.bin"))
    loc.write_bytes(b"hello", "sub", "a.bin")
    check(f"{label} store write creates parents", loc.exists("sub", "a.bin"))
    check(f"{label} store read round-trips", loc.read_bytes("sub", "a.bin") == b"hello")
    listed = loc.listdir("sub")
    check(f"{label} store lists children", len(listed) == 1)
    # Listed entries must be usable as standalone locations — merge_tile_registry
    # reads the per-zone files straight from a listing.
    check(
        f"{label} store listing entries are readable locations",
        remote.read_bytes(listed[0], loc.storage_options) == b"hello",
    )
    loc.remove("sub", "a.bin")
    check(f"{label} store remove deletes", not loc.exists("sub", "a.bin"))
    loc.remove("sub", "a.bin")  # removing twice must not raise
    check(f"{label} store remove is idempotent", True)

# ---------------------------------------------------------------------------
# Permission-denied probes
# ---------------------------------------------------------------------------
# S3 answers 403, not 404, for a key that does not exist when the caller
# lacks s3:ListBucket. Write-scoped credentials hit this on every probe, so
# callers that can treat "don't know" as "absent" must be able to say so.


class _DenyingFS:
    def exists(self, path):
        raise PermissionError("Forbidden")

    def ls(self, path, detail=False):
        raise PermissionError("Forbidden")

    def cat_file(self, path, start=None, end=None):
        raise PermissionError("Forbidden")


# The tolerant paths log a warning by design; silence it for the run.
logging.getLogger("geotessera.remote").setLevel(logging.ERROR)

_real_get_fs = remote.get_fs
remote.get_fs = lambda loc, so=None: (
    _DenyingFS() if remote.is_url(loc) else _real_get_fs(loc, so)
)
try:
    denied = "s3://bucket/store.zarr/zarr.json"
    try:
        remote.exists(denied)
        check("denied probe raises by default", False)
    except PermissionError:
        check("denied probe raises by default", True)

    check(
        "denied probe can assume absent",
        remote.exists(denied, on_denied=False) is False,
    )
    check(
        "denied probe can assume present", remote.exists(denied, on_denied=True) is True
    )

    try:
        remote.listdir("s3://bucket/store.zarr.build/_registry")
        check("denied listing raises by default", False)
    except PermissionError:
        check("denied listing raises by default", True)

    check(
        "denied listing can fall back to empty",
        remote.listdir("s3://bucket/x", on_denied=[]) == [],
    )
    # A registry read that is refused must look like "no registry yet", not
    # crash a fill that is otherwise fine.
    from geotessera.zarr import _read_parquet_at  # noqa: E402

    check(
        "denied registry read reads as absent",
        _read_parquet_at(StoreLocation("s3://bucket/store.zarr.build"), "x.parquet")
        is None,
    )
finally:
    remote.get_fs = _real_get_fs


# ---------------------------------------------------------------------------
# Per-zone ingestion registry
# ---------------------------------------------------------------------------


def tile(lon, lat, zone=31):
    return TileInfo(
        lon=lon,
        lat=lat,
        year=2024,
        epsg=32600 + zone,
        transform=None,
        height=10,
        width=10,
        landmask_path="",
        embedding_path="",
        scales_path="",
    )


store = StoreLocation(str(TMP / "reg_store"))
check("registry empty before any fill", _get_written_tiles(store, 2024, 31) == set())

_record_written_tiles(store, [tile(0.05, 52.05), tile(0.15, 52.05)], 2024, 31)
check(
    "registry records this zone/year",
    _get_written_tiles(store, 2024, 31) == {(0.05, 52.05), (0.15, 52.05)},
)
check("registry isolates other zones", _get_written_tiles(store, 2024, 30) == set())
check("registry isolates other years", _get_written_tiles(store, 2023, 31) == set())

# A second zone writes its own object — the file a sibling job owns is
# untouched, which is what makes concurrent zone fills safe.
_record_written_tiles(store, [tile(-0.05, 52.05, zone=30)], 2024, 30)
parts = sorted(Path(p).name for p in store.state.listdir(REGISTRY_DIR_NAME))
check(
    "one registry object per zone/year",
    parts == ["utm30_2024.parquet", "utm31_2024.parquet"],
)

# Appending to a zone keeps earlier rows and dedupes repeats.
_record_written_tiles(store, [tile(0.15, 52.05), tile(0.25, 52.05)], 2024, 31)
check(
    "registry append keeps and dedupes",
    _get_written_tiles(store, 2024, 31)
    == {(0.05, 52.05), (0.15, 52.05), (0.25, 52.05)},
)

n = merge_tile_registry(store)
check("merge folds every zone into the root registry", n == 4)
check(
    "merged registry lands in the state sibling",
    store.state.exists("_registry.parquet"),
)
check("nothing written into the store itself", not store.exists("_registry.parquet"))
check("state sibling sits next to the store", store.state.url == store.url + ".build")

# The same merge must work through a URL location, since that is how a
# remote store is finished after a sweep.
url_store = StoreLocation(url(TMP / "reg_store_url"))
for zone, tiles in [(31, [tile(0.05, 52.05)]), (30, [tile(-0.05, 52.05, zone=30)])]:
    _record_written_tiles(url_store, tiles, 2024, zone)
check("merge over a url location", merge_tile_registry(url_store) == 2)
check(
    "url store resumes from its own registry",
    _get_written_tiles(url_store, 2024, 31) == {(0.05, 52.05)},
)

# Stores built before the split kept the registry inside the hierarchy;
# it must still be read so those resume correctly.
legacy = StoreLocation(str(TMP / "legacy_store"))
legacy.write_bytes(store.state.read_bytes("_registry.parquet"), "_registry.parquet")
check(
    "legacy root registry still resumes",
    _get_written_tiles(legacy, 2024, 31)
    == {(0.05, 52.05), (0.15, 52.05), (0.25, 52.05)},
)

# ---------------------------------------------------------------------------
# Advisory zone locks
# ---------------------------------------------------------------------------

lock_store = StoreLocation(str(TMP / "lock_store"))
_acquire_zone_lock(lock_store, 31, 2024)
check(
    "lock object created in the state sibling",
    lock_store.state.exists(LOCK_DIR_NAME, "utm31_2024.json"),
)
check("no lock object inside the store", not lock_store.exists(LOCK_DIR_NAME))

try:
    _acquire_zone_lock(lock_store, 31, 2024)
    check("second acquire refused", False)
except RuntimeError as e:
    check("second acquire refused", "locked by" in str(e))

# A different zone or year is a different lock, so sibling jobs proceed.
_acquire_zone_lock(lock_store, 30, 2024)
_acquire_zone_lock(lock_store, 31, 2023)
check("sibling zone/year locks independent", True)

_acquire_zone_lock(lock_store, 31, 2024, force=True)
check("force takes over a stale lock", True)

_release_zone_lock(lock_store, 31, 2024)
check(
    "release removes the lock", not lock_store.exists(LOCK_DIR_NAME, "utm31_2024.json")
)

# ---------------------------------------------------------------------------
# Shard index: rewriting a shard must carry its already-written neighbours
# ---------------------------------------------------------------------------

from rasterio.transform import Affine  # noqa: E402

grid = UnifiedZoneGrid(
    zone=31,
    years=[2024],
    canonical_epsg=32631,
    origin_x=0.0,
    origin_y=100000.0,
    width_px=8192,
    height_px=8192,
)


def placed(lon, x_off):
    t = tile(lon, 52.05)
    t.transform = Affine(10.0, 0.0, x_off, 0.0, -10.0, 100000.0)
    t.height, t.width = 1000, 1000
    return t


old = placed(0.05, 0.0)  # shard (0, 0)
new = placed(0.15, 10000.0)  # shard (0, 0), adjacent
far = placed(0.95, 60000.0)  # shard (0, 1)

check("shard coords derived per tile", shard_coords_for_tiles([old], grid) == {(0, 0)})
check(
    "distant tile lands in another shard",
    shard_coords_for_tiles([far], grid) == {(0, 1)},
)

touched = shard_coords_for_tiles([new], grid)
specs = build_shard_index([old, new, far], grid, 0, restrict_to=touched)
check("only touched shards are rebuilt", [(s.sr, s.sc) for s in specs] == [(0, 0)])
check(
    "rebuilt shard carries the already-written neighbour",
    len(specs[0].tiles) == 2,
)

unrestricted = build_shard_index([old, new, far], grid, 0)
check("unrestricted index covers every shard", len(unrestricted) == 2)

# ---------------------------------------------------------------------------
# Extending the time axis
# ---------------------------------------------------------------------------

import zarr  # noqa: E402
from zarr.codecs import BloscCodec  # noqa: E402

from geotessera.zarr import extend_store  # noqa: E402

ext = StoreLocation(str(TMP / "ext_store"))
root = zarr.open_group(ext.url, mode="w", zarr_format=3)
for zname in ("utm30", "utm31"):
    grp = root.create_group(zname)
    grp.create_array(
        "embeddings",
        shape=(1, 4, 32, 32),
        chunks=(1, 4, 16, 16),
        shards=(1, 4, 32, 32),
        dtype=np.int8,
        fill_value=np.int8(0),
        compressors=BloscCodec(cname="zstd"),
        dimension_names=["time", "band", "y", "x"],
    )
    grp.create_array(
        "scales",
        shape=(1, 32, 32),
        chunks=(1, 16, 16),
        shards=(1, 32, 32),
        dtype=np.float32,
        fill_value=np.float32("inf"),
        dimension_names=["time", "y", "x"],
    )
    t = grp.create_array("time", shape=(1,), dtype=np.int32, dimension_names=["time"])
    t[:] = [2024]
# Existing data that must survive the resize untouched.
root["utm31"]["embeddings"][0] = 7
root["utm31"]["scales"][0] = 0.5

from geotessera.zarr import (  # noqa: E402
    STRETCH_ARRAY_NAMES,
    create_stretch_arrays,
)

# A pre-stats store must be refused (extending it would leave the stretch
# arrays permanently short) and pointed at the backfill.
try:
    extend_store(ext, [2026])
    check("extend refuses a store without stretch arrays", False)
except ValueError as e:
    check(
        "extend refuses a store without stretch arrays",
        "backfill-stretch-stats" in str(e),
    )

for zname in ("utm30", "utm31"):
    create_stretch_arrays(root[zname], n_years=1, k=50)

check("extend adds the year to every zone", extend_store(ext, [2026]) == 2)
check(
    "extend grows the stretch arrays too",
    all(
        root[z][a].shape[0] == 2
        for z in ("utm30", "utm31")
        for a in STRETCH_ARRAY_NAMES
    ),
)

g31 = zarr.open_group(ext.url, mode="r", use_consolidated=False)["utm31"]
check("time axis grew", [int(v) for v in g31["time"][:]] == [2024, 2026])
check("arrays grew along time", g31["embeddings"].shape[0] == 2)
check(
    "existing year untouched by the resize",
    bool((np.asarray(g31["embeddings"][0]) == 7).all())
    and bool((np.asarray(g31["scales"][0]) == 0.5).all()),
)
check(
    "new year reads as freshly initialised",
    bool((np.asarray(g31["embeddings"][1]) == 0).all())
    and bool(np.isinf(np.asarray(g31["scales"][1])).all()),
)

check("extend is idempotent", extend_store(ext, [2026]) == 0)

try:
    extend_store(ext, [2020])
    check("inserting an earlier year refused", False)
except ValueError as e:
    check("inserting an earlier year refused", "only be appended" in str(e))

ext.state.write_bytes(b"{}", LOCK_DIR_NAME, "utm30_2026.json")
try:
    extend_store(ext, [2027])
    check("extend refuses while a fill lock is held", False)
except RuntimeError as e:
    check("extend refuses while a fill lock is held", "fill lock" in str(e))
check(
    "extend --force overrides a stale lock", extend_store(ext, [2027], force=True) == 2
)

# ---------------------------------------------------------------------------
# Stretch statistics
# ---------------------------------------------------------------------------

from geotessera.zarr import (  # noqa: E402
    merge_stretch_samples,
    shard_stretch_stats,
    update_zone_stretch_stats,
    weighted_percentile,
)

nprng = np.random.default_rng(11)
B = 128
semb = nprng.integers(-128, 127, (B, 96, 96), dtype=np.int8)
ssc = (nprng.random((96, 96)).astype(np.float32) * 0.01 + 0.001)
ssc[:20, :20] = np.nan
ssc[80:, 80:] = np.inf

sst = shard_stretch_stats(semb, ssc, sample_cap=200, seed=5)
svalid = np.isfinite(ssc)
sx = semb.reshape(B, -1)[:, svalid.ravel()].astype(np.float64) * ssc.ravel()[
    svalid.ravel()
]
check("stats count exact", sst["n"] == int(svalid.sum()))
check(
    "stats sum matches population",
    bool(np.allclose(sst["sum"], sx.sum(1), rtol=1e-5)),
)
_truth = sx @ sx.T
check(
    "stats product matches population",
    float(np.abs(sst["prod"] - _truth).max()) < 1e-5 * float(np.abs(_truth).max()),
)
check(
    "sampled pixels are valid pixels",
    bool(np.isfinite(sst["sample_scales"]).all()),
)

ha = shard_stretch_stats(semb[:, :48, :], ssc[:48, :], 50, seed=1)
hb = shard_stretch_stats(semb[:, 48:, :], ssc[48:, :], 50, seed=2)
check("stats additive across shards", ha["n"] + hb["n"] == sst["n"])
check(
    "sums additive across shards",
    bool(np.allclose(ha["sum"] + hb["sum"], sst["sum"], rtol=1e-5)),
)

heavy = (np.ones((500, B), np.int8), np.ones(500, np.float32), 50.0)
light = (np.zeros((500, B), np.int8), np.zeros(500, np.float32), 1.0)
me, ms = merge_stretch_samples([heavy, light], 300, seed=4)
check("merge respects capacity", len(me) == 300)
check(
    "merge favours high-weight rows",
    float((me[:, 0] == 1).mean()) > 0.8,
)

vv = nprng.normal(size=4000)
check(
    "weighted percentile matches numpy under uniform weights",
    float(
        np.abs(
            weighted_percentile(vv, np.ones(4000), np.array([2.0, 50.0, 98.0]))
            - np.percentile(vv, [2, 50, 98])
        ).max()
    )
    < 0.02,
)

# Zone-array round trip: create, fold twice, contents accumulate.
import zarr  # noqa: E402

zs = zarr.open_group(str(TMP / "stats.zarr"), mode="w", zarr_format=3)
create_stretch_arrays(zs, n_years=2, k=100)
check(
    "stretch arrays created",
    all(n in zs for n in STRETCH_ARRAY_NAMES),
)
cand = [(sst["sample_emb"], sst["sample_scales"], sst["sample_weight"])]
update_zone_stretch_stats(zs, 0, sst["n"], sst["sum"], sst["prod"], cand, seed=1)
update_zone_stretch_stats(zs, 0, sst["n"], sst["sum"], sst["prod"], cand, seed=2)
check("stats fold additively", int(zs["stretch_stats_count"][0]) == 2 * sst["n"])
check(
    "sample capacity bounded",
    int(zs["stretch_sample_count"][0]) <= 100,
)
check("other year untouched", int(zs["stretch_stats_count"][1]) == 0)

# ---------------------------------------------------------------------------
# Storage options and source layout
# ---------------------------------------------------------------------------

for var in ("AWS_ENDPOINT_URL", "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
    os.environ.pop(var, None)

check("no options when nothing configured", remote.build_storage_options() is None)
check(
    "explicit endpoint and anon are passed through",
    remote.build_storage_options(endpoint_url="https://s3.example", anon=True)
    == {"endpoint_url": "https://s3.example", "anon": True},
)

os.environ["AWS_ENDPOINT_URL"] = "https://from-env.example"
check(
    "endpoint falls back to the environment",
    remote.build_storage_options()["endpoint_url"] == "https://from-env.example",
)
check(
    "explicit endpoint beats the environment",
    remote.build_storage_options(endpoint_url="https://explicit.example")[
        "endpoint_url"
    ]
    == "https://explicit.example",
)
os.environ.pop("AWS_ENDPOINT_URL")

check(
    "canned acl becomes an s3fs write kwarg",
    remote.build_storage_options(acl="bucket-owner-full-control")[
        "s3_additional_kwargs"
    ]
    == {"ACL": "bucket-owner-full-control"},
)
try:
    remote.build_storage_options(acl="not-an-acl")
    check("bad acl rejected up front", False)
except ValueError:
    check("bad acl rejected up front", True)

src = TileSource.for_url("s3://bucket/tessera", "v1", {"anon": True})
e, s = src.embedding_locations(0.05, 52.05, 2024)
check(
    "embedding location follows the published layout",
    e == "s3://bucket/tessera/npy/v1/2024/grid_0.05_52.05/grid_0.05_52.05.npy",
)
check(
    "scales location follows the published layout",
    s == "s3://bucket/tessera/npy/v1/2024/grid_0.05_52.05/grid_0.05_52.05_scales.npy",
)
check(
    "landmask location follows the published layout",
    src.landmask_location(0.05, 52.05)
    == "s3://bucket/tessera/landmasks/v1/grid_0.05_52.05.tiff",
)
check("url source reports remote", src.is_remote)

import shutil  # noqa: E402

shutil.rmtree(TMP, ignore_errors=True)

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall checks passed")
