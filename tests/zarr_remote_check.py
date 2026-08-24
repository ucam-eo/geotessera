"""Checks for location-transparent I/O and parallel-safe zarr fill state.

Run by tests/zarr.t.  Every check works on small in-memory arrays and a
temporary directory, using ``file://`` URLs to drive the same fsspec code
path a remote ``s3://`` store takes — so the suite stays offline and fast
while still covering the byte-range reader and the legacy ingestion-registry
merge.

Prints one ``ok - <name>`` line per check and exits non-zero on the first
failure.
"""

import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from geotessera import remote
from geotessera.zarr import (
    REGISTRY_DIR_NAME,
    StoreLocation,
    TileInfo,
    TileSource,
    UnifiedZoneGrid,
    build_shard_index,
    load_merged_registry,
    merge_tile_registry,
    shard_coords_for_tiles,
)

# Concurrent group creation leaves transient ".partial" temporaries visible to
# whoever enumerates the hierarchy at that instant. They are cleaned up (this
# suite asserts the pyramid ends up complete), and every CLI entry point
# filters the same warning; matching that here keeps the output stable.
warnings.filterwarnings("ignore", message="Object at .* is not recognized")

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
# Legacy ingestion-registry merge
# ---------------------------------------------------------------------------
# Fills are stateless now, but pre-stateless builds left per-(zone, year)
# tracking parts in the state sibling, and older stores still a single
# ``_registry.parquet`` at the store root. Consolidation must fold both into
# the merged registry, so write what those builds wrote and check the merge.


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


def write_legacy_part(store, coords, year, zone):
    """Write a per-zone tracking part the way a pre-stateless build did."""
    import io

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        [
            {
                "year": np.int32(year),
                "zone": np.int32(zone),
                "tile_lon": lon,
                "tile_lat": lat,
                "written_at": pd.Timestamp.now(tz="UTC"),
                "geometry": Point(lon, lat),
            }
            for lon, lat in coords
        ],
        crs="EPSG:4326",
    )
    buf = io.BytesIO()
    gdf.to_parquet(buf)
    store.state.write_bytes(
        buf.getvalue(), REGISTRY_DIR_NAME, f"utm{zone:02d}_{year}.parquet"
    )


def written(store, year, zone):
    merged = load_merged_registry(store)
    if merged is None or merged.empty:
        return set()
    sub = merged[(merged["year"] == year) & (merged["zone"] == zone)]
    return set(zip(sub["tile_lon"], sub["tile_lat"]))


store = StoreLocation(str(TMP / "reg_store"))
check("no merged registry before consolidation", written(store, 2024, 31) == set())

write_legacy_part(store, [(0.05, 52.05), (0.15, 52.05)], 2024, 31)
write_legacy_part(store, [(-0.05, 52.05)], 2024, 30)
parts = sorted(Path(p).name for p in store.state.listdir(REGISTRY_DIR_NAME))
check(
    "one registry object per zone/year",
    parts == ["utm30_2024.parquet", "utm31_2024.parquet"],
)

n = merge_tile_registry(store)
check("merge folds every zone into the root registry", n == 3)
check(
    "merged registry lands in the state sibling",
    store.state.exists("_registry.parquet"),
)
check("nothing written into the store itself", not store.exists("_registry.parquet"))
check("state sibling sits next to the store", store.state.url == store.url + ".build")
check(
    "merged registry keeps this zone/year",
    written(store, 2024, 31) == {(0.05, 52.05), (0.15, 52.05)},
)
check(
    "merged registry keeps sibling zones apart",
    written(store, 2024, 30) == {(-0.05, 52.05)},
)

# The same merge must work through a URL location, since that is how a
# remote store is finished after a sweep.
url_store = StoreLocation(url(TMP / "reg_store_url"))
write_legacy_part(url_store, [(0.05, 52.05)], 2024, 31)
write_legacy_part(url_store, [(-0.05, 52.05)], 2024, 30)
check("merge over a url location", merge_tile_registry(url_store) == 2)
check(
    "merged registry readable over a url location",
    written(url_store, 2024, 31) == {(0.05, 52.05)},
)

# Stores built before the split kept the registry inside the hierarchy;
# it must still be read so those records survive a merge too.
legacy = StoreLocation(str(TMP / "legacy_store"))
legacy.write_bytes(store.state.read_bytes("_registry.parquet"), "_registry.parquet")
check(
    "legacy root registry is still read",
    written(legacy, 2024, 31) == {(0.05, 52.05), (0.15, 52.05)},
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
    create_stretch_arrays(root[zname], n_years=1, k=50, n_shard_rows=1, n_shard_cols=1)

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

check("extend appends a further year", extend_store(ext, [2027]) == 2)

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
ssc = nprng.random((96, 96)).astype(np.float32) * 0.01 + 0.001
ssc[:20, :20] = np.nan
ssc[80:, 80:] = np.inf

sst = shard_stretch_stats(semb, ssc, sample_cap=200, seed=5)
svalid = np.isfinite(ssc)
sx = (
    semb.reshape(B, -1)[:, svalid.ravel()].astype(np.float64)
    * ssc.ravel()[svalid.ravel()]
)
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

# Sentinel scales: huge-finite nodata values must not count as data. Some
# published scales files carry ~FLT_MAX sentinels that pass isfinite() —
# they inflated N by 100x and overflowed the product sums to inf.
jemb = nprng.integers(-128, 127, (B, 64, 64), dtype=np.int8)
jsc = nprng.random((64, 64)).astype(np.float32) * 0.01 + 0.001
jsc[:8, :8] = np.float32(3.4e38)  # FLT_MAX-style sentinel
jsc[8, 8] = np.float32(0.0)  # degenerate
jsc[9, 9] = np.float32(-1.0)  # negative
jst = shard_stretch_stats(jemb, jsc, sample_cap=100, seed=3)
check("sentinel scales excluded from N", jst["n"] == 64 * 64 - 64 - 2)
check("junk-free sums stay finite", bool(np.isfinite(jst["sum"]).all()))
check("junk-free products stay finite", bool(np.isfinite(jst["prod"]).all()))
check(
    "sampled scales all plausible",
    float(jst["sample_scales"].max()) < 1.0,
)

# Zone-array round trip: create, fold twice, contents accumulate.
import zarr  # noqa: E402

zs = zarr.open_group(str(TMP / "stats.zarr"), mode="w", zarr_format=3)
create_stretch_arrays(zs, n_years=2, k=100, n_shard_rows=2, n_shard_cols=3)
check(
    "stretch arrays created",
    all(n in zs for n in STRETCH_ARRAY_NAMES),
)
cand = [(sst["sample_emb"], sst["sample_scales"], sst["sample_weight"])]
update_zone_stretch_stats(
    zs, 0, sst["n"], sst["sum"], sst["prod"], cand, seen_coords=[(0, 1)], seed=1
)
update_zone_stretch_stats(
    zs, 0, sst["n"], sst["sum"], sst["prod"], cand, seen_coords=[(1, 2)], seed=2
)
check("stats fold additively", int(zs["stretch_stats_count"][0]) == 2 * sst["n"])
mask0 = np.asarray(zs["stretch_stats_shards"][0])
check(
    "coverage mask accumulates seen shards",
    mask0[0, 1] == 1 and mask0[1, 2] == 1 and int(mask0.sum()) == 2,
)
check(
    "sample capacity bounded",
    int(zs["stretch_sample_count"][0]) <= 100,
)
check("other year untouched", int(zs["stretch_stats_count"][1]) == 0)

# ---------------------------------------------------------------------------
# Preview work list from shard footprints
# ---------------------------------------------------------------------------

from geotessera.zarr import GLOBAL_CHUNK, _chunks_for_shards  # noqa: E402

# One shard at UTM zone 31's origin near (0.0E, ~0.9N): its footprint is
# ~41 km, so the candidate set must be a handful of chunks, not a
# bounding-box sweep.
transform31 = [10.0, 0.0, 166021.44, 0.0, -10.0, 100000.0]
chunks, regions = _chunks_for_shards({(0, 0)}, 32631, transform31, (8192, 8192))
check("shard footprint yields a small chunk set", 0 < len(chunks) < 200)
check("an ordinary footprint needs one region", len(regions) == 1)
r0, r1, c0, c1 = regions[0]
check(
    "footprint bounds are chunk-aligned and ordered",
    r0 < r1 and c0 < c1 and r0 % GLOBAL_CHUNK == 0 and c1 % GLOBAL_CHUNK == 0,
)
check(
    "empty shard set yields no work",
    _chunks_for_shards(set(), 32631, transform31, (8192, 8192))[0] == set(),
)

# Two far-apart shards must not fill the space between them: the union is
# exactly the two footprints, despite a row span of >1000 chunks.
near, _ = _chunks_for_shards({(0, 0)}, 32631, transform31, (1011712, 8192))
far, _ = _chunks_for_shards({(200, 0)}, 32631, transform31, (1011712, 8192))
both, _ = _chunks_for_shards({(0, 0), (200, 0)}, 32631, transform31, (1011712, 8192))
rows = {c[0] for c in both}
check(
    "sparse shards keep a sparse work list",
    both == near | far and max(rows) - min(rows) > 1000,
)

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

# The npy/ tree is keyed by dataset (version *and* variant); landmasks are
# keyed by version alone. Deriving the npy path from the version silently
# points a fill at a prefix that does not exist — every shard then fails with
# "The specified key does not exist", which is how the v1.1 conversion died.
# v1 hides the bug because its directory predates the variant-suffix scheme.
from geotessera.registry import dataset_path  # noqa: E402

v11 = TileSource.for_url(
    "s3://bucket/tessera", "v1.1", {"anon": True},
    dataset_dir=dataset_path("1.1", "cambridge"),
)
e11, s11 = v11.embedding_locations(0.05, 52.05, 2024)
check(
    "variant-suffixed dataset uses its own npy directory",
    e11 == "s3://bucket/tessera/npy/v1.1-cam/2024/grid_0.05_52.05/grid_0.05_52.05.npy"
    and s11.endswith("npy/v1.1-cam/2024/grid_0.05_52.05/grid_0.05_52.05_scales.npy"),
)
check(
    "landmasks stay keyed by version, not dataset",
    v11.landmask_location(0.05, 52.05)
    == "s3://bucket/tessera/landmasks/v1.1/grid_0.05_52.05.tiff",
)
check(
    "v1 is unaffected (dataset dir equals version path)",
    TileSource.for_url(
        "s3://bucket/tessera", "v1", {"anon": True},
        dataset_dir=dataset_path("1.0", "vultr"),
    ).embedding_locations(0.05, 52.05, 2024)[0]
    == "s3://bucket/tessera/npy/v1/2024/grid_0.05_52.05/grid_0.05_52.05.npy",
)


# ---------------------------------------------------------------------------
# Preview work list: antimeridian footprints must not span the globe
# ---------------------------------------------------------------------------
# A shard straddling 180 samples corners near -180 and +180. Taking the naive
# min/max of those makes it claim every chunk column at its latitude, which
# for utm01/utm60 enqueued ~2.3M bogus chunks.

from geotessera.zarr import (  # noqa: E402
    GLOBAL_CHUNK,
    GLOBAL_LEVEL0_W,
    _chunks_for_shards,
)

N_COLS = GLOBAL_LEVEL0_W // GLOBAL_CHUNK


def _shard_cols(epsg, origin_x, origin_y):
    """Columns and coarsening regions of one 4096px shard at a zone origin."""
    chunks, regions = _chunks_for_shards(
        {(0, 0)},
        epsg,
        [10.0, 0.0, origin_x, 0.0, -10.0, origin_y],
        (4096, 4096),
    )
    cols = {c for _r, c in chunks}
    return cols, (max(cols) - min(cols)) if cols else 0, regions, chunks


# EPSG:32660 has central meridian 177E, so easting ~834000 sits on the
# antimeridian at the equator; this shard straddles it.
wrap_cols, wrap_span, wrap_regions, wrap_chunks = _shard_cols(
    32660, 810_000.0, 500_000.0
)
check(
    "antimeridian shard does not claim the whole grid width",
    len(wrap_cols) < N_COLS // 10,
)
check(
    "antimeridian shard reaches both grid edges",
    min(wrap_cols) == 0 and max(wrap_cols) == N_COLS - 1,
)
# One enclosing rectangle would span every column and make the coarsening
# read and rewrite the entire grid width; two tight ones must not.
check("antimeridian footprint splits into two regions", len(wrap_regions) == 2)
wrap_slots = sum(
    ((b - a) // GLOBAL_CHUNK) * ((d - c) // GLOBAL_CHUNK) for a, b, c, d in wrap_regions
)
wrap_rows = {r for r, _c in wrap_chunks}
enclosing = (max(wrap_rows) - min(wrap_rows) + 1) * (
    max(wrap_cols) - min(wrap_cols) + 1
)
check(
    "split regions stay near the real chunk count",
    wrap_slots <= 2 * len(wrap_chunks),
)
check(
    "split beats a single enclosing rectangle by orders of magnitude",
    wrap_slots * 100 < enclosing,
)

# A shard well inside the same zone must be unaffected by the wrap handling.
mid_cols, mid_span, mid_regions, _mid_chunks = _shard_cols(32660, 500_000.0, 500_000.0)
check("mid-zone shard stays contiguous", mid_span == len(mid_cols) - 1)
check("mid-zone shard spans few columns", len(mid_cols) < 20)
check("mid-zone shard needs one region", len(mid_regions) == 1)


# ---------------------------------------------------------------------------
# Global preview pyramid: destination may be local or remote
# ---------------------------------------------------------------------------
# The pyramid is written through a StoreLocation, so a file:// destination
# drives the same fsspec path an s3:// one takes.

import zarr  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from geotessera.zarr import (  # noqa: E402
    GLOBAL_CHUNK,
    GLOBAL_LEVEL0_H,
    GLOBAL_LEVEL0_W,
    GLOBAL_NUM_BANDS,
    _coarsen_zone_pyramid,
    _ensure_global_store,
    _preview_marker_parts,
)

for label, dest_loc in (
    ("local path", str(TMP / "pyr_local.zarr")),
    ("file:// url", url(TMP / "pyr_url.zarr")),
):
    dest = StoreLocation.resolve(dest_loc)
    dest.open_group(mode="a", zarr_format=3)
    _ensure_global_store(dest, 4)

    root = dest.open_group(mode="r+", zarr_format=3)
    check(
        f"pyramid level 0 has the global shape over {label}",
        root["global_rgb/0/rgb"].shape
        == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS),
    )
    check(f"pyramid levels created over {label}", "global_rgb/3/rgb" in root)
    check(
        f"pyramid registers its conventions over {label}",
        {c["name"] for c in root["global_rgb"].attrs["zarr_conventions"]}
        == {"spatial:", "proj:", "multiscales"},
    )
    check(
        f"pyramid keeps per-level geometry over {label}",
        "spatial:shape" in root["global_rgb"].attrs["multiscales"]["layout"][1],
    )

    # A second ensure on a matching pyramid must not wipe what is there.
    r0, c0 = 4 * GLOBAL_CHUNK, 6 * GLOBAL_CHUNK
    root["global_rgb/0/rgb"][r0 : r0 + GLOBAL_CHUNK, c0 : c0 + GLOBAL_CHUNK, :] = (
        np.full((GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS), 200, dtype=np.uint8)
    )
    _ensure_global_store(dest, 4)
    check(
        f"re-ensure keeps existing pyramid data over {label}",
        int(dest.open_group(mode="r", zarr_format=3)["global_rgb/0/rgb"][r0, c0, 0])
        == 200,
    )

    _coarsen_zone_pyramid(
        dest=dest,
        row_start=r0,
        row_end=r0 + GLOBAL_CHUNK,
        col_start=c0,
        col_end=c0 + GLOBAL_CHUNK,
        num_levels=4,
        workers=2,
    )
    check(
        f"coarsening writes level 1 over {label}",
        int(dest.open_group(mode="r", zarr_format=3)["global_rgb/1/rgb"][r0 // 2, c0 // 2, 0])
        == 200,
    )

    # start_level walks the region down without touching the levels below it,
    # which is what lets a parallel sweep stop early and a global pass finish.
    g = dest.open_group(mode="r+", zarr_format=3)
    g["global_rgb/1/rgb"][r0 // 2, c0 // 2, :] = 0
    g["global_rgb/2/rgb"][r0 // 4, c0 // 4, :] = 0
    _coarsen_zone_pyramid(
        dest=dest,
        row_start=r0,
        row_end=r0 + GLOBAL_CHUNK,
        col_start=c0,
        col_end=c0 + GLOBAL_CHUNK,
        num_levels=4,
        workers=2,
        start_level=2,
    )
    g = dest.open_group(mode="r", zarr_format=3)
    check(
        f"start_level leaves shallower levels alone over {label}",
        int(g["global_rgb/1/rgb"][r0 // 2, c0 // 2, 0]) == 0,
    )
    check(
        f"start_level still coarsens deeper levels over {label}",
        int(g["global_rgb/2/rgb"][r0 // 4, c0 // 4, 0]) > 0,
    )

    # Markers follow an explicit state_url rather than the <store>.build
    # sibling, so a remote pyramid can keep its bookkeeping on local disk.
    elsewhere = StoreLocation.resolve(
        dest_loc, None, url(TMP / f"state_{label.split()[0]}")
    )
    parts = _preview_marker_parts(7)
    elsewhere.state.write_bytes(b"zone=7\n", *parts)
    check(
        f"state_url redirects markers off the store for {label}",
        elsewhere.state.exists(*parts, on_denied=False)
        and not dest.state.exists(*parts, on_denied=False),
    )

    # Resume markers belong to the state sibling, never the published store.
    state, parts = dest.state, _preview_marker_parts(30)
    state.write_bytes(b"zone=30\n", *parts)
    check(
        f"preview marker lands in the state sibling over {label}",
        state.exists(*parts, on_denied=False)
        and not dest.exists(*parts, on_denied=False),
    )
    state.remove(*parts)
    check(
        f"preview marker is removable over {label}",
        not state.exists(*parts, on_denied=False),
    )

# Coarsening driven by the chunk set must match the rectangle sweep on every
# tile that holds data, while touching far fewer tiles. The rectangle is a
# terrible proxy for where a zone's data is (measured 75-98% empty), and that
# waste was the dominant cost of a preview build.
cmp_a = StoreLocation.resolve(str(TMP / "coarse_rect.zarr"))
cmp_b = StoreLocation.resolve(str(TMP / "coarse_set.zarr"))
data_chunks = {(8, 12), (8, 13), (9, 12), (40, 60)}  # two clusters, far apart
for loc in (cmp_a, cmp_b):
    loc.open_group(mode="a", zarr_format=3)
    _ensure_global_store(loc, 3)
    arr = loc.open_group(mode="r+", zarr_format=3)["global_rgb/0/rgb"]
    for cr, cc in data_chunks:
        arr[
            cr * GLOBAL_CHUNK : (cr + 1) * GLOBAL_CHUNK,
            cc * GLOBAL_CHUNK : (cc + 1) * GLOBAL_CHUNK,
            :,
        ] = np.full(
            (GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS), 200, dtype=np.uint8
        )

rows = [r for r, _c in data_chunks]
cols = [c for _r, c in data_chunks]
_coarsen_zone_pyramid(
    dest=cmp_a,
    num_levels=3,
    workers=2,
    row_start=min(rows) * GLOBAL_CHUNK,
    row_end=(max(rows) + 1) * GLOBAL_CHUNK,
    col_start=min(cols) * GLOBAL_CHUNK,
    col_end=(max(cols) + 1) * GLOBAL_CHUNK,
)
_coarsen_zone_pyramid(dest=cmp_b, num_levels=3, workers=2, chunks=data_chunks)

# Compare only the windows the data can reach — the full level-1 array is
# 900k x 1.8M x 4, which is terabytes if materialised.
same = True
for lvl in (1, 2):
    aa = cmp_a.open_group(mode="r", zarr_format=3)[f"global_rgb/{lvl}/rgb"]
    bb = cmp_b.open_group(mode="r", zarr_format=3)[f"global_rgb/{lvl}/rgb"]
    for cr, cc in data_chunks:
        r0 = (cr >> lvl) * GLOBAL_CHUNK
        c0 = (cc >> lvl) * GLOBAL_CHUNK
        wa = np.asarray(aa[r0 : r0 + GLOBAL_CHUNK, c0 : c0 + GLOBAL_CHUNK, :])
        wb = np.asarray(bb[r0 : r0 + GLOBAL_CHUNK, c0 : c0 + GLOBAL_CHUNK, :])
        same = same and np.array_equal(wa, wb)
check("chunk-driven coarsening matches the rectangle sweep exactly", same)

lvl1 = cmp_b.open_group(mode="r", zarr_format=3)["global_rgb/1/rgb"]
got = [
    int(np.asarray(lvl1[(cr // 2) * GLOBAL_CHUNK, (cc // 2) * GLOBAL_CHUNK, 0]))
    for cr, cc in sorted(data_chunks)
]
check("coarsened data lands where it should", all(v == 200 for v in got))
# The rectangle spanning both clusters is ~32x21 chunks; the chunk set is 4.
rect_tiles = ((max(rows) - min(rows) + 1) // 2 + 1) * ((max(cols) - min(cols) + 1) // 2 + 1)
check(
    "chunk set walks far fewer tiles than the rectangle",
    len({(r // 2, c // 2) for r, c in data_chunks}) * 8 < rect_tiles,
)


# A parallel zone sweep calls _ensure_global_store from every zone at once.
# Exactly one may create the structure; the rest must wait, not fail.
race_dest = StoreLocation.resolve(str(TMP / "pyr_race.zarr"))
race_dest.open_group(mode="a", zarr_format=3)


def _racer(_i):
    try:
        _ensure_global_store(race_dest, 3)
        return None
    except Exception as exc:  # noqa: BLE001 - the point is to see any failure
        return f"{type(exc).__name__}: {exc}"


with ThreadPoolExecutor(max_workers=8) as _pool:
    race_errs = [e for e in _pool.map(_racer, range(8)) if e]

check(
    "concurrent _ensure_global_store calls all succeed",
    not race_errs,
)
check(
    "the raced pyramid is complete",
    "multiscales" in dict(race_dest.open_group(mode="r", path="global_rgb").attrs)
    and "global_rgb/2/rgb" in race_dest.open_group(mode="r", zarr_format=3),
)

# Consolidation goes through the location too, so it works on either.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Consolidated metadata")
    zarr.consolidate_metadata(
        StoreLocation.resolve(url(TMP / "pyr_url.zarr")).as_zarr_store()
    )
check(
    "pyramid consolidates over a url",
    StoreLocation.resolve(url(TMP / "pyr_url.zarr")).exists(
        "zarr.json", on_denied=False
    ),
)

# ---------------------------------------------------------------------------
# Nested embedding depths (docs/specs/zarr-matryoshka-depths.md)
# ---------------------------------------------------------------------------

from geotessera.zarr import (  # noqa: E402
    N_BANDS,
    SHARD_SIZE,
    _create_zone_group,
    _existing_shards,
    depth_array_name,
    depth_band_dim,
    depth_inner_chunk,
    depths_attr_value,
    store_depths,
    validate_matryoshka_depths,
)

# Chunk geometry: bytes per inner chunk stay at or below the depth-128 budget,
# so no depth produces a chunk larger than the layout already in production.
check("depth-128 keeps the 32px inner chunk", depth_inner_chunk(N_BANDS) == 32)
check("depth-16 widens to 64px", depth_inner_chunk(16) == 64)
check("depth-4 widens to 128px", depth_inner_chunk(4) == 128)
check(
    "no depth exceeds the chunk byte budget",
    all(
        d * depth_inner_chunk(d) ** 2 <= N_BANDS * 32 * 32
        for d in (1, 4, 16, 32, 64, 128)
    ),
)
check(
    "every depth's chunk tiles the shard exactly",
    all(SHARD_SIZE % depth_inner_chunk(d) == 0 for d in (1, 4, 16, 32, 64, 128)),
)

# Naming and the root attribute.
check("full depth keeps the base array name", depth_array_name(N_BANDS) == "embeddings")
check("nested depth gets a suffixed name", depth_array_name(4) == "embeddings_d4")
check("nested depth gets its own band dim", depth_band_dim(4) == "band_d4")
check("full depth keeps the plain band dim", depth_band_dim(N_BANDS) == "band")
check(
    "the depths attribute lists the full depth too",
    depths_attr_value((4, 16))
    == [
        {"dimensions": 4, "array": "embeddings_d4"},
        {"dimensions": 16, "array": "embeddings_d16"},
        {"dimensions": 128, "array": "embeddings"},
    ],
)

# Gating: v1/v1.1 dimensions are not ordered, so a prefix of them is an
# arbitrary slice and the store would look correct while being meaningless.
for bad_version in ("1.0", "1.1"):
    try:
        validate_matryoshka_depths((4,), bad_version)
        refused = False
    except ValueError:
        refused = True
    check(f"depths refused for dataset v{bad_version}", refused)

check(
    "depths accepted and sorted for v2.0",
    validate_matryoshka_depths((16, 4), "2.0") == (4, 16),
)
check("no depths requested is always fine", validate_matryoshka_depths((), "1.0") == ())

for bad, why in [((0,), "below 1"), ((4, 128), "at the full depth"), ((200,), "above")]:
    try:
        validate_matryoshka_depths(bad, "2.0")
        refused = False
    except ValueError:
        refused = True
    check(f"depth {why} is refused", refused)

# A real store: create a zone group with depths and write one shard.
depth_root = StoreLocation.resolve(url(TMP / "depths.zarr"))
depth_root.open_group(mode="w-", zarr_format=3, use_consolidated=None)
depth_grid = UnifiedZoneGrid(
    zone=31,
    years=[2024],
    canonical_epsg=32631,
    origin_x=0.0,
    origin_y=100000.0,
    width_px=SHARD_SIZE,
    height_px=SHARD_SIZE,
)
zone = _create_zone_group(depth_grid, depth_root, 8, depths=(4, 16))

check(
    "every declared depth array exists",
    all(depth_array_name(d) in zone for d in (4, 16, N_BANDS)),
)
check(
    "depth arrays share the spatial shard grid with embeddings",
    all(
        zone[depth_array_name(d)].shards[2:] == zone["embeddings"].shards[2:]
        for d in (4, 16)
    ),
)
check(
    "depth arrays carry the expected chunk geometry",
    zone["embeddings_d4"].chunks == (1, 4, 128, 128)
    and zone["embeddings_d16"].chunks == (1, 16, 64, 64),
)
check(
    "scales is shared, not duplicated per depth",
    "scales" in zone and not any(f"scales_d{d}" in zone for d in (4, 16)),
)
check(
    "each depth band axis has a matching coordinate array",
    all(zone[depth_band_dim(d)].shape == (d,) for d in (4, 16)),
)

# Write one shard the way _fill_and_write_shard does — prefixes first, full
# depth last — and check the prefixes really are prefixes.
rng = np.random.default_rng(0)
emb = rng.integers(-128, 127, size=(N_BANDS, SHARD_SIZE, SHARD_SIZE), dtype=np.int8)

_zone_rw = depth_root.open_group(mode="r+", path="utm31", zarr_format=3)
for _d in (4, 16):
    _zone_rw[depth_array_name(_d)][0, :, 0:SHARD_SIZE, 0:SHARD_SIZE] = emb[:_d]
_zone_rw["embeddings"][0, :, 0:SHARD_SIZE, 0:SHARD_SIZE] = emb
_zone_rw["scales"][0, 0:SHARD_SIZE, 0:SHARD_SIZE] = np.float32(0.05)

check(
    "depth-4 array is the first 4 dims of the full array",
    np.array_equal(
        np.asarray(_zone_rw["embeddings_d4"][0, :, :512, :512]),
        np.asarray(_zone_rw["embeddings"][0, :4, :512, :512]),
    ),
)
check(
    "depth-16 array is the first 16 dims of the full array",
    np.array_equal(
        np.asarray(_zone_rw["embeddings_d16"][0, :, :512, :512]),
        np.asarray(_zone_rw["embeddings"][0, :16, :512, :512]),
    ),
)

# The resume oracle must address every depth with the same shard coordinate,
# which is what lets `embeddings` alone stand for all of them.
check(
    "shard listing finds the same coord in every depth",
    all(
        _existing_shards(
            depth_root, "utm31", 0, {(0, 0)}, array_name=depth_array_name(d)
        )
        == {(0, 0)}
        for d in (4, 16, N_BANDS)
    ),
)

# Root discovery.
_droot = depth_root.open_group(mode="r+", zarr_format=3)
_droot.attrs["geoemb:depths"] = depths_attr_value((4, 16))
check("store_depths reads back the declared depths", store_depths(depth_root) == (4, 16))
check(
    "a store with no declaration has no depths",
    store_depths(StoreLocation.resolve(url(TMP / "pyr_url.zarr"))) == (),
)

# ---------------------------------------------------------------------------
# Seam-aware point reads (geotessera/store.py)
# ---------------------------------------------------------------------------
# A point read must survive the one-pixel gaps left at tile and zone seams
# without ever turning water into land.

import xarray as xr  # noqa: E402

from geotessera.store import (  # noqa: E402
    NODATA,
    OUTSIDE,
    VALID,
    WATER,
    _seam_neighbours,
    _zone_for_lon,
)

check("zone interior needs no neighbours", _seam_neighbours(3.0) == [])
# On a seam the natural zone is the eastern one, so the list holds only its
# western neighbour: what probe() has not already tried.
check("exactly on a seam adds the western zone", _seam_neighbours(138.0) == [53])
check("just east of a seam looks west", _seam_neighbours(138.02) == [53])
check("just west of a seam looks east", _seam_neighbours(137.98) == [54])

# The band must cover every point a neighbouring zone could hold. A tile is
# 0.1 degrees and belongs to the zone containing its centre, so that is
# exactly the points within half a tile of the boundary.
check(
    "the whole half-tile band on each side consults a neighbour",
    all(_seam_neighbours(138.0 + d) for d in (-0.05, -0.03, -0.01, 0.01, 0.03, 0.05)),
)
# ... and no further: a point a whole tile away is unambiguously its own
# zone's, so the neighbour lookup stays off the common path.
check(
    "a point a tile-width from the seam needs no neighbour",
    _seam_neighbours(138.2) == [] and _seam_neighbours(137.8) == [],
)
check("zone 1 wraps west to 60", _seam_neighbours(-179.98) == [60])
check("zone 60 wraps east to 1", _seam_neighbours(179.98) == [1])


def _fake_zone(scales_2d, epsg=32653, px=10.0, ox=300000.0, oy=4050000.0):
    """A minimal zone Dataset the .tessera accessor can read."""
    h, w = scales_2d.shape
    b = 4
    emb = np.tile(np.arange(1, b + 1, dtype=np.int8)[:, None, None], (1, h, w))
    ds = xr.Dataset(
        {
            "embeddings": (("time", "band", "y", "x"), emb[None]),
            "scales": (("time", "y", "x"), np.asarray(scales_2d)[None]),
        },
        coords={
            "time": [2024],
            "band": np.arange(b),
            "x": ox + (np.arange(w) + 0.5) * px,
            "y": oy - (np.arange(h) + 0.5) * px,
        },
        attrs={
            "proj:code": f"EPSG:{epsg}",
            "spatial:transform": [px, 0.0, ox, 0.0, -px, oy],
            "geoemb:dimensions": b,
        },
    )
    return ds


INF, NAN = np.float32("inf"), np.float32("nan")

# The artifact found at tile corners in the published v1 store: a patch of
# data with a single unwritten pixel at its centre.
sc = np.full((5, 5), np.float32(0.05))
sc[2, 2] = INF
holed = _fake_zone(sc)
cx = float(holed.x[2])
cy = float(holed.y[2])

v, st = holed.tessera.probe(cx, cy, 2024, search_px=0)
check("without repair a one-pixel hole reads as nodata", v is None and st == NODATA)

v, st = holed.tessera.probe(cx, cy, 2024, search_px=1)
check("with repair the hole resolves to a neighbour", v is not None and st == VALID)
check(
    "the repaired value is a real embedding, not a fill value",
    v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
)

# A NaN centre is a real answer and must never be searched away.
sw = np.full((5, 5), np.float32(0.05))
sw[2, 2] = NAN
watery = _fake_zone(sw)
v, st = watery.tessera.probe(float(watery.x[2]), float(watery.y[2]), 2024, search_px=1)
check("water is reported as water, not repaired into land", v is None and st == WATER)

# An entirely unwritten neighbourhood stays nodata however wide the search.
blank = _fake_zone(np.full((5, 5), INF))
v, st = blank.tessera.probe(float(blank.x[2]), float(blank.y[2]), 2024, search_px=2)
check("an all-unwritten window stays nodata", v is None and st == NODATA)

# Snapping must not silently answer for a point outside the grid.
v, st = holed.tessera.probe(float(holed.x[0]) - 10_000.0, cy, 2024)
check("a point far outside the grid reports outside", v is None and st == OUTSIDE)

# The nearest valid pixel wins, not the first one scanned.
sc2 = np.full((5, 5), INF)
sc2[2, 3] = np.float32(0.07)  # immediately east of centre
sc2[0, 0] = np.float32(0.09)  # further away
near = _fake_zone(sc2)
v, st = near.tessera.probe(float(near.x[2]), float(near.y[2]), 2024, search_px=2)
check(
    "repair picks the nearest valid pixel",
    v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
)

check(
    "zone routing still agrees with the plain longitude rule away from seams",
    _zone_for_lon(2.35) == 31 and _zone_for_lon(-120.5) == 10,
)

# ---------------------------------------------------------------------------
# Landmask-free stores
# ---------------------------------------------------------------------------
# v2 inference covers every pixel of a tile it emits, so a present tile needs
# no mask. The grid must then be sized from the embeddings: the published
# landmask stops well north of v2's southernmost tiles.

from geotessera.zarr import (  # noqa: E402
    _fill_and_write_shard,
    _gather_landmask_tiles_by_zone,
    store_uses_landmask,
)


class _FakeRegistry:
    """Just enough of Registry for the two coordinate accessors used here."""

    def __init__(self, emb, land):
        import pandas as pd

        self._emb, self._land = emb, land
        idx = pd.MultiIndex.from_tuples(
            [(y, int(round(lo * 100)), int(round(la * 100))) for y, lo, la in emb],
            names=["year", "lon_i", "lat_i"],
        )
        self._registry_gdf = pd.DataFrame(index=idx)

    @property
    def available_landmasks(self):
        return self._land


# Embeddings reach lat -80; the landmask stops at -20, as v2's does.
reg = _FakeRegistry(
    emb=[(2024, 3.05, -80.05), (2024, 3.05, 10.05), (2025, 3.05, -80.05)],
    land=[(3.05, -20.05), (3.05, 10.05)],
)
by_land = _gather_landmask_tiles_by_zone(reg)
by_emb = _gather_landmask_tiles_by_zone(reg, from_embeddings=True)
check("landmask grid uses the landmask tiles", sorted(by_land[31]) == [(3.05, -20.05), (3.05, 10.05)])
check(
    "landmask-free grid uses the embedding tiles",
    sorted(by_emb[31]) == [(3.05, -80.05), (3.05, 10.05)],
)
check(
    "and so reaches tiles the landmask never covers",
    min(la for _, la in by_emb[31]) < min(la for _, la in by_land[31]),
)
check("embedding tiles are deduplicated across years", len(by_emb[31]) == 2)

# The flag round-trips through a store root, defaulting to "masked".
lm_root = StoreLocation.resolve(url(TMP / "nolm.zarr"))
_r = lm_root.open_group(mode="w-", zarr_format=3, use_consolidated=None)
check("a store says masked unless told otherwise", store_uses_landmask(lm_root) is True)
_r.attrs["geoemb:landmask"] = False
check("a landmask-free store reports itself", store_uses_landmask(lm_root) is False)

# Drive a real shard write both ways and count landmask reads.
import geotessera.zarr as _z  # noqa: E402
from geotessera.zarr import ShardSpec, ShardTileOverlap  # noqa: E402

TW = 64  # tile side, in pixels
grid2 = UnifiedZoneGrid(
    zone=31, years=[2024], canonical_epsg=32631,
    origin_x=0.0, origin_y=100000.0, width_px=SHARD_SIZE, height_px=SHARD_SIZE,
)
zone2 = _create_zone_group(grid2, lm_root, 8)

ov = ShardTileOverlap(
    embedding_path="emb", scales_path="sc", landmask_path="lm",
    t_row_start=0, t_row_end=TW, t_col_start=0, t_col_end=TW,
    s_row_start=0, s_row_end=TW, s_col_start=0, s_col_end=TW,
)
spec2 = ShardSpec(
    time_index=0, sr=0, sc=0, row_px=0, col_px=0, tiles=[ov],
)

lm_calls = []
_real_lm = _z._load_landmask_slice
_real_npy = remote.read_npy_window
# Landmask marks the whole tile as water, so if it is consulted the scales
# come back NaN and the difference is visible in the data, not just the count.
_z._load_landmask_slice = lambda *a, **k: (lm_calls.append(a[0]), np.zeros((TW, TW), np.uint8))[1]
remote.read_npy_window = lambda path, r0, r1, c0, c1, storage_options=None: (
    np.ones((r1 - r0, c1 - c0, N_BANDS), np.int8)
    if "emb" in str(path)
    else np.full((r1 - r0, c1 - c0), np.float32(0.05))
)
try:
    for use_lm, label in ((True, "with"), (False, "without")):
        lm_calls.clear()
        buf = np.zeros((N_BANDS, SHARD_SIZE, SHARD_SIZE), np.int8)
        sbuf = np.full((SHARD_SIZE, SHARD_SIZE), np.float32("inf"))
        _fill_and_write_shard(spec2, zone2, buf, sbuf, None, 0, (), use_lm)
        got = np.asarray(zone2["scales"][0, 0, 0])
        if use_lm:
            check("a masked fill reads the landmask", len(lm_calls) == 1)
            check("and its water pixels become NaN", bool(np.isnan(got)))
        else:
            check("a landmask-free fill reads no landmask", lm_calls == [])
            check("and its pixels keep real scales", float(got) == np.float32(0.05))
finally:
    _z._load_landmask_slice = _real_lm
    remote.read_npy_window = _real_npy

# The scale sanity check must survive: it rejects nodata inside the tile,
# which is a different thing from water and is not the landmask's job.
remote.read_npy_window = lambda path, r0, r1, c0, c1, storage_options=None: (
    np.ones((r1 - r0, c1 - c0, N_BANDS), np.int8)
    if "emb" in str(path)
    else np.full((r1 - r0, c1 - c0), np.float32(1e6))  # sentinel-huge: nodata
)
try:
    buf = np.zeros((N_BANDS, SHARD_SIZE, SHARD_SIZE), np.int8)
    sbuf = np.full((SHARD_SIZE, SHARD_SIZE), np.float32("inf"))
    _fill_and_write_shard(spec2, zone2, buf, sbuf, None, 0, (), False)
    check(
        "out-of-range scales are still rejected without a landmask",
        bool(np.isnan(np.asarray(zone2["scales"][0, 0, 0]))),
    )
finally:
    remote.read_npy_window = _real_npy

# ---------------------------------------------------------------------------
# Preview source selection
# ---------------------------------------------------------------------------
# A bands-mode preview only touches the first few dimensions, so it can read a
# nested-depth prefix instead of all 128. A pca preview projects every
# dimension and must not.

from geotessera.zarr import RGB_PREVIEW_BANDS, preview_source_array  # noqa: E402

_dr = depth_root.open_group(mode="r", zarr_format=3)
check(
    "bands mode reads the shallowest adequate depth",
    preview_source_array(_dr, RGB_PREVIEW_BANDS, "bands") == "embeddings_d4",
)
check(
    "pca mode always reads the full array",
    preview_source_array(_dr, RGB_PREVIEW_BANDS, "pca") == "embeddings",
)
check(
    "a depth too shallow for the bands is skipped",
    preview_source_array(_dr, (0, 1, 2, 3, 4, 5), "bands") == "embeddings_d16",
)
check(
    "a single-depth store falls back to embeddings",
    preview_source_array(
        StoreLocation.resolve(url(TMP / "pyr_url.zarr")).open_group(mode="r"),
        RGB_PREVIEW_BANDS,
        "bands",
    )
    == "embeddings",
)

# The whole point: the bands a preview reads must be identical either way.
_zrw = depth_root.open_group(mode="r+", path="utm31", zarr_format=3)
check(
    "d4 colour bands are byte-identical to the full array's",
    np.array_equal(
        np.asarray(_zrw["embeddings_d4"][0, 0:3, :256, :256]),
        np.asarray(_zrw["embeddings"][0, 0:3, :256, :256]),
    ),
)
check(
    "and the RGB chunk built from each is the same",
    np.array_equal(
        _z._compute_rgb_chunk(
            np.asarray(_zrw["embeddings_d4"][0, 0:3, :64, :64]),
            np.asarray(_zrw["scales"][0, :64, :64]),
            RGB_PREVIEW_BANDS, [-1.0] * 3, [1.0] * 3,
        ),
        _z._compute_rgb_chunk(
            np.asarray(_zrw["embeddings"][0, 0:3, :64, :64]),
            np.asarray(_zrw["scales"][0, :64, :64]),
            RGB_PREVIEW_BANDS, [-1.0] * 3, [1.0] * 3,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Per-zone stretch blending
# ---------------------------------------------------------------------------
# A single global stretch flattens regions that sit at one end of the global
# distribution. Per-zone stretches fix that but step at zone boundaries; the
# tent blend must remove the step exactly, not approximately.

from geotessera.zarr import (  # noqa: E402
    _zone_centre_lon,
    blend_stretches,
    stretch_for_lon,
)


def _st(v):
    """A stretch whose every number is v, so blends are easy to read."""
    return {"min": [v] * 3, "max": [v + 1] * 3, "cdf": [[v] * 4] * 3, "bands": [0, 1, 2]}


check("zone centre is the central meridian", _zone_centre_lon(31) == 3.0)
check("zone 1 centre is -177", _zone_centre_lon(1) == -177.0)

mid = blend_stretches(_st(0.0), _st(10.0), 0.5)
check("an even blend is the midpoint", mid["min"] == [5.0] * 3)
check("and blends the cdf breakpoints too", mid["cdf"][0] == [5.0] * 4)
check("weight 0 returns the first operand", blend_stretches(_st(2.0), _st(9.0), 0.0)["min"] == [2.0] * 3)
check("weight 1 returns the second", blend_stretches(_st(2.0), _st(9.0), 1.0)["min"] == [9.0] * 3)
check("a missing operand passes the other through", blend_stretches(None, _st(4.0), 0.5)["min"] == [4.0] * 3)
check("both missing yields nothing", blend_stretches(None, None, 0.5) is None)

# The property that matters: no step at a zone boundary. Give each zone a
# distinct stretch so any discontinuity is glaring.
ZS = {z: _st(float(z)) for z in range(1, 61)}

west = stretch_for_lon(ZS, -1e-9)   # zone 30, at its eastern edge
east = stretch_for_lon(ZS, +1e-9)   # zone 31, at its western edge
check(
    "the two sides of a zone boundary agree",
    max(abs(a - b) for a, b in zip(west["min"], east["min"])) < 1e-6,
)
check(
    "and they agree on the cdf as well",
    max(abs(a - b) for a, b in zip(west["cdf"][0], east["cdf"][0])) < 1e-6,
)
check(
    "a boundary blends the two zones evenly",
    abs(west["min"][0] - 30.5) < 1e-6,
)
check(
    "a zone centre uses that zone alone",
    abs(stretch_for_lon(ZS, _zone_centre_lon(31))["min"][0] - 31.0) < 1e-9,
)

# Walking across a boundary must never jump: the largest step between adjacent
# samples should be proportional to the step in longitude, not to the
# difference between the two zones' stretches.
samples = [stretch_for_lon(ZS, lo)["min"][0] for lo in np.arange(-0.5, 0.5, 0.01)]
steps = [abs(b - a) for a, b in zip(samples, samples[1:])]
check("crossing a boundary is continuous", max(steps) < 0.01)
check("and monotonic through it", all(b >= a - 1e-9 for a, b in zip(samples, samples[1:])))

# Antimeridian: zone 60's eastern neighbour is zone 1.
check(
    "zone 60 blends east into zone 1",
    abs(stretch_for_lon(ZS, 179.999)["min"][0] - 30.5) < 1.0,
)
check(
    "zone 1 blends west into zone 60",
    abs(stretch_for_lon(ZS, -179.999)["min"][0] - 30.5) < 1.0,
)

# A zone with no stretch of its own falls back without a discontinuity.
partial = {30: _st(30.0)}
check(
    "a zone without a stretch uses the fallback",
    stretch_for_lon(partial, 20.0, fallback=_st(99.0))["min"] == [99.0] * 3,
)
check(
    "and the fallback blends smoothly with a real neighbour",
    30.0 < stretch_for_lon(partial, -0.001, fallback=_st(99.0))["min"][0] < 99.0,
)

# ---------------------------------------------------------------------------
# Tile-footprint chunk filtering
# ---------------------------------------------------------------------------
# The work list comes from written shards, but a shard is 4096 px against a
# tile's 0.1 degrees, so on a sparse dataset most of it holds nothing. The
# filter must be exact: dropping a chunk a tile can reach would lose data.

from geotessera.zarr import (  # noqa: E402
    GLOBAL_BASE_RES,
    GLOBAL_BOUNDS,
    GLOBAL_CHUNK,
    TILE_SIZE_DEG,
    chunks_for_tile_centres,
)

_west, _s_, _e_, _north = GLOBAL_BOUNDS


def _chunk_of(lon, lat):
    return (int((_north - lat) / GLOBAL_BASE_RES) // GLOBAL_CHUNK,
            int((lon - _west) / GLOBAL_BASE_RES) // GLOBAL_CHUNK)


one = chunks_for_tile_centres([(0.05, 52.05)])
check("a tile yields at least one chunk", len(one) >= 1)
check("and includes the chunk under its centre", _chunk_of(0.05, 52.05) in one)

# Every corner of the footprint must be covered — a missed corner is lost data.
corners = [(0.05 + dx, 52.05 + dy)
           for dx in (-TILE_SIZE_DEG / 2 + 1e-9, TILE_SIZE_DEG / 2 - 1e-9)
           for dy in (-TILE_SIZE_DEG / 2 + 1e-9, TILE_SIZE_DEG / 2 - 1e-9)]
check(
    "every corner of a tile falls in a returned chunk",
    all(_chunk_of(x, y) in one for x, y in corners),
)

# A chunk is 512 * 0.0001 = 0.0512 deg; a 0.1 deg tile spans 2-3 of them.
check("a tile spans a plausible number of chunks", 4 <= len(one) <= 12)

check("no tiles yields no chunks", chunks_for_tile_centres([]) == set())

# Two distant tiles must not merge into a filled rectangle between them.
far = chunks_for_tile_centres([(0.05, 52.05), (20.05, 52.05)])
check(
    "distant tiles stay disjoint",
    len(far) == 2 * len(one) and _chunk_of(10.0, 52.05) not in far,
)

# Adjacent tiles overlap only on their shared edge.
adj = chunks_for_tile_centres([(0.05, 52.05), (0.15, 52.05)])
check("adjacent tiles share at most an edge of chunks", len(adj) < 2 * len(one))

# The filter is an intersection, so it can only ever remove chunks.
shardish = one | {(r, c) for r in range(900, 910) for c in range(900, 910)}
check(
    "intersecting can only shrink the work list",
    (shardish & chunks_for_tile_centres([(0.05, 52.05)])) <= shardish,
)
check(
    "and keeps every chunk the tile needs",
    one <= (shardish & chunks_for_tile_centres([(0.05, 52.05)])) | (one - shardish),
)

import shutil  # noqa: E402

shutil.rmtree(TMP, ignore_errors=True)

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
    status = 1
else:
    print("\nall checks passed")
    status = 0

# Exit without interpreter finalisation. With the zarr/xarray/dask state this
# suite piles up, module teardown on 3.13 sprays "Error in sys.excepthook"
# noise after the verdict — from inside the dependencies, with every thread
# already joined, and only when certain unrelated packages are absent. Every
# check has run and the tempdir is gone, so nothing is left worth finalising.
# First drop the blosc mutex numcodecs keeps at module level and collect, so
# its semaphore unregisters, then stop the multiprocessing resource tracker
# so a bare os._exit leaves it nothing to warn about.
import gc  # noqa: E402
import multiprocessing.resource_tracker as _rt  # noqa: E402

from numcodecs import blosc as _blosc  # noqa: E402

_blosc._MUTEX = None
_blosc._MUTEX_IS_INIT = False
gc.collect()
if _rt._resource_tracker is not None:
    _rt._resource_tracker._stop()
sys.stdout.flush()
sys.stderr.flush()
os._exit(status)
