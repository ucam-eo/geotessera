"""Tessera Zarr store: single store with time dimension.

Layout:
    tessera.zarr/
        zarr.json                     # root: geoemb: convention attrs
        utm{zone:02d}/                # one group per UTM zone
            embeddings                # int8    (T, B, H, W)
            scales                    # float32 (T, H, W)
            time                      # int32   (T,)
            x                         # float64 (W,)
            y                         # float64 (H,)
            band                      # int32   (B,)

The store contains nothing but Zarr. Build-time bookkeeping lives in a
sibling location, ``<store>.build`` by default:

    tessera.zarr.build/
        _registry/utm{zone:02d}_{year}.parquet   # per-zone ingestion tracking
        _registry.parquet             # merged tracking (written by consolidate)
        _locks/utm{zone:02d}_{year}.json         # advisory fill locks
        _preview/zone_{zone}_done     # global-preview resume markers

Dimension order: (time, band, y, x) — ML-standard NCHW.
Inner chunks: (1, 128, 32, 32), Shards: (1, 128, 4096, 4096).

Scale sentinels:
    NaN   = water (permanent, from landmask)
    +inf  = land, no data yet (set at init, replaced by real scale on fill)
    finite = valid data

Locations
---------
The store and the tile inputs are addressed as *locations*: either local
filesystem paths or fsspec URLs (``s3://bucket/prefix``).  :mod:`geotessera.remote`
resolves the difference, so a store on one S3 node can be filled from tiles on
another without any local mirror.

Parallel fills
--------------
A UTM zone's pixels live entirely within its own ``utm{zone}`` group, and
shards never straddle zones, so one process per zone can fill the same store
concurrently.  All per-fill state is keyed by (zone, year) — the ingestion
registry and the advisory lock — so no two zone jobs touch the same object.
The one shared object, the root ``zarr.json``, is only rewritten by
consolidation, which a zone fill skips by default when ``zones`` is set;
run ``zarr-consolidate`` once after the sweep instead.
"""

from __future__ import annotations

import io
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import geopandas
    import pandas
    import rich.console
    import zarr
    from rasterio.transform import Affine

    from .registry import Registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BANDS = 128

# Global preview grid (fixed extent, never changes)
GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
GLOBAL_BASE_RES = 0.0001  # degrees (~10m at equator)
GLOBAL_LEVEL0_W = 3_600_000  # ceil(360 / 0.0001)
GLOBAL_LEVEL0_H = 1_800_000  # ceil(180 / 0.0001)
GLOBAL_CHUNK = 512
GLOBAL_NUM_BANDS = 4
GLOBAL_DEFAULT_LEVELS = 10

# GeoZarr convention registration entries
GEOEMB_CONVENTION = {
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
    "spec_url": "https://github.com/geo-embeddings/embeddings-zarr-convention/blob/v1/README.md",
    "schema_url": "https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/refs/tags/v1/schema.json",
}


# ---------------------------------------------------------------------------
# Data types (shared)
# ---------------------------------------------------------------------------


@dataclass
class TileInfo:
    """Metadata for a single tile to be placed in a zone store."""

    lon: float
    lat: float
    year: int
    epsg: int
    transform: Affine
    height: int
    width: int
    landmask_path: str
    embedding_path: str
    scales_path: str


@dataclass
class ShardTileOverlap:
    """One tile's contribution to one shard — precomputed slice coordinates."""

    embedding_path: str
    scales_path: str
    landmask_path: str
    # Tile-local region to read
    t_row_start: int
    t_row_end: int
    t_col_start: int
    t_col_end: int
    # Shard-buffer region to write into
    s_row_start: int
    s_row_end: int
    s_col_start: int
    s_col_end: int


@dataclass
class ShardSpec:
    """Everything a shard worker needs to write one complete shard."""

    sr: int  # shard row index
    sc: int  # shard col index
    row_px: int  # pixel row in zone grid (sr * SHARD_SIZE)
    col_px: int  # pixel col in zone grid (sc * SHARD_SIZE)
    tiles: List[ShardTileOverlap]
    time_index: int = 0


# ---------------------------------------------------------------------------
# Locations: the store and the tile inputs
# ---------------------------------------------------------------------------


@dataclass
class StoreLocation:
    """A Zarr store addressed by local path or fsspec URL.

    Wraps the handful of operations the build pipeline needs beyond the Zarr
    API itself — existence checks and reading/writing build-state objects —
    so callers never branch on local-vs-remote.
    """

    url: str
    storage_options: Optional[Dict[str, Any]] = None
    state_url: Optional[str] = None
    state_storage_options: Optional[Dict[str, Any]] = None

    @classmethod
    def resolve(
        cls,
        store: "str | Path | StoreLocation",
        storage_options: Optional[Dict[str, Any]] = None,
        state_url: Optional[str] = None,
        state_storage_options: Optional[Dict[str, Any]] = None,
    ) -> "StoreLocation":
        """Coerce a path, URL, or existing location into a StoreLocation."""
        if isinstance(store, StoreLocation):
            return store
        return cls(str(store), storage_options, state_url, state_storage_options)

    @property
    def state(self) -> "StoreLocation":
        """Where build-time state lives — a sibling of the store, not inside it.

        The ingestion registry and fill locks are the builder's bookkeeping,
        not published data. Keeping them out of the store leaves a clean Zarr
        hierarchy: nothing for readers to trip over and nothing for
        ``consolidate_metadata`` to warn about. Defaults to ``<store>.build``
        alongside the store, so a sweep on another host still finds it.
        """
        return StoreLocation(
            self.state_url or f"{self.url.rstrip('/')}.build",
            self.state_storage_options
            if self.state_storage_options is not None
            else self.storage_options,
        )

    @property
    def is_remote(self) -> bool:
        from . import remote

        return remote.is_url(self.url)

    def join(self, *parts: str) -> str:
        from . import remote

        return remote.join(self.url, *parts)

    def exists(self, *parts: str, on_denied: Optional[bool] = None) -> bool:
        from . import remote

        return remote.exists(
            self.join(*parts), self.storage_options, on_denied=on_denied
        )

    def read_bytes(self, *parts: str) -> bytes:
        from . import remote

        return remote.read_bytes(self.join(*parts), self.storage_options)

    def write_bytes(self, data: bytes, *parts: str) -> None:
        from . import remote

        remote.write_bytes(self.join(*parts), data, self.storage_options)

    def remove(self, *parts: str) -> None:
        from . import remote

        remote.remove(self.join(*parts), self.storage_options)

    def listdir(self, *parts: str, on_denied: Optional[List[str]] = None) -> List[str]:
        from . import remote

        return remote.listdir(
            self.join(*parts), self.storage_options, on_denied=on_denied
        )

    def _ensure_backend(self) -> None:
        """Fail early, with an actionable message, if the backend is missing.

        zarr raises its own import error deep inside store construction; this
        surfaces ours (which names the ``geotessera[s3]`` extra) first.
        """
        if self.is_remote:
            from . import remote

            remote.get_fs(self.url, self.storage_options)

    def as_zarr_store(self, read_only: bool = False):
        """Return something ``zarr`` accepts as a store.

        Local paths pass through as strings; remote URLs become an
        ``FsspecStore`` so credentials/endpoint reach APIs like
        ``consolidate_metadata`` that take no ``storage_options``.
        """
        if not self.is_remote:
            return self.url
        self._ensure_backend()
        from zarr.storage import FsspecStore

        return FsspecStore.from_url(
            self.url, storage_options=self.storage_options, read_only=read_only
        )

    def open_group(
        self,
        mode: str = "r",
        path: Optional[str] = None,
        zarr_format: Optional[int] = None,
        use_consolidated: Optional[bool] = False,
    ) -> "zarr.Group":
        """Open the store (or a group within it) with the right backend."""
        import zarr

        self._ensure_backend()
        kwargs: Dict[str, Any] = {
            "mode": mode,
            "use_consolidated": use_consolidated,
        }
        if path is not None:
            kwargs["path"] = path
        if zarr_format is not None:
            kwargs["zarr_format"] = zarr_format
        if self.storage_options:
            kwargs["storage_options"] = self.storage_options
        return zarr.open_group(self.url, **kwargs)

    def __str__(self) -> str:
        return self.url


@dataclass
class TileSource:
    """Where the NPY tiles and landmask GeoTIFFs for a fill live.

    ``embeddings_root`` contains ``{year}/grid_{lon}_{lat}/grid_{lon}_{lat}.npy``
    and its ``_scales.npy`` sibling; ``landmasks_root`` contains
    ``grid_{lon}_{lat}.tiff``.  Both are locations, so a fill can stream from a
    remote bucket with no local mirror.
    """

    embeddings_root: str
    landmasks_root: str
    storage_options: Optional[Dict[str, Any]] = None

    @property
    def is_remote(self) -> bool:
        from . import remote

        return remote.is_url(self.embeddings_root)

    def embedding_locations(self, lon: float, lat: float, year: int) -> Tuple[str, str]:
        """Return (embedding, scales) locations for a tile."""
        from . import remote
        from .registry import tile_to_embedding_paths

        emb_rel, scales_rel = tile_to_embedding_paths(lon, lat, year)
        return (
            remote.join(self.embeddings_root, emb_rel.as_posix()),
            remote.join(self.embeddings_root, scales_rel.as_posix()),
        )

    def landmask_location(self, lon: float, lat: float) -> str:
        """Return the landmask location for a tile."""
        from . import remote
        from .registry import tile_to_landmask_filename

        return remote.join(self.landmasks_root, tile_to_landmask_filename(lon, lat))

    @classmethod
    def for_url(
        cls,
        root: str,
        version_path: str,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> "TileSource":
        """Build a source from a repository root in the published layout.

        The Source Cooperative repository — and any mirror of it — lays tiles
        out as ``{root}/npy/{version}/{year}/grid_.../`` with landmasks under
        ``{root}/landmasks/{version}/``.  ``root`` may be an ``s3://`` URL for
        a credentialed mirror or an ``https://`` URL for the public front.
        """
        from . import remote

        return cls(
            embeddings_root=remote.join(root, "npy", version_path),
            landmasks_root=remote.join(root, "landmasks", version_path),
            storage_options=storage_options,
        )

    @classmethod
    def for_local_mirror(cls, registry: "Registry") -> "TileSource":
        """Build a source from a registry's local ``embeddings_dir``.

        The local mirror can be in two shapes:

        * flat: ``<base_dir>/global_0.1_degree_representation/<year>/...`` (what
          the geotessera-download CLI writes — variant info in a sidecar)
        * S3-mirror: ``<base_dir>/<version_path>/global_0.1_degree_representation/
          <year>/...`` (what ``aws s3 cp --recursive`` produces)

        The S3-mirror layout is preferred when present so users keeping a
        multi-version mirror under one root can point zarr-fill at the top.
        Landmasks are STRICTLY per-version — no cross-version fallback. Each
        Tessera version has its own landmask grid and mixing them silently
        would corrupt water masking.
        """
        from .registry import (
            EMBEDDINGS_DIR_NAME,
            LANDMASKS_DIR_NAME,
            TESSERA_MIRROR_ENDPOINT,
            TESSERA_MIRROR_REPO,
        )

        def landmask_sync_hint(dest: Path) -> str:
            return (
                f"  aws s3 sync --no-sign-request "
                f"--endpoint-url {TESSERA_MIRROR_ENDPOINT} "
                f"s3://{TESSERA_MIRROR_REPO}/landmasks/"
                f"{registry._version_path}/ {dest}/"
            )

        emb_candidate = (
            registry._embeddings_dir / registry._version_path / EMBEDDINGS_DIR_NAME
        )
        lm_s3_mirror = (
            registry._embeddings_dir / registry._version_path / LANDMASKS_DIR_NAME
        )
        lm_flat = registry._embeddings_dir / LANDMASKS_DIR_NAME

        if emb_candidate.exists():
            base_emb = str(emb_candidate)
            # When embeddings are in S3-mirror layout, landmasks must match.
            # Don't fall back to the flat layout — that would silently pick up
            # the wrong version's landmasks.
            if lm_s3_mirror.exists():
                base_lm = str(lm_s3_mirror)
            else:
                raise FileNotFoundError(
                    f"Landmask directory not found for {registry._version_path}: "
                    f"expected {lm_s3_mirror}. Landmasks are per-version and "
                    f"cannot be reused across versions. Fetch them with:\n"
                    + landmask_sync_hint(lm_s3_mirror)
                )
        else:
            base_emb = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME)
            if lm_flat.exists():
                base_lm = str(lm_flat)
            else:
                raise FileNotFoundError(
                    f"Landmask directory not found: expected {lm_flat}. "
                    f"Fetch them with:\n" + landmask_sync_hint(lm_flat)
                )

        return cls(embeddings_root=base_emb, landmasks_root=base_lm)


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------


def epsg_is_south(epsg: int) -> bool:
    """Check if an EPSG code is a southern hemisphere UTM zone."""
    return 32701 <= epsg <= 32760


def zone_canonical_epsg(zone: int) -> int:
    """Get the canonical (northern hemisphere) EPSG code for a UTM zone."""
    return 32600 + zone


def northing_to_canonical(northing: float, epsg: int) -> float:
    """Convert a northing to canonical coordinates.

    Southern hemisphere tiles use a false northing of 10,000,000m.
    We subtract this for a continuous axis.
    """
    if epsg_is_south(epsg):
        return northing - 10_000_000.0
    return northing


def _zone_group_name(zone: int) -> str:
    """Return the group name for a UTM zone within a store."""
    return f"utm{zone:02d}"


def tile_zone(lon: float) -> int:
    """UTM zone number (1-60) containing a longitude."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


def project_tile(
    lon: float,
    lat: float,
    year: int = 0,
    transformer_cache: Optional[Dict[int, Any]] = None,
    landmask_path: str = "",
    embedding_path: str = "",
    scales_path: str = "",
    pixel_size: float = 10.0,
) -> TileInfo:
    """Compute a tile's UTM footprint from its centre coordinates alone.

    Deterministic — no file is opened — so it works for a landmask tile whose
    embeddings do not exist as readily as for one that has data.
    """
    from pyproj import Transformer as ProjTransformer
    from rasterio.transform import Affine

    if transformer_cache is None:
        transformer_cache = {}

    zone_num = tile_zone(lon)
    epsg = (32700 if lat < 0 else 32600) + zone_num

    if epsg not in transformer_cache:
        transformer_cache[epsg] = ProjTransformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
    proj = transformer_cache[epsg]

    west, east = lon - 0.05, lon + 0.05
    south, north = lat - 0.05, lat + 0.05
    ul_e, ul_n = proj.transform(west, north)
    ur_e, ur_n = proj.transform(east, north)
    ll_e, ll_n = proj.transform(west, south)
    lr_e, lr_n = proj.transform(east, south)

    origin_e = min(ul_e, ll_e)
    origin_n = max(ul_n, ur_n)

    return TileInfo(
        lon=lon,
        lat=lat,
        year=year,
        epsg=epsg,
        transform=Affine(pixel_size, 0.0, origin_e, 0.0, -pixel_size, origin_n),
        height=round((origin_n - min(ll_n, lr_n)) / pixel_size),
        width=round((max(ur_e, lr_e) - origin_e) / pixel_size),
        landmask_path=landmask_path,
        embedding_path=embedding_path,
        scales_path=scales_path,
    )


# ---------------------------------------------------------------------------
# Landmask handling
# ---------------------------------------------------------------------------


def _load_landmask_slice(
    landmask_path: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    storage_options: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Load a sub-region of a landmask GeoTIFF, locally or from a remote store.

    Returns a 2D uint8 array where 0 = water.  If the landmask cannot be
    read (missing file, shape mismatch, etc.) returns all-ones (all land)
    so that no pixels are masked.
    """
    from . import remote

    try:
        return remote.read_tiff_window(
            landmask_path,
            row_start,
            row_end,
            col_start,
            col_end,
            storage_options=storage_options,
        )
    except Exception as e:
        logger.warning(f"Failed to read landmask slice from {landmask_path}: {e}")
        return np.ones((row_end - row_start, col_end - col_start), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tile info gathering
# ---------------------------------------------------------------------------


def gather_tile_infos(
    registry: "Registry",
    year: int,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
    source: Optional[TileSource] = None,
) -> Dict[int, List[TileInfo]]:
    """Gather tile metadata and group by UTM zone.

    Computes grid info deterministically from coordinates (no file I/O).

    Args:
        source: Where the tile inputs live.  Defaults to the registry's local
            mirror; pass a :class:`TileSource` built with
            :meth:`TileSource.for_url` to stream from a remote bucket.
    """
    # Get tiles for this year from MultiIndex, filtering to those with data
    gdf = registry._registry_gdf
    try:
        year_slice = gdf.loc[year]
        # Filter to tiles that have actual embedding data in the registry
        valid = year_slice["grid_size"] > 0
        if "scales_size" in year_slice.columns:
            valid = valid & (year_slice["scales_size"] > 0)
        year_slice = year_slice[valid]
        tiles = [
            (year, lon_i / 100.0, lat_i / 100.0)
            for lon_i, lat_i in year_slice.index.unique()
        ]
    except KeyError:
        tiles = []

    if console is not None:
        console.print(f"  Found {len(tiles):,} tiles for year {year}")

    zone_set = set(zones) if zones is not None else None

    # Pre-filter by UTM zone (deterministic from longitude)
    if zone_set is not None:
        before = len(tiles)
        tiles = [
            (y, lon, lat)
            for y, lon, lat in tiles
            if int(math.floor((lon + 180.0) / 6.0)) + 1 in zone_set
        ]
        if console is not None:
            console.print(
                f"  Filtered to {len(tiles):,} tiles in zone(s) "
                f"{','.join(str(z) for z in sorted(zone_set))} "
                f"(skipped {before - len(tiles):,})"
            )

    # Build TileInfos using computed grid (no file I/O)
    if source is None:
        source = TileSource.for_local_mirror(registry)

    zones_dict: Dict[int, List[TileInfo]] = {}
    transformer_cache: Dict[int, Any] = {}

    for tile_year, tile_lon, tile_lat in tiles:
        zone_num = tile_zone(tile_lon)
        if zone_set is not None and zone_num not in zone_set:
            continue

        emb_path, scales_path = source.embedding_locations(
            tile_lon, tile_lat, tile_year
        )
        ti = project_tile(
            tile_lon,
            tile_lat,
            tile_year,
            transformer_cache,
            landmask_path=source.landmask_location(tile_lon, tile_lat),
            embedding_path=emb_path,
            scales_path=scales_path,
        )
        zones_dict.setdefault(zone_num, []).append(ti)

    if console is not None:
        total_matched = sum(len(t) for t in zones_dict.values())
        zone_summary = ", ".join(
            f"zone {z}: {len(t)}" for z, t in sorted(zones_dict.items())
        )
        console.print(
            f"  {total_matched} tiles in {len(zones_dict)} zone(s): {zone_summary}"
        )

    return zones_dict


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _run_parallel(
    fn, items, workers, console=None, label="Processing", progress_callback=None
):
    """Run fn(item) in a ThreadPoolExecutor, with optional Rich progress.

    Args:
        fn: Callable that takes one item and returns a result.
        items: Iterable of items to process.
        workers: Number of threads.
        console: Optional Rich Console for progress display.
        label: Description for the progress bar.
        progress_callback: Optional callable(completed, total) called after
            each item completes.  Used for cross-process progress reporting
            when ``console`` is not available.

    Returns:
        List of (item, result) tuples for successful calls.
        Failed calls are logged and skipped.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = list(items)
    results = []
    completed = 0

    def _execute(pool):
        nonlocal completed
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append((item, future.result()))
            except Exception as e:
                logger.warning(f"{label} failed for {item}: {e}")
            completed += 1
            yield item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if console is not None:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(label, total=len(items))
                for _ in _execute(pool):
                    progress.advance(task)
        else:
            for _ in _execute(pool):
                if progress_callback is not None:
                    progress_callback(completed, len(items))

    return results


# ---------------------------------------------------------------------------
# Global preview helpers
# ---------------------------------------------------------------------------


def _preview_marker_path(store_path: Path, zone_num: int) -> Path:
    """Resume marker for a zone's global-preview reprojection.

    Kept in the state sibling (``<store>.build/_preview/``) rather than the
    store, for the same reason as the ingestion registry: the published Zarr
    hierarchy should contain only Zarr.
    """
    return Path(f"{str(store_path).rstrip('/')}.build") / "_preview" / (
        f"zone_{zone_num}_done"
    )


def _zone_output_bounds(
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
) -> Tuple[int, int, int, int]:
    """Compute the chunk-aligned output bounds for a zone in global grid pixels.

    Returns (row_start, row_end, col_start, col_end).
    """
    from pyproj import Transformer

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    west, _south, _east, north = GLOBAL_BOUNDS

    to_4326 = Transformer.from_crs(
        f"EPSG:{zone_epsg}",
        "EPSG:4326",
        always_xy=True,
    )
    corners_utm = [
        (src_origin_e, src_origin_n),
        (src_origin_e + src_w * src_pixel, src_origin_n),
        (src_origin_e, src_origin_n - src_h * src_pixel),
        (src_origin_e + src_w * src_pixel, src_origin_n - src_h * src_pixel),
    ]
    mid_e = src_origin_e + src_w * src_pixel / 2
    mid_n = src_origin_n - src_h * src_pixel / 2
    corners_utm += [
        (mid_e, src_origin_n),
        (mid_e, src_origin_n - src_h * src_pixel),
        (src_origin_e, mid_n),
        (src_origin_e + src_w * src_pixel, mid_n),
    ]
    corners_4326 = [to_4326.transform(e, n) for e, n in corners_utm]
    lons = [c[0] for c in corners_4326]
    lats = [c[1] for c in corners_4326]

    zlon_min, zlon_max = min(lons), max(lons)
    zlat_min, zlat_max = min(lats), max(lats)

    col_start = max(
        0,
        (
            int(math.floor((zlon_min - west) / GLOBAL_BASE_RES))
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    col_end = min(
        GLOBAL_LEVEL0_W,
        (
            (int(math.ceil((zlon_max - west) / GLOBAL_BASE_RES)) + GLOBAL_CHUNK - 1)
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    row_start = max(
        0,
        (
            int(math.floor((north - zlat_max) / GLOBAL_BASE_RES))
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    row_end = min(
        GLOBAL_LEVEL0_H,
        (
            (int(math.ceil((north - zlat_min) / GLOBAL_BASE_RES)) + GLOBAL_CHUNK - 1)
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )

    return (row_start, row_end, col_start, col_end)


def _coarsen_zone_pyramid(
    store_path: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    num_levels: int,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Update pyramid levels 1 through num_levels-1 for the affected region.

    Reads from the previous level and writes coarsened data to the current
    level, processing in 2D tiles parallelised with a thread pool.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import zarr

    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )

    prev_row_start, prev_row_end = row_start, row_end
    prev_col_start, prev_col_end = col_start, col_end

    for lvl in range(1, num_levels):
        prev_arr_path = f"global_rgb/{lvl - 1}/rgb"
        cur_arr_path = f"global_rgb/{lvl}/rgb"

        if prev_arr_path not in root or cur_arr_path not in root:
            break

        prev_arr = root[prev_arr_path]
        cur_arr = root[cur_arr_path]
        cur_h, cur_w = cur_arr.shape[:2]

        lr_start = max(0, (prev_row_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lr_end = min(
            cur_h,
            ((prev_row_end // 2 + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK),
        )
        lc_start = max(0, (prev_col_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lc_end = min(
            cur_w,
            ((prev_col_end // 2 + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK),
        )

        if lr_end <= lr_start or lc_end <= lc_start:
            break

        if console is not None:
            console.print(
                f"    Level {lvl}: rows {lr_start}-{lr_end}, cols {lc_start}-{lc_end}"
            )

        tile_size = GLOBAL_CHUNK  # output tile dimension

        def _coarsen_tile(
            r0, c0, _prev_arr=prev_arr, _cur_arr=cur_arr, _cur_h=cur_h, _cur_w=cur_w
        ):
            r1 = min(r0 + tile_size, _cur_h)
            c1 = min(c0 + tile_size, _cur_w)
            sr0 = r0 * 2
            sr1 = min(sr0 + (r1 - r0) * 2, _prev_arr.shape[0])
            sc0 = c0 * 2
            sc1 = min(sc0 + (c1 - c0) * 2, _prev_arr.shape[1])
            tile = np.asarray(_prev_arr[sr0:sr1, sc0:sc1, :]).astype(np.float32)
            th = tile.shape[0] // 2
            tw = tile.shape[1] // 2
            if th == 0 or tw == 0:
                return
            coarsened = (
                tile[: th * 2, : tw * 2, :]
                .reshape(th, 2, tw, 2, GLOBAL_NUM_BANDS)
                .mean(axis=(1, 3))
            )
            result = np.clip(coarsened, 0, 255).astype(np.uint8)
            _cur_arr[r0 : r0 + th, c0 : c0 + tw, :] = result

        tile_args = [
            (r0, c0)
            for r0 in range(lr_start, lr_end, tile_size)
            for c0 in range(lc_start, lc_end, tile_size)
        ]

        if console is not None:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                ptask = progress.add_task(
                    f"Pyramid level {lvl}",
                    total=len(tile_args),
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_coarsen_tile, r0, c0): (r0, c0)
                        for r0, c0 in tile_args
                    }
                    for future in as_completed(futures):
                        future.result()
                        progress.advance(ptask)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_coarsen_tile, r0, c0) for r0, c0 in tile_args]
                for future in as_completed(futures):
                    future.result()

        prev_row_start, prev_row_end = lr_start, lr_end
        prev_col_start, prev_col_end = lc_start, lc_end


# ---------------------------------------------------------------------------
# Store constants
# ---------------------------------------------------------------------------

SHARD_SIZE = 4096  # spatial pixels per shard side
INNER_CHUNK = 32  # spatial pixels per inner chunk side

# Default per-(zone, year) capacity of the raw pixel sample kept for the
# global stretch quantiles (docs/specs/zarr-stretch-stats.md).
STRETCH_SAMPLE_K = 20_000
DEFAULT_WORKERS = 4  # fewer workers due to larger shard buffers (~2GB each)

# Each shard worker holds a full (N_BANDS, SHARD_SIZE, SHARD_SIZE) int8
# buffer plus its float32 scales — the dominant cost of a fill, and the
# reason the worker count is bounded by RAM rather than by cores.
WORKER_BUFFER_BYTES = N_BANDS * SHARD_SIZE * SHARD_SIZE + 4 * SHARD_SIZE * SHARD_SIZE

# Peak is several times that buffer, and planning against the raw figure
# gets fills OOM-killed. On top of the buffer, zarr's sharding codec
# compresses every inner chunk and assembles the shard, and s3fs holds the
# upload body. Measured 4.3 GiB per worker on a sparse three-tile shard; a
# dense one on a loaded host was killed holding 12 GiB.
WORKER_PEAK_BYTES = 3 * WORKER_BUFFER_BYTES

# With --spill-dir the shard buffers are memory-mapped, so their pages are
# reclaimable page cache rather than anonymous memory. Measured on Linux,
# that takes the buffer's contribution to RssAnon from 1.73 GiB to 0.35 GiB.
WORKER_PEAK_BYTES_SPILLED = WORKER_PEAK_BYTES - WORKER_BUFFER_BYTES


def _total_memory_bytes() -> Optional[int]:
    """Memory a fill can realistically use, or None if undeterminable.

    Prefers MemAvailable over physical RAM: these hosts are often shared,
    and what is free right now is what decides whether the kernel starts
    killing workers.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _warn_worker_memory(workers: int, console=None, spilled: bool = False) -> None:
    """Warn when the requested worker count cannot fit in RAM.

    A fill that is OOM-killed leaves no traceback, so the cause is easy to
    miss; say it up front instead.
    """
    per_worker = WORKER_PEAK_BYTES_SPILLED if spilled else WORKER_PEAK_BYTES
    needed = workers * per_worker
    total = _total_memory_bytes()
    gib = 2**30
    message = (
        f"{workers} workers can peak around {needed / gib:.0f} GiB "
        f"(~{per_worker / gib:.1f} GiB each"
        + (
            ", shard buffers spilled to disk)"
            if spilled
            else f": a {WORKER_BUFFER_BYTES / gib:.1f} GiB shard buffer plus "
            f"compression and the upload body)"
        )
    )
    if total is None:
        logger.info(message)
        return
    if needed > 0.8 * total:
        safe = max(1, int(0.8 * total // per_worker))
        hint = "" if spilled else " (or --spill-dir to cut ~2 GiB per worker)"
        text = (
            f"{message}, but only {total / gib:.0f} GiB is available here. "
            f"The fill will likely be OOM-killed — consider "
            f"--workers {safe}{hint}."
        )
        if console:
            console.print(f"  [yellow]{text}[/yellow]")
        else:
            logger.warning(text)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class UnifiedZoneGrid:
    """Describes the pixel grid for a UTM zone spanning all years."""

    zone: int
    years: List[int]
    canonical_epsg: int
    origin_x: float  # UTM easting of top-left corner
    origin_y: float  # UTM northing of top-left corner
    width_px: int
    height_px: int
    pixel_size: float = 10.0


def _tile_pixel_offset(
    tile_info: TileInfo,
    grid: UnifiedZoneGrid,
) -> Tuple[int, int]:
    """Pixel offset of a tile within the unified zone grid."""
    tile_x = tile_info.transform.c
    tile_y = northing_to_canonical(tile_info.transform.f, tile_info.epsg)
    col = round((tile_x - grid.origin_x) / grid.pixel_size)
    row = round((grid.origin_y - tile_y) / grid.pixel_size)
    return row, col


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------


def shard_coords_for_tiles(
    tile_infos: List[TileInfo],
    grid: UnifiedZoneGrid,
) -> set:
    """Return the set of (shard_row, shard_col) covered by these tiles."""
    coords = set()
    for ti in tile_infos:
        row, col = _tile_pixel_offset(ti, grid)
        for sr in range(row // SHARD_SIZE, (row + ti.height - 1) // SHARD_SIZE + 1):
            for sc in range(col // SHARD_SIZE, (col + ti.width - 1) // SHARD_SIZE + 1):
                coords.add((sr, sc))
    return coords


def build_shard_index(
    tile_infos: List[TileInfo],
    grid: UnifiedZoneGrid,
    time_index: int,
    restrict_to: Optional[set] = None,
) -> List[ShardSpec]:
    """Build shard index for one year's tiles against a unified zone grid.

    Args:
        restrict_to: Optional set of (shard_row, shard_col) to emit specs for.
            A shard write replaces the whole shard, so an incremental fill has
            to pass *every* tile overlapping the shards it rewrites — not just
            the new ones — or previously written neighbours are zeroed out.
            Callers get that by passing all of the zone's tiles here along with
            the shard coordinates the new tiles touch.
    """
    shard_map: Dict[Tuple[int, int], List[ShardTileOverlap]] = {}

    for ti in tile_infos:
        row, col = _tile_pixel_offset(ti, grid)
        h, w = ti.height, ti.width

        sr_start = row // SHARD_SIZE
        sr_end = (row + h - 1) // SHARD_SIZE
        sc_start = col // SHARD_SIZE
        sc_end = (col + w - 1) // SHARD_SIZE

        for sr in range(sr_start, sr_end + 1):
            for sc in range(sc_start, sc_end + 1):
                if restrict_to is not None and (sr, sc) not in restrict_to:
                    continue
                shard_top = sr * SHARD_SIZE
                shard_left = sc * SHARD_SIZE

                t_row_start = max(0, shard_top - row)
                t_row_end = min(h, shard_top + SHARD_SIZE - row)
                t_col_start = max(0, shard_left - col)
                t_col_end = min(w, shard_left + SHARD_SIZE - col)

                s_row_start = max(0, row - shard_top)
                s_row_end = s_row_start + (t_row_end - t_row_start)
                s_col_start = max(0, col - shard_left)
                s_col_end = s_col_start + (t_col_end - t_col_start)

                ov = ShardTileOverlap(
                    embedding_path=ti.embedding_path,
                    scales_path=ti.scales_path,
                    landmask_path=ti.landmask_path,
                    t_row_start=t_row_start,
                    t_row_end=t_row_end,
                    t_col_start=t_col_start,
                    t_col_end=t_col_end,
                    s_row_start=s_row_start,
                    s_row_end=s_row_end,
                    s_col_start=s_col_start,
                    s_col_end=s_col_end,
                )
                shard_map.setdefault((sr, sc), []).append(ov)

    specs = []
    for (sr, sc), overlaps in sorted(shard_map.items()):
        specs.append(
            ShardSpec(
                sr=sr,
                sc=sc,
                row_px=sr * SHARD_SIZE,
                col_px=sc * SHARD_SIZE,
                time_index=time_index,
                tiles=overlaps,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Store initialisation (zarr-init)
# ---------------------------------------------------------------------------


def _gather_landmask_tiles_by_zone(
    registry: "Registry",
) -> Dict[int, List[Tuple[float, float]]]:
    """Group landmask tile coordinates by UTM zone.

    Returns dict mapping zone number to list of (lon, lat) centres.
    """
    tiles = registry.available_landmasks  # [(lon, lat), ...]
    by_zone: Dict[int, List[Tuple[float, float]]] = {}
    for lon, lat in tiles:
        zone_num = int(math.floor((lon + 180) / 6)) + 1
        zone_num = max(1, min(60, zone_num))
        by_zone.setdefault(zone_num, []).append((lon, lat))
    return by_zone


def _compute_zone_grid_from_landmask(
    zone: int,
    tile_coords: List[Tuple[float, float]],
    years: List[int],
) -> UnifiedZoneGrid:
    """Compute a unified zone grid from landmask tile coordinates.

    Projects tile bounding boxes to UTM and computes the union extent,
    snapped to SHARD_SIZE boundaries.
    """
    from pyproj import Transformer

    epsg = zone_canonical_epsg(zone)
    pixel_size = 10.0
    proj = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for lon, lat in tile_coords:
        # Each tile is 0.1 degrees
        west, east = lon - 0.05, lon + 0.05
        south, north = lat - 0.05, lat + 0.05

        corners_x, corners_y = proj.transform(
            [west, east, west, east],
            [north, north, south, south],
        )
        min_x = min(min_x, min(corners_x))
        max_x = max(max_x, max(corners_x))
        min_y = min(min_y, min(corners_y))
        max_y = max(max_y, max(corners_y))

    origin_x = math.floor(min_x / pixel_size) * pixel_size
    origin_y = math.ceil(max_y / pixel_size) * pixel_size
    extent_right = math.ceil(max_x / pixel_size) * pixel_size
    extent_bottom = math.floor(min_y / pixel_size) * pixel_size

    width_px = round((extent_right - origin_x) / pixel_size)
    height_px = round((origin_y - extent_bottom) / pixel_size)

    width_px = math.ceil(width_px / SHARD_SIZE) * SHARD_SIZE
    height_px = math.ceil(height_px / SHARD_SIZE) * SHARD_SIZE

    return UnifiedZoneGrid(
        zone=zone,
        years=years,
        canonical_epsg=epsg,
        origin_x=origin_x,
        origin_y=origin_y,
        width_px=width_px,
        height_px=height_px,
        pixel_size=pixel_size,
    )


def init_store(
    registry: "Registry",
    output_path: "str | Path | StoreLocation",
    years: List[int],
    geotessera_version: str = "unknown",
    model_version: str = "1.0",
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    state_url: Optional[str] = None,
    stretch_sample_size: int = STRETCH_SAMPLE_K,
) -> str:
    """Create a tessera store with time dimension from the landmask registry.

    Creates all UTM zones that have landmask coverage.  For each zone, the
    grid extent is computed from landmask tiles (not embeddings), so only
    the landmask registry is needed.

    The scales array is initialised with sentinels:
    - NaN  = water (permanent, from landmask)
    - +inf = land, no data yet (replaced by real scale values during fill)

    No embedding data is written.  The embeddings array stays at fill_value (0).

    ``output_path`` may be a local path or an fsspec URL such as
    ``s3://bucket/tessera.zarr``; the whole store is metadata-only at this
    point, so initialising directly on the target object store is cheap.
    """
    import zarr

    store = StoreLocation.resolve(output_path, storage_options, state_url)
    if not store.is_remote:
        if Path(store.url).exists():
            raise FileExistsError(f"Store already exists: {store}")
    else:
        # Write-scoped credentials commonly cannot list the prefix, and S3
        # then answers 403 rather than 404 for the key we are probing. That
        # is not a reason to refuse to create a store — say so and carry on.
        try:
            if store.exists("zarr.json"):
                raise FileExistsError(f"Store already exists: {store}")
        except PermissionError:
            if console:
                console.print(
                    "  [yellow]Could not check whether a store already exists "
                    "here (permission denied on HEAD — credentials without "
                    "s3:ListBucket get 403 instead of 404 for a missing key). "
                    "Creating it; an existing store's root would be "
                    "overwritten.[/yellow]"
                )

    years = sorted(years)
    T = len(years)

    if console:
        console.print(f"Initialising store at [bold]{store}[/bold]")
        console.print(f"  Years: {years[0]}-{years[-1]} ({T} time steps)")

    # Get landmask coverage grouped by UTM zone
    landmask_by_zone = _gather_landmask_tiles_by_zone(registry)

    if not landmask_by_zone:
        raise ValueError("No landmask tiles found in registry")

    if console:
        console.print(f"  {len(landmask_by_zone)} zone(s) with land coverage")

    # Create root group via zarr API (not manual JSON) so consolidation
    # preserves attributes correctly. Mode "w-" creates but never clobbers:
    # "w" would delete the destination prefix first, which needs list and
    # delete permissions and would destroy an existing store if our
    # pre-check above could not see it.
    from zarr.errors import ContainsArrayError, ContainsGroupError

    try:
        root = store.open_group(mode="w-", zarr_format=3, use_consolidated=None)
    except (ContainsGroupError, ContainsArrayError) as e:
        raise FileExistsError(f"Store already exists: {store}") from e
    root.attrs.update(
        {
            "zarr_conventions": [GEOEMB_CONVENTION],
            "geoemb:type": "pixel",
            "geoemb:dimensions": N_BANDS,
            "geoemb:model": f"https://geotessera.org/model/{model_version}",
            "geoemb:source_data": [
                "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
                "https://sentinel.esa.int/web/sentinel/missions/sentinel-2",
            ],
            "geoemb:data_type": "int8",
            "geoemb:gsd": 10.0,
            "geoemb:spatial_layout": "utm_zones",
            "geoemb:build_version": geotessera_version,
            "geoemb:quantization": {
                "method": "per_pixel_scale",
                "original_dtype": "float32",
                "quantized_dtype": "int8",
                "scale": {
                    "type": "array",
                    "array_name": "scales",
                    "nodata": "+inf",
                },
            },
        }
    )

    # Create each zone group from landmask coverage
    for zone_num in sorted(landmask_by_zone.keys()):
        tile_coords = landmask_by_zone[zone_num]
        grid = _compute_zone_grid_from_landmask(zone_num, tile_coords, years)

        if console:
            w_km = grid.width_px * grid.pixel_size / 1000
            h_km = grid.height_px * grid.pixel_size / 1000
            n_shards_x = grid.width_px // SHARD_SIZE
            n_shards_y = grid.height_px // SHARD_SIZE
            console.print(
                f"  Zone {zone_num} "
                f"[dim]EPSG:{grid.canonical_epsg}[/dim] "
                f"[dim]{grid.width_px}x{grid.height_px}px "
                f"({w_km:.0f}x{h_km:.0f}km) "
                f"{n_shards_x}x{n_shards_y} shards[/dim]"
            )

        _create_zone_group(grid, store, stretch_sample_size)

    # Nothing else is written into the store: ingestion tracking and locks
    # are build state and live in the state sibling, created on first fill.

    # Consolidate metadata so HTTP readers can discover the hierarchy
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        warnings.filterwarnings("ignore", message="Object at .* is not recognized")
        zarr.consolidate_metadata(store.as_zarr_store())

    if console:
        console.print(
            "  [green]Store initialised (metadata only, no data written)[/green]"
        )

    return store.url


def _create_zone_group(
    grid: UnifiedZoneGrid,
    store_location: StoreLocation,
    stretch_sample_size: int = STRETCH_SAMPLE_K,
) -> "zarr.Group":
    """Create a zone group with empty (T, B, H, W) arrays."""
    from zarr.codecs import BloscCodec

    zone_group = _zone_group_name(grid.zone)

    root_reopen = store_location.open_group(mode="r+", zarr_format=3)
    store = root_reopen.create_group(zone_group)

    T = len(grid.years)
    H = grid.height_px
    W = grid.width_px

    # Main data arrays — (T, B, H, W) layout
    store.create_array(
        "embeddings",
        shape=(T, N_BANDS, H, W),
        chunks=(1, N_BANDS, INNER_CHUNK, INNER_CHUNK),
        shards=(1, N_BANDS, SHARD_SIZE, SHARD_SIZE),
        dtype=np.int8,
        fill_value=np.int8(0),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "band", "y", "x"],
    )
    # fill_value=+inf means unwritten land pixels read as "no data yet".
    # Water pixels are written as NaN during zarr-fill (from landmask).
    # Clients: isinf(scales) → land/no-data, isnan(scales) → water,
    #          isfinite(scales) → valid embedding data.
    store.create_array(
        "scales",
        shape=(T, H, W),
        chunks=(1, INNER_CHUNK, INNER_CHUNK),
        shards=(1, SHARD_SIZE, SHARD_SIZE),
        dtype=np.float32,
        fill_value=np.float32("inf"),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "y", "x"],
    )

    # Coordinate arrays
    x_coords = grid.origin_x + (np.arange(W) + 0.5) * grid.pixel_size
    y_coords = grid.origin_y - (np.arange(H) + 0.5) * grid.pixel_size
    time_coords = np.array(grid.years, dtype=np.int32)
    band_coords = np.arange(N_BANDS, dtype=np.int32)

    for name, data, dim in [
        ("x", x_coords, "x"),
        ("y", y_coords, "y"),
        ("time", time_coords, "time"),
        ("band", band_coords, "band"),
    ]:
        store.create_array(
            name,
            shape=data.shape,
            dtype=data.dtype,
            fill_value=0,
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=[dim],
        )
        store[name][:] = data

    # Per-zone stretch statistics, populated by zarr-fill (see
    # docs/specs/zarr-stretch-stats.md). Plain arrays: zone groups carry no
    # geoemb: attributes.
    create_stretch_arrays(store, T, stretch_sample_size)

    # Use geozarr-toolkit for proj: and spatial: convention metadata
    from geozarr_toolkit import create_geozarr_attrs

    x_min = grid.origin_x
    x_max = grid.origin_x + W * grid.pixel_size
    y_max = grid.origin_y
    y_min = grid.origin_y - H * grid.pixel_size

    geozarr_attrs = create_geozarr_attrs(
        dimensions=["y", "x"],
        crs=f"EPSG:{grid.canonical_epsg}",
        transform=[
            grid.pixel_size,
            0.0,
            grid.origin_x,
            0.0,
            -grid.pixel_size,
            grid.origin_y,
        ],
        bbox=[x_min, y_min, x_max, y_max],
        shape=[H, W],
        registration="pixel",
    )

    # Fix convention descriptions to match upstream schemas exactly
    # (geozarr-toolkit has a bug: "Spatial coordinate and transformation
    # information" instead of "Spatial coordinate information")
    for conv in geozarr_attrs.get("zarr_conventions", []):
        if conv.get("uuid") == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4":
            conv["description"] = "Spatial coordinate information"

    # Zone groups only carry proj: and spatial: conventions (geoemb: is on root)

    store.attrs.update(geozarr_attrs)

    return store


# ---------------------------------------------------------------------------
# Stretch statistics (per-zone, collected at fill time)
# ---------------------------------------------------------------------------
# The global RGB stretch needs a mean/covariance (for PCA) and quantiles (for
# the percentile stretch and equalisation CDF) over every valid pixel in the
# store. Recomputing those by re-reading shards costs terabytes; instead each
# fill records, per (zone, year), the exact sufficient statistics for the
# covariance — which are additive across zones — plus a weighted raw-pixel
# sample for the quantiles, which are not.
#
# These live as ordinary arrays inside each zone group. Zone groups carry no
# geoemb: attributes (the convention keeps those on the root), so the filled
# sample count is itself an array rather than an attr.

STRETCH_ARRAY_NAMES = (
    "stretch_stats_count",
    "stretch_stats_sum",
    "stretch_stats_prod",
    "stretch_sample",
    "stretch_sample_scales",
    "stretch_sample_count",
)


def create_stretch_arrays(group: "zarr.Group", n_years: int, k: int) -> None:
    """Create the per-zone stretch-statistics arrays in *group*.

    One chunk per year on the time axis, so a (zone, year) update touches
    exactly one chunk per array and ``zarr-extend`` grows them the same way
    it grows ``embeddings``.
    """
    from zarr.codecs import BloscCodec

    comp = BloscCodec(cname="zstd", clevel=3)
    T = n_years
    specs = [
        ("stretch_stats_count", (T,), (1,), np.int64, 0, ["time"]),
        ("stretch_stats_sum", (T, N_BANDS), (1, N_BANDS), np.float64, 0.0,
         ["time", "band"]),
        ("stretch_stats_prod", (T, N_BANDS, N_BANDS), (1, N_BANDS, N_BANDS),
         np.float64, 0.0, ["time", "band", "band2"]),
        ("stretch_sample", (T, k, N_BANDS), (1, k, N_BANDS), np.int8,
         np.int8(0), ["time", "sample", "band"]),
        # +inf matches the "land, no data" sentinel, so padding slots can
        # never be mistaken for real pixels even by a reader that ignores
        # stretch_sample_count.
        ("stretch_sample_scales", (T, k), (1, k), np.float32,
         np.float32("inf"), ["time", "sample"]),
        ("stretch_sample_count", (T,), (1,), np.int64, 0, ["time"]),
    ]
    for name, shape, chunks, dtype, fill, dims in specs:
        group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill,
            compressors=comp,
            dimension_names=dims,
        )


def _shard_sample_cap(k_slots: int, n_shards: int) -> int:
    """Per-shard sample size: a few times K spread over the shards.

    Oversampling by 4x gives the weighted merge enough candidates to
    approximate a uniform draw without ballooning the result queue.
    """
    return min(k_slots, max(64, -(-4 * k_slots // max(1, n_shards))))


def shard_stretch_stats(
    emb_buf: np.ndarray,
    scales_buf: np.ndarray,
    sample_cap: int,
    seed: Optional[int] = None,
    block: int = 262_144,
) -> Optional[Dict[str, Any]]:
    """Exact (n, S, M) sufficient statistics plus a pixel sample for one shard.

    Works on the shard buffers the fill already holds: ``emb_buf`` is
    ``(B, S, S)`` int8, ``scales_buf`` ``(S, S)`` float32.  Valid pixels are
    those with finite scale.  The sum-of-products matrix is accumulated in
    float64 from float32 block GEMMs — each block sums ~2.6e5 terms of O(1)
    magnitude, so the block partials carry ~7 significant digits and the
    float64 accumulation loses nothing that a covariance of 1e9 pixels could
    show.

    Returns None when the shard has no valid pixels.
    """
    valid = np.isfinite(scales_buf)
    flat = np.flatnonzero(valid.ravel())
    n = int(flat.size)
    if n == 0:
        return None

    emb_flat = emb_buf.reshape(emb_buf.shape[0], -1)
    scales_flat = scales_buf.ravel()

    s = np.zeros(N_BANDS, dtype=np.float64)
    m = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
    for i in range(0, n, block):
        idx = flat[i : i + block]
        xb = emb_flat[:, idx].astype(np.float32) * scales_flat[idx].astype(np.float32)
        s += xb.sum(axis=1, dtype=np.float64)
        m += (xb @ xb.T).astype(np.float64)

    rng = np.random.default_rng(seed)
    k = min(sample_cap, n)
    pick = flat[rng.choice(n, size=k, replace=False)]
    return {
        "n": n,
        "sum": s,
        "prod": m,
        "sample_emb": np.ascontiguousarray(emb_flat[:, pick].T),  # (k, B) int8
        "sample_scales": scales_flat[pick].astype(np.float32),
        # Each returned row stands for n/k pixels of the shard's population.
        "sample_weight": n / k,
    }


def merge_stretch_samples(
    candidates: List[Tuple[np.ndarray, np.ndarray, float]],
    k: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw K rows from weighted candidate pools (Efraimidis–Spirakis).

    ``candidates`` is a list of ``(emb (n, B) int8, scales (n,) f32, weight
    per row)``.  Rows are selected with probability proportional to their
    weight, without replacement, so pooling per-shard samples of different
    coverage reproduces a uniform draw over the union population.
    """
    embs = [c[0] for c in candidates if len(c[0])]
    if not embs:
        return (
            np.zeros((0, N_BANDS), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
        )
    emb = np.concatenate(embs, axis=0)
    scales = np.concatenate([c[1] for c in candidates if len(c[0])], axis=0)
    weights = np.concatenate(
        [np.full(len(c[0]), max(c[2], 1e-12)) for c in candidates if len(c[0])]
    )

    if len(emb) <= k:
        return emb, scales

    rng = np.random.default_rng(seed)
    keys = rng.random(len(emb)) ** (1.0 / weights)
    top = np.argpartition(keys, -k)[-k:]
    return emb[top], scales[top]


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, qs: np.ndarray
) -> np.ndarray:
    """Percentiles of a weighted sample (qs in 0..100)."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order].astype(np.float64)
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= w.sum()
    return np.interp(np.asarray(qs, dtype=np.float64) / 100.0, cdf, v)


def update_zone_stretch_stats(
    zone_group: "zarr.Group",
    time_index: int,
    n: int,
    s: np.ndarray,
    m: np.ndarray,
    sample_candidates: List[Tuple[np.ndarray, np.ndarray, float]],
    seed: Optional[int] = None,
) -> None:
    """Fold one fill run's statistics into a zone's arrays (read-modify-write).

    The additive triple is summed onto what is stored; the sample is re-drawn
    from the stored sample and the new candidates together, weighted so the
    result still approximates a uniform draw over all pixels either has seen.
    Caller must hold the (zone, year) fill lock — this is the same
    single-writer context the shard writes ran under.
    """
    t = time_index
    count_arr = zone_group["stretch_stats_count"]
    prev_n = int(count_arr[t])

    count_arr[t] = prev_n + n
    zone_group["stretch_stats_sum"][t] = (
        np.asarray(zone_group["stretch_stats_sum"][t]) + s
    )
    zone_group["stretch_stats_prod"][t] = (
        np.asarray(zone_group["stretch_stats_prod"][t]) + m
    )

    k = zone_group["stretch_sample"].shape[1]
    stored_k = int(zone_group["stretch_sample_count"][t])
    pool = list(sample_candidates)
    if stored_k > 0:
        pool.append(
            (
                np.asarray(zone_group["stretch_sample"][t, :stored_k]),
                np.asarray(zone_group["stretch_sample_scales"][t, :stored_k]),
                max(prev_n, 1) / stored_k,
            )
        )
    emb, scales = merge_stretch_samples(pool, k, seed=seed)

    filled = len(emb)
    if filled:
        zone_group["stretch_sample"][t, :filled] = emb
        zone_group["stretch_sample_scales"][t, :filled] = scales
    zone_group["stretch_sample_count"][t] = filled


# ---------------------------------------------------------------------------
# Tile registry (GeoParquet tracking which tiles are written)
# ---------------------------------------------------------------------------
# This is build bookkeeping, so it lives in the state sibling rather than the
# store: an incremental fill needs to know which tiles it already wrote, but a
# reader of the published store never does. Tracking is sharded by (zone,
# year) under ``_registry/`` so that concurrent per-zone fills never
# read-modify-write the same object, and ``consolidate_store`` merges the
# parts into one ``_registry.parquet``.
#
# Stores built before the split keep a single ``_registry.parquet`` at the
# store root; it is still read so those stores resume correctly, but nothing
# is written back into the store any more.

REGISTRY_DIR_NAME = "_registry"
MERGED_REGISTRY_NAME = "_registry.parquet"
LEGACY_REGISTRY_NAME = MERGED_REGISTRY_NAME


def _zone_registry_name(zone: int, year: int) -> str:
    return f"{_zone_group_name(zone)}_{year}.parquet"


def _empty_tile_registry() -> "geopandas.GeoDataFrame":
    """An empty registry frame with the canonical schema."""
    import geopandas as gpd
    import pandas as pd

    return gpd.GeoDataFrame(
        {
            "year": pd.array([], dtype="int32"),
            "zone": pd.array([], dtype="int32"),
            "tile_lon": pd.array([], dtype="float64"),
            "tile_lat": pd.array([], dtype="float64"),
            "written_at": pd.array([], dtype="datetime64[ns, UTC]"),
            "geometry": gpd.array.GeometryArray(
                gpd.points_from_xy([], []),
            ),
        },
        crs="EPSG:4326",
    )


def _read_parquet_at(store: StoreLocation, *parts: str):
    """Read a GeoParquet object from the store, or None if absent.

    Reads straight through rather than probing first: it halves the round
    trips, and an existence probe is unreliable anyway against credentials
    without list permission, where a missing key answers 403 not 404.
    """
    import geopandas as gpd

    try:
        data = store.read_bytes(*parts)
    except (FileNotFoundError, PermissionError):
        return None
    except Exception as e:
        logger.warning(f"Could not read {store.join(*parts)}: {e}")
        return None

    try:
        return gpd.read_parquet(io.BytesIO(data))
    except Exception as e:
        logger.warning(f"Could not parse {store.join(*parts)}: {e}")
        return None


def _write_parquet_at(store: StoreLocation, gdf, *parts: str) -> None:
    """Write a GeoParquet object into the store (local path or remote URL)."""
    buf = io.BytesIO()
    gdf.to_parquet(buf)
    store.write_bytes(buf.getvalue(), *parts)


_MERGED_UNSET = object()


def load_merged_registry(store: StoreLocation):
    """Read the merged ingestion registry, from the state dir or an old store.

    Returns None when neither exists (a store nobody has filled yet).
    """
    merged = _read_parquet_at(store.state, MERGED_REGISTRY_NAME)
    if merged is not None:
        return merged
    # Stores built before the split kept it inside the Zarr hierarchy.
    return _read_parquet_at(store, LEGACY_REGISTRY_NAME)


def _get_written_tiles(
    store: StoreLocation,
    year: int,
    zone: int,
    merged=_MERGED_UNSET,
) -> set:
    """Return set of (tile_lon, tile_lat) already written for a year/zone.

    Reads this zone/year's own tracking file, then unions in any rows the
    merged registry holds for the same zone/year — including one left inside
    an older store, so those resume correctly too.

    Args:
        merged: A pre-loaded merged registry frame (or None if there isn't
            one). It covers the whole store, so a multi-zone fill should read
            it once and pass it in rather than re-fetching it per zone/year.
    """
    written: set = set()

    zone_gdf = _read_parquet_at(
        store.state, REGISTRY_DIR_NAME, _zone_registry_name(zone, year)
    )
    if zone_gdf is not None and not zone_gdf.empty:
        written |= set(zip(zone_gdf["tile_lon"], zone_gdf["tile_lat"]))

    if merged is _MERGED_UNSET:
        merged = load_merged_registry(store)
    if merged is not None and not merged.empty:
        mask = (merged["year"] == year) & (merged["zone"] == zone)
        subset = merged[mask]
        written |= set(zip(subset["tile_lon"], subset["tile_lat"]))

    return written


def _record_written_tiles(
    store: StoreLocation,
    tile_infos: List[TileInfo],
    year: int,
    zone: int,
) -> None:
    """Append newly written tiles to this zone/year's tracking file.

    Single-writer by construction: only the process filling (zone, year)
    touches this object, so parallel zone sweeps need no locking here.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    now = pd.Timestamp.now(tz="UTC")
    rows = [
        {
            "year": np.int32(year),
            "zone": np.int32(zone),
            "tile_lon": ti.lon,
            "tile_lat": ti.lat,
            "written_at": now,
            "geometry": Point(ti.lon, ti.lat),
        }
        for ti in tile_infos
    ]
    if not rows:
        return

    new_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    name = _zone_registry_name(zone, year)
    existing = _read_parquet_at(store.state, REGISTRY_DIR_NAME, name)

    if existing is not None and not existing.empty:
        combined = gpd.GeoDataFrame(
            pd.concat([existing, new_gdf], ignore_index=True), crs="EPSG:4326"
        ).drop_duplicates(subset=["year", "zone", "tile_lon", "tile_lat"], keep="last")
    else:
        combined = new_gdf

    _write_parquet_at(store.state, combined, REGISTRY_DIR_NAME, name)


def merge_tile_registry(
    store: StoreLocation,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Merge every per-zone tracking file into one ``_registry.parquet``.

    Both the parts and the result live in the state sibling, never inside the
    Zarr hierarchy. Run this once after a parallel sweep, when no fill is in
    flight — it is the only step that rewrites a store-wide object. Returns
    the total row count in the merged registry.
    """
    import geopandas as gpd
    import pandas as pd
    from . import remote

    state = store.state
    frames = []
    previous = load_merged_registry(store)
    if previous is not None and not previous.empty:
        frames.append(previous)

    n_parts = 0
    for entry in state.listdir(REGISTRY_DIR_NAME, on_denied=[]):
        if not entry.endswith(".parquet"):
            continue
        try:
            data = remote.read_bytes(entry, state.storage_options)
            part = gpd.read_parquet(io.BytesIO(data))
        except Exception as e:
            logger.warning(f"Skipping unreadable registry part {entry}: {e}")
            continue
        n_parts += 1
        if not part.empty:
            frames.append(part)

    if not frames:
        merged = _empty_tile_registry()
    else:
        merged = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs="EPSG:4326"
        ).drop_duplicates(subset=["year", "zone", "tile_lon", "tile_lat"], keep="last")

    _write_parquet_at(state, merged, MERGED_REGISTRY_NAME)

    if console:
        console.print(
            f"  Merged {n_parts} per-zone registry file(s) into "
            f"{state.join(MERGED_REGISTRY_NAME)}: {len(merged):,} tiles"
        )
    return len(merged)


# ---------------------------------------------------------------------------
# Advisory zone locks
# ---------------------------------------------------------------------------
# Two processes filling the same (zone, year) would each rewrite whole shards
# from their own tile subset and silently erase each other's pixels. Object
# stores give us no atomic create, so this is advisory only — it catches the
# common accident (the same zone launched twice) rather than enforcing
# mutual exclusion.

LOCK_DIR_NAME = "_locks"


def _lock_name(zone: int, year: int) -> str:
    return f"{_zone_group_name(zone)}_{year}.json"


def _acquire_zone_lock(
    store: StoreLocation, zone: int, year: int, force: bool = False
) -> None:
    """Claim (zone, year) for this process, or raise if someone else holds it."""
    import json
    import socket
    import pandas as pd

    state = store.state
    name = _lock_name(zone, year)
    if not force and state.exists(LOCK_DIR_NAME, name, on_denied=False):
        try:
            held = json.loads(state.read_bytes(LOCK_DIR_NAME, name))
        except Exception:
            held = {}
        raise RuntimeError(
            f"Zone {zone} year {year} is locked by "
            f"{held.get('host', '?')}:{held.get('pid', '?')} "
            f"since {held.get('acquired_at', 'unknown time')}. "
            f"Another fill is in progress, or a previous one died. "
            f"Re-run with --force-lock to take it over."
        )

    payload = {
        "zone": zone,
        "year": year,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    state.write_bytes(json.dumps(payload).encode(), LOCK_DIR_NAME, name)


def _release_zone_lock(store: StoreLocation, zone: int, year: int) -> None:
    """Drop this process's claim on (zone, year)."""
    store.state.remove(LOCK_DIR_NAME, _lock_name(zone, year))


# ---------------------------------------------------------------------------
# Shard writing (NCHW layout)
# ---------------------------------------------------------------------------

_worker_store = None
_worker_source_options: Optional[Dict[str, Any]] = None
_worker_spill_dir: Optional[str] = None
_worker_sample_cap: int = 0  # 0 = stats collection off


def _init_shard_worker(
    store_url: str,
    zone_group: str,
    store_options: Optional[Dict[str, Any]] = None,
    source_options: Optional[Dict[str, Any]] = None,
    spill_dir: Optional[str] = None,
    sample_cap: int = 0,
) -> None:
    """Process pool initializer: open the zone group once per worker.

    Both option dicts are plain picklable mappings, so a worker rebuilds its
    own filesystem connections rather than inheriting an unforkable client.
    """
    global _worker_store, _worker_source_options, _worker_spill_dir
    global _worker_sample_cap

    from . import remote

    remote.quieten_dependency_logging()
    remote.reset_after_fork()
    remote.die_with_parent()

    _worker_store = StoreLocation(store_url, store_options).open_group(
        mode="r+", path=zone_group, zarr_format=3
    )
    _worker_source_options = source_options
    _worker_spill_dir = spill_dir
    _worker_sample_cap = sample_cap


def _write_one_shard(
    spec: ShardSpec,
    store: "zarr.Group",
    source_options: Optional[Dict[str, Any]] = None,
    spill_dir: Optional[str] = None,
    sample_cap: int = 0,
) -> "bool | Dict[str, Any]":
    """Write one shard in NCHW layout: (T, B, H, W).

    Tile reads go through :mod:`geotessera.remote`, so ``spec`` may reference
    either local paths or remote URLs — a remote tile costs one ranged GET for
    the rows this shard needs, not the whole 150 MB object.
    """
    S = SHARD_SIZE

    # Allocate BHW buffer (bands-first for NCHW write). With a spill
    # directory the two buffers are memory-mapped instead of anonymous: the
    # pages become reclaimable page cache, so the kernel evicts them under
    # pressure rather than the OOM killer taking the whole worker. Measured
    # on Linux, the buffer's contribution to RssAnon drops from 1.73 GiB to
    # 0.35 GiB.
    spill = _open_spill(spill_dir)
    if spill is None:
        emb_buf = np.zeros((N_BANDS, S, S), dtype=np.int8)
        # Start with +inf (land/nodata); landmask sets water to NaN,
        # valid tiles overwrite with finite scales.
        scales_buf = np.full((S, S), np.float32("inf"))
    else:
        emb_buf = np.memmap(
            spill / "emb.buf", dtype=np.int8, mode="w+", shape=(N_BANDS, S, S)
        )
        scales_buf = np.memmap(
            spill / "scales.buf", dtype=np.float32, mode="w+", shape=(S, S)
        )
        scales_buf[:] = np.float32("inf")

    try:
        return _fill_and_write_shard(
            spec, store, emb_buf, scales_buf, source_options, sample_cap
        )
    finally:
        del emb_buf, scales_buf
        if spill is not None:
            shutil.rmtree(spill, ignore_errors=True)


def _open_spill(spill_dir: Optional[str]):
    """Create a per-shard scratch directory, or None to stay in memory."""
    if not spill_dir:
        return None
    import tempfile

    Path(spill_dir).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="shard-", dir=spill_dir))


def _fill_and_write_shard(
    spec: ShardSpec,
    store: "zarr.Group",
    emb_buf: np.ndarray,
    scales_buf: np.ndarray,
    source_options: Optional[Dict[str, Any]] = None,
    sample_cap: int = 0,
) -> "bool | Dict[str, Any]":
    """Populate the shard buffers from their tiles and write them out.

    With ``sample_cap > 0`` the return value is the shard's stretch
    statistics (see :func:`shard_stretch_stats`) — collected here because
    this is the one moment every decoded pixel of the shard is in memory.
    """
    from . import remote

    t = spec.time_index
    S = SHARD_SIZE

    has_data = False
    for ov in spec.tiles:
        # Read HWB tile, transpose to BHW
        tile_slice = remote.read_npy_window(
            ov.embedding_path,
            ov.t_row_start,
            ov.t_row_end,
            ov.t_col_start,
            ov.t_col_end,
            storage_options=source_options,
        )
        emb_buf[
            :,
            ov.s_row_start : ov.s_row_end,
            ov.s_col_start : ov.s_col_end,
        ] = tile_slice.transpose(2, 0, 1)

        # Scales
        s = np.array(
            remote.read_npy_window(
                ov.scales_path,
                ov.t_row_start,
                ov.t_row_end,
                ov.t_col_start,
                ov.t_col_end,
                storage_options=source_options,
            ),
            dtype=np.float32,
        )

        # Landmask
        lm = _load_landmask_slice(
            ov.landmask_path,
            ov.t_row_start,
            ov.t_row_end,
            ov.t_col_start,
            ov.t_col_end,
            storage_options=source_options,
        )
        s[lm == 0] = np.float32("nan")
        s[~np.isfinite(s)] = np.float32("nan")

        scales_buf[ov.s_row_start : ov.s_row_end, ov.s_col_start : ov.s_col_end] = s
        has_data = True

    if not has_data:
        return False

    r, c = spec.row_px, spec.col_px
    store["embeddings"][t, :, r : r + S, c : c + S] = emb_buf
    store["scales"][t, r : r + S, c : c + S] = scales_buf

    if sample_cap > 0:
        stats = shard_stretch_stats(emb_buf, scales_buf, sample_cap)
        if stats is not None:
            return stats
    return True


def _write_one_shard_worker(spec: ShardSpec) -> "bool | Dict[str, Any]":
    """Picklable wrapper for process pool."""
    return _write_one_shard(
        spec,
        _worker_store,
        _worker_source_options,
        _worker_spill_dir,
        _worker_sample_cap,
    )


# ---------------------------------------------------------------------------
# Fill orchestration (zarr-fill)
# ---------------------------------------------------------------------------


def _existing_shards(
    store: StoreLocation,
    zone_group: str,
    time_index: int,
    wanted: set,
    console: Optional["rich.console.Console"] = None,
) -> set:
    """Which of *wanted* shard coordinates already exist in the store.

    A written shard is a single object under the array's chunk prefix, so its
    presence is proof the shard landed — bookkeeping that survives a ``kill
    -9`` and needs no state file. Zarr v3's default chunk key encoding puts
    the (time, band, row, col) chunk grid indices in the key, and the band
    dimension is one chunk wide, so the shard for (sr, sc) at time t is
    ``embeddings/c/{t}/0/{sr}/{sc}``.

    Prefers one listing of the time slice's prefix; falls back to probing
    each wanted shard when listing is refused, which credentials without
    ``s3:ListBucket`` will be.
    """
    prefix = f"{zone_group}/embeddings/c/{time_index}/0"

    try:
        rows = store.listdir(prefix)
    except PermissionError:
        # Credentials without s3:ListBucket. Probe each shard this fill would
        # touch instead — bounded by the work in hand, not the whole zone.
        if console:
            console.print(
                f"    [dim]Cannot list the store; probing {len(wanted)} "
                f"shard(s) individually[/dim]"
            )
        return {
            (sr, sc)
            for sr, sc in sorted(wanted)
            if store.exists(prefix, str(sr), str(sc), on_denied=False)
        }

    # An empty listing means nothing has been written for this time slice,
    # which is the common case on a fresh zone — no probing needed.
    present = set()
    for row_entry in rows:
        sr_name = row_entry.rstrip("/").rsplit("/", 1)[-1]
        if not sr_name.isdigit():
            continue
        sr = int(sr_name)
        for col_entry in store.listdir(prefix, sr_name, on_denied=[]):
            sc_name = col_entry.rstrip("/").rsplit("/", 1)[-1]
            if sc_name.isdigit() and (sr, int(sc_name)) in wanted:
                present.add((sr, int(sc_name)))
    return present


def _store_years(store: StoreLocation, zones: Optional[List[int]] = None) -> List[int]:
    """Read the store's year axis from a zone's ``time`` coordinate array.

    Tries the requested zones first so a single-zone fill against a remote
    store needs no listing of the whole hierarchy.
    """
    root = store.open_group(mode="r")

    def years_from(name: str) -> Optional[List[int]]:
        try:
            return [int(v) for v in root[name]["time"][:]]
        except Exception:
            return None

    tried = set()
    for zone in zones or []:
        name = _zone_group_name(zone)
        tried.add(name)
        years = years_from(name)
        if years:
            return years

    # Fall back to a listing only if the requested zones told us nothing —
    # enumerating the hierarchy is a round trip we can usually skip.
    for name in _member_names(root):
        if not name.startswith("utm") or name in tried:
            continue
        years = years_from(name)
        if years:
            return years
    return []


def _member_names(group: "zarr.Group") -> List[str]:
    """Sorted member names of a group, without the sidecar warnings.

    A Tessera store deliberately keeps non-Zarr objects at its root (the
    ingestion registry, the lock directory), and zarr warns about each one
    every time the hierarchy is enumerated. They are expected here.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Object at .* is not recognized")
        return sorted(group.keys())


def _zone_group_names(
    store: StoreLocation, zones: Optional[List[int]] = None
) -> List[str]:
    """Names of the store's UTM zone groups, optionally filtered."""
    import re

    root = store.open_group(mode="r")
    pattern = re.compile(r"^utm(\d{2})$")
    names = []
    for name in _member_names(root):
        m = pattern.match(name)
        if m and (zones is None or int(m.group(1)) in zones):
            names.append(name)
    return names


def extend_store(
    store_path: "str | Path | StoreLocation",
    years: List[int],
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    zones: Optional[List[int]] = None,
    consolidate: bool = True,
    force: bool = False,
    state_url: Optional[str] = None,
) -> int:
    """Append new years to an existing store's time axis.

    The time dimension is chunked one year per chunk, so growing it is a
    metadata-only edit: existing chunks keep their keys and are never
    rewritten, and the new slice reads back with the same sentinels a
    freshly initialised year has (embeddings at 0, scales at +inf). Fill it
    afterwards with ``zarr-fill --year <new>``.

    Years must extend the axis at the end. Inserting an earlier year would
    shift every existing chunk's time index — a full rewrite of the store —
    so it is refused rather than done silently.

    This is a single-writer operation: it rewrites array metadata for every
    zone, so no fill may be in flight. Returns the number of zone groups
    extended.
    """
    store = StoreLocation.resolve(store_path, storage_options, state_url)
    years = sorted(set(int(y) for y in years))
    if not years:
        raise ValueError("No years given to add")

    held = [Path(p).name for p in store.state.listdir(LOCK_DIR_NAME, on_denied=[])]
    if held and not force:
        raise RuntimeError(
            f"{len(held)} fill lock(s) present ({', '.join(sorted(held)[:4])}"
            f"{'...' if len(held) > 4 else ''}). Extending rewrites array "
            f"metadata for every zone, so wait for the sweep to finish. "
            f"Use --force if these are stale."
        )

    zone_names = _zone_group_names(store, zones)
    if not zone_names:
        raise ValueError(f"No UTM zone groups found in {store}")

    if console:
        console.print(f"Extending [bold]{store}[/bold] with years {years}")
        console.print(f"  {len(zone_names)} zone group(s)")

    extended = 0
    skipped = 0
    for name in zone_names:
        group = store.open_group(mode="r+", path=name, zarr_format=3)
        existing = [int(v) for v in group["time"][:]]

        missing = [y for y in years if y not in existing]
        if not missing:
            skipped += 1
            continue

        earliest_new = min(missing)
        if existing and earliest_new <= max(existing):
            raise ValueError(
                f"{name}: cannot add {earliest_new} to a time axis ending at "
                f"{max(existing)}. Years may only be appended — inserting an "
                f"earlier one would renumber every existing chunk."
            )

        old_t = len(existing)
        new_t = old_t + len(missing)

        # Every time-indexed array must grow together, or the group's axes
        # desynchronise. A zone from a store that predates the stretch
        # statistics lacks those arrays; extending it would leave them
        # permanently short, so refuse and point at the repair path.
        absent = [a for a in STRETCH_ARRAY_NAMES if a not in group]
        if absent:
            raise ValueError(
                f"{name}: missing stretch-statistics array(s) "
                f"{', '.join(absent)} — this store predates fill-time stretch "
                f"statistics. Run `zarr-fill --backfill-stretch-stats "
                f"--zones {name[3:]}` first, then re-run zarr-extend."
            )

        # Order matters only for crash-safety: grow the data arrays before
        # advertising the year on the time axis, so a run interrupted midway
        # never leaves a year readers can select but not read.
        for arr_name in ("embeddings", "scales", *STRETCH_ARRAY_NAMES):
            arr = group[arr_name]
            arr.resize((new_t,) + tuple(arr.shape[1:]))

        time_arr = group["time"]
        time_arr.resize((new_t,))
        time_arr[old_t:new_t] = np.array(missing, dtype=time_arr.dtype)

        extended += 1
        if console:
            console.print(
                f"  {name}: {old_t} -> {new_t} time steps "
                f"[dim](added {', '.join(str(y) for y in missing)})[/dim]"
            )

    if console and skipped:
        console.print(f"  {skipped} zone(s) already had every year")

    # Array metadata changed, so the consolidated root is now stale — unlike
    # a fill, this step genuinely requires re-consolidation.
    if consolidate and extended:
        consolidate_store(store, console=console)

    return extended


def load_landmask_tiles(
    dataset_version: str = "v1",
    landmasks_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> List[Tuple[float, float]]:
    """Land tile centres for a dataset version, without touching the manifest.

    Land coverage is all that is needed to say which shards can ever hold
    data, and the landmask registry is ~19 MB against the manifest's ~200 MB.
    Fetched from the public mirror and cached unless *landmasks_path* points
    at a local copy.
    """
    import pandas as pd

    from .registry import _parse_dataset_version, download_file_to_temp
    from .registry import landmasks_parquet_url

    version_path, _ = _parse_dataset_version(dataset_version)

    if landmasks_path and Path(landmasks_path).exists():
        path = Path(landmasks_path)
    else:
        if cache_dir is None:
            if os.name == "nt":
                base = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            cache_dir = base / "geotessera" / version_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            download_file_to_temp(
                landmasks_parquet_url(version_path),
                cache_path=cache_dir / "landmasks.parquet",
            )
        )

    df = pd.read_parquet(path, columns=["lon", "lat"])
    return list(zip(df["lon"].astype(float), df["lat"].astype(float)))


def _shards_from_tiles(
    tile_coords: List[Tuple[float, float]],
    grid: UnifiedZoneGrid,
    transformer_cache: Dict[int, Any],
) -> Dict[Tuple[int, int], int]:
    """Map tile centres to the shards they touch, with a tile count each."""
    per_shard: Dict[Tuple[int, int], int] = {}
    for lon, lat in tile_coords:
        ti = project_tile(lon, lat, transformer_cache=transformer_cache)
        for coord in shard_coords_for_tiles([ti], grid):
            per_shard[coord] = per_shard.get(coord, 0) + 1
    return per_shard


def scan_store(
    registry: Optional["Registry"],
    store_path: "str | Path | StoreLocation",
    years: Optional[List[int]] = None,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    source: Optional[TileSource] = None,
    state_url: Optional[str] = None,
    output: Optional[str] = None,
    dataset_version: str = "v1",
    landmasks_path: Optional[str] = None,
) -> "pandas.DataFrame":
    """Inventory a store's shards, without writing data.

    Answers "how much is left to fill" from the store itself rather than from
    bookkeeping, by listing the shard objects that exist and comparing them
    with the shards that could hold data. Each shard is classified:

    ``written``
        The shard object is in the store.
    ``missing``
        Land falls here but no shard object exists — the work still to do.
    ``empty``
        No land falls in this shard, so it is ocean or outside coverage and
        will never be filled. Reported separately so the percentages are
        over land, not over the zone's bounding box.

    The land denominator comes from the landmask registry (~19 MB, fetched
    and cached automatically), so no tile mirror or manifest is needed —
    scanning a remote store on its own is enough. Passing *registry* instead
    narrows it to the tiles that version's manifest lists **for each year**,
    which is exact where a year's embedding coverage is smaller than the land
    area, at the cost of loading the much larger manifest.

    Returns a DataFrame with one row per (zone, year, shard), also written to
    *output* as parquet when given.
    """
    import pandas as pd

    store = StoreLocation.resolve(store_path, storage_options, state_url)

    all_years = _store_years(store, zones)
    if not all_years:
        raise ValueError("Store has no years (checked zone time coords)")
    scan_years = [y for y in (years or all_years) if y in all_years]

    if console:
        console.print(f"Scanning [bold]{store}[/bold]")
        console.print(f"  Years: {scan_years}")

    transformer_cache: Dict[int, Any] = {}
    land_by_zone: Dict[int, List[Tuple[float, float]]] = {}
    if registry is None:
        if console:
            console.print("  Land coverage from the landmask registry")
        for lon, lat in load_landmask_tiles(dataset_version, landmasks_path):
            zone_num = tile_zone(lon)
            if zones is None or zone_num in zones:
                land_by_zone.setdefault(zone_num, []).append((lon, lat))

    rows: List[Dict[str, Any]] = []

    for scan_year in scan_years:
        if registry is not None:
            year_tiles = gather_tile_infos(
                registry, scan_year, zones=zones, console=None, source=source
            )
            zone_items: List[Tuple[int, Any]] = sorted(year_tiles.items())
        else:
            zone_items = sorted(land_by_zone.items())

        for zone_num, coverage in zone_items:
            zone_group = _zone_group_name(zone_num)
            try:
                zone_store = store.open_group(mode="r", path=zone_group)
                zone_years = [int(v) for v in zone_store["time"][:]]
            except Exception:
                continue
            if scan_year not in zone_years:
                continue
            time_index = zone_years.index(scan_year)

            attrs = dict(zone_store.attrs)
            transform = attrs["spatial:transform"]
            shape = attrs["spatial:shape"]
            grid = UnifiedZoneGrid(
                zone=zone_num,
                years=all_years,
                canonical_epsg=int(attrs["proj:code"].split(":")[1]),
                origin_x=transform[2],
                origin_y=transform[5],
                width_px=shape[1],
                height_px=shape[0],
            )

            # Tiles per shard, so the index records how much work each holds.
            if registry is not None:
                per_shard = {}
                for ti in coverage:
                    for coord in shard_coords_for_tiles([ti], grid):
                        per_shard[coord] = per_shard.get(coord, 0) + 1
            else:
                per_shard = _shards_from_tiles(coverage, grid, transformer_cache)

            expected = set(per_shard)
            present = _existing_shards(
                store, zone_group, time_index, expected, console=console
            )

            n_rows_grid = math.ceil(grid.height_px / SHARD_SIZE)
            n_cols_grid = math.ceil(grid.width_px / SHARD_SIZE)
            for sr in range(n_rows_grid):
                for sc in range(n_cols_grid):
                    coord = (sr, sc)
                    n_tiles = per_shard.get(coord, 0)
                    if n_tiles == 0:
                        status = "empty"
                    elif coord in present:
                        status = "written"
                    else:
                        status = "missing"
                    rows.append(
                        {
                            "zone": zone_num,
                            "year": scan_year,
                            "shard_row": sr,
                            "shard_col": sc,
                            "n_tiles": n_tiles,
                            "status": status,
                        }
                    )

            if console:
                n_exp = len(expected)
                n_have = len(present & expected)
                pct = 100.0 * (n_exp - n_have) / n_exp if n_exp else 0.0
                console.print(
                    f"  utm{zone_num:02d} {scan_year}: "
                    f"{n_have}/{n_exp} shards written, {pct:.1f}% to fill"
                )

    df = pd.DataFrame(
        rows,
        columns=["zone", "year", "shard_row", "shard_col", "n_tiles", "status"],
    )

    if output:
        from . import remote

        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        remote.write_bytes(output, buf.getvalue(), storage_options)
        if console:
            console.print(f"  Wrote shard index to [bold]{output}[/bold]")

    return df


def summarise_scan(df: "pandas.DataFrame", console: "rich.console.Console") -> None:
    """Print per-zone/year and per-year fill summaries from a scan."""
    from rich.table import Table

    if df.empty:
        console.print("[yellow]Nothing scanned.[/yellow]")
        return

    def _pct(sub) -> float:
        expected = int((sub["status"] != "empty").sum())
        missing = int((sub["status"] == "missing").sum())
        return 100.0 * missing / expected if expected else 0.0

    detail = Table(title="Fill needed by zone and year")
    for col in ("Zone", "Year", "Land shards", "Written", "Missing", "% to fill"):
        detail.add_column(col, justify="right" if col != "Zone" else "left")

    for (zone, year), sub in df.groupby(["zone", "year"], sort=True):
        expected = int((sub["status"] != "empty").sum())
        written = int((sub["status"] == "written").sum())
        missing = int((sub["status"] == "missing").sum())
        if expected == 0:
            continue
        detail.add_row(
            f"utm{int(zone):02d}",
            str(int(year)),
            f"{expected:,}",
            f"{written:,}",
            f"{missing:,}",
            f"{_pct(sub):.1f}%",
        )
    console.print(detail)

    by_year = Table(title="Fill needed by year (all scanned zones)")
    for col in ("Year", "Land shards", "Written", "Missing", "% to fill"):
        by_year.add_column(col, justify="right" if col != "Year" else "left")
    for year, sub in df.groupby("year", sort=True):
        expected = int((sub["status"] != "empty").sum())
        if expected == 0:
            continue
        by_year.add_row(
            str(int(year)),
            f"{expected:,}",
            f"{int((sub['status'] == 'written').sum()):,}",
            f"{int((sub['status'] == 'missing').sum()):,}",
            f"{_pct(sub):.1f}%",
        )
    console.print(by_year)

    total_expected = int((df["status"] != "empty").sum())
    total_missing = int((df["status"] == "missing").sum())
    total_empty = int((df["status"] == "empty").sum())
    overall = 100.0 * total_missing / total_expected if total_expected else 0.0
    console.print(
        f"Overall: {total_missing:,}/{total_expected:,} land shards to fill "
        f"({overall:.1f}%); {total_empty:,} shard(s) are water/no-coverage."
    )


def fill_store(
    registry: "Registry",
    store_path: "str | Path | StoreLocation",
    year: Optional[int] = None,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
    workers: Optional[int] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    source: Optional[TileSource] = None,
    consolidate: Optional[bool] = None,
    force_lock: bool = False,
    state_url: Optional[str] = None,
    skip_existing_shards: bool = True,
    spill_dir: Optional[str] = None,
    collect_stretch_stats: bool = True,
) -> int:
    """Incrementally fill a store with tile data.

    Reads the tile registry to skip already-written tiles.
    Returns the number of shards written.

    Args:
        store_path: Local path or fsspec URL of an initialised store.
        zones: Restrict the fill to these UTM zones.  One process per zone
            can run concurrently against the same store.
        storage_options: fsspec options for the store (endpoint, credentials).
        source: Where the tile inputs live; defaults to the registry's local
            mirror.
        consolidate: Rewrite the root consolidated metadata when done.
            Defaults to True for a whole-store fill and False when ``zones``
            is set, because the root object is the one thing parallel zone
            jobs share — run ``zarr-consolidate`` once after the sweep.
        force_lock: Take over a (zone, year) lock held by another process.
        skip_existing_shards: Scan for shards already in the store and skip
            them (the default). A shard is always written from every tile
            covering it, so its presence means it is complete, and the
            objects outlive the ingestion registry — which makes this both
            the cheapest resume and the only one that survives a kill -9.
            Set False to rebuild them, which is needed only when the tile
            inventory has grown: a tile added to the manifest afterwards
            falls inside an existing shard and would otherwise be skipped
            rather than merged in.
    """
    store = StoreLocation.resolve(store_path, storage_options, state_url)
    if workers is None:
        workers = DEFAULT_WORKERS
    if consolidate is None:
        consolidate = zones is None

    all_years = _store_years(store, zones)
    if not all_years:
        raise ValueError("Store has no years (checked zone time coords)")

    fill_years = [year] if year is not None else all_years

    _warn_worker_memory(workers, console, spilled=bool(spill_dir))

    if console:
        console.print(f"Filling store at [bold]{store}[/bold]")
        console.print(f"  Years to fill: {fill_years}")
        if source is not None and source.is_remote:
            console.print(f"  Streaming tiles from [bold]{source.embeddings_root}[/bold]")

    total_shards_written = 0
    total_shards_failed = 0

    # The merged registry spans the whole store, so fetch it once rather
    # than per zone and year.
    merged_registry = load_merged_registry(store)

    for fill_year in fill_years:
        if fill_year not in all_years:
            if console:
                console.print(
                    f"  [yellow]Year {fill_year} not in store, skipping[/yellow]"
                )
            continue

        # Gather tiles for this year
        year_tiles = gather_tile_infos(
            registry,
            fill_year,
            zones=zones,
            console=console,
            source=source,
        )

        for zone_num, tile_infos in sorted(year_tiles.items()):
            zone_group = _zone_group_name(zone_num)

            # Open the group rather than probing for the prefix: reading
            # utm{n}/zarr.json needs only GetObject, whereas an existence
            # check on a prefix needs list permission the writer may lack.
            try:
                zone_store = store.open_group(mode="r", path=zone_group)
            except Exception:
                if console:
                    console.print(
                        f"  [yellow]Zone {zone_num} not initialised, skipping[/yellow]"
                    )
                continue

            # Check which tiles are already written
            written = _get_written_tiles(
                store, fill_year, zone_num, merged=merged_registry
            )
            remaining = [ti for ti in tile_infos if (ti.lon, ti.lat) not in written]

            if not remaining:
                if console:
                    console.print(
                        f"  Zone {zone_num} year {fill_year}: "
                        f"all {len(tile_infos)} tiles already written"
                    )
                continue

            if console:
                console.print(
                    f"  Zone {zone_num} year {fill_year}: "
                    f"{len(remaining)}/{len(tile_infos)} tiles to write"
                )

            # Resolve the time index against *this* zone's own axis. An
            # interrupted zarr-extend can leave zones with different lengths,
            # and a store-wide index would then address the wrong year.
            zone_years = [int(v) for v in zone_store["time"][:]]
            if fill_year not in zone_years:
                if console:
                    console.print(
                        f"    [yellow]Zone {zone_num} has no {fill_year} on its "
                        f"time axis ({zone_years}); run zarr-extend first. "
                        f"Skipping.[/yellow]"
                    )
                continue
            time_index = zone_years.index(fill_year)

            zone_attrs = dict(zone_store.attrs)
            transform = zone_attrs["spatial:transform"]
            shape = zone_attrs["spatial:shape"]

            grid = UnifiedZoneGrid(
                zone=zone_num,
                years=all_years,
                canonical_epsg=int(zone_attrs["proj:code"].split(":")[1]),
                origin_x=transform[2],
                origin_y=transform[5],
                width_px=shape[1],
                height_px=shape[0],
            )

            # A shard write replaces the whole shard, so every shard we touch
            # must be rebuilt from all of its tiles — including ones an
            # earlier run already wrote, which would otherwise be zeroed.
            touched = shard_coords_for_tiles(remaining, grid)
            shard_specs = build_shard_index(
                tile_infos, grid, time_index, restrict_to=touched
            )

            # The shard objects in the store are the ground truth for what
            # landed — unlike the ingestion registry they survive a kill -9,
            # so a crashed run can be resumed by scanning for them.
            skipped_specs: List[ShardSpec] = []
            if skip_existing_shards:
                present = _existing_shards(
                    store,
                    zone_group,
                    time_index,
                    {(s.sr, s.sc) for s in shard_specs},
                    console=console,
                )
                if present:
                    skipped_specs = [
                        s for s in shard_specs if (s.sr, s.sc) in present
                    ]
                    shard_specs = [
                        s for s in shard_specs if (s.sr, s.sc) not in present
                    ]

            if console:
                # Spell the arithmetic out. The count of shards to write is
                # otherwise hard to reconcile with zarr-scan, which counts
                # every land shard, whereas a fill only considers those
                # covering tiles the registry has not already recorded.
                n_land = len(shard_coords_for_tiles(tile_infos, grid))
                n_recorded = n_land - len(touched)
                console.print(
                    f"    Shards: {n_land:,} land, "
                    f"{n_recorded:,} recorded done, "
                    f"{len(skipped_specs):,} found in store, "
                    f"[bold]{len(shard_specs):,} to write[/bold] "
                    f"({workers} workers)"
                )

            # Stretch statistics: collect only when the zone has the arrays
            # (stores initialised before the feature lack them; repair with
            # --backfill-stretch-stats). Per-shard cap sized so the expected
            # candidate pool is a few times K without ballooning the result
            # queue.
            sample_cap = 0
            if collect_stretch_stats and "stretch_sample" in zone_store:
                k_slots = zone_store["stretch_sample"].shape[1]
                sample_cap = _shard_sample_cap(k_slots, len(shard_specs))
            elif collect_stretch_stats and console:
                console.print(
                    f"    [yellow]Zone {zone_num} has no stretch-statistics "
                    f"arrays (store predates them); skipping collection. "
                    f"Backfill later with --backfill-stretch-stats.[/yellow]"
                )

            _acquire_zone_lock(store, zone_num, fill_year, force=force_lock)
            try:
                written_count, failed, shard_stats = _write_shards(
                    store=store,
                    zone_group=zone_group,
                    shard_specs=shard_specs,
                    workers=workers,
                    source_options=source.storage_options if source else None,
                    label=f"    Zone {zone_num} y{fill_year}",
                    console=console,
                    spill_dir=spill_dir,
                    sample_cap=sample_cap,
                )

                total_shards_written += written_count
                total_shards_failed += len(failed)

                if shard_stats:
                    zone_rw = store.open_group(mode="r+", path=zone_group)
                    update_zone_stretch_stats(
                        zone_rw,
                        time_index,
                        n=sum(st["n"] for st in shard_stats),
                        s=sum(st["sum"] for st in shard_stats),
                        m=sum(st["prod"] for st in shard_stats),
                        sample_candidates=[
                            (st["sample_emb"], st["sample_scales"], st["sample_weight"])
                            for st in shard_stats
                        ],
                    )
                    if console:
                        console.print(
                            f"    [dim]Stretch stats: "
                            f"{sum(st['n'] for st in shard_stats):,} pixels "
                            f"folded in[/dim]"
                        )

                if console:
                    console.print(
                        f"    [green]{written_count}/{len(shard_specs)} "
                        f"shards written[/green]"
                    )
                    if failed:
                        console.print(
                            f"    [red]{len(failed)} shard(s) failed[/red]"
                        )

                # Record tiles whose shards all landed — counting the ones we
                # skipped as landed, since they are already in the store — so
                # a retry picks up exactly the work still outstanding.
                done = {(s.sr, s.sc) for s in skipped_specs} | (
                    {(s.sr, s.sc) for s in shard_specs} - failed
                )
                recorded = [
                    ti
                    for ti in remaining
                    if shard_coords_for_tiles([ti], grid) <= done
                ]
                _record_written_tiles(store, recorded, fill_year, zone_num)
            finally:
                _release_zone_lock(store, zone_num, fill_year)

    # A failed shard leaves its tiles unrecorded, so re-running finishes the
    # job. Surface it as an error rather than a quiet partial success — a
    # sweep orchestrator has no other way to tell the zone needs a retry.
    if total_shards_failed:
        raise RuntimeError(
            f"{total_shards_failed} shard(s) failed "
            f"({total_shards_written} written). Re-run the same command to "
            f"retry only the unfinished tiles."
        )

    # Re-consolidate metadata after filling. Skipped for a zone-restricted
    # fill: the root object is shared with any sibling zone jobs.
    if total_shards_written > 0:
        if consolidate:
            consolidate_store(store, console=console)
        elif console:
            console.print(
                "  [dim]Skipped consolidation (zone-restricted fill). "
                "Run `geotessera-registry zarr-consolidate` once the sweep "
                "finishes.[/dim]"
            )

    return total_shards_written


def _write_shards(
    store: StoreLocation,
    zone_group: str,
    shard_specs: List[ShardSpec],
    workers: int,
    source_options: Optional[Dict[str, Any]],
    label: str,
    console: Optional["rich.console.Console"],
    spill_dir: Optional[str] = None,
    sample_cap: int = 0,
) -> Tuple[int, set, List[Dict[str, Any]]]:
    """Run the shard writes through a process pool.

    Returns (shards written, set of (sr, sc) that failed, per-shard stretch
    statistics — empty when collection is off or no shard had valid pixels).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    written_count = 0
    failed: set = set()
    stats_results: List[Dict[str, Any]] = []
    initargs = (
        store.url,
        zone_group,
        store.storage_options,
        source_options,
        spill_dir,
        sample_cap,
    )

    # "spawn", not the Linux default of "fork": a forked worker inherits the
    # parent's fsspec event-loop object without the thread that runs it, and
    # deadlocks the first time it talks to the store.
    mp_context = multiprocessing.get_context("spawn")

    def _drain(pool, advance=None):
        nonlocal written_count
        futures = {
            pool.submit(_write_one_shard_worker, spec): spec for spec in shard_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
                if result:
                    written_count += 1
                if isinstance(result, dict):
                    stats_results.append(result)
            except Exception as e:
                logger.warning(f"Shard ({spec.sr},{spec.sc}) failed: {e}")
                failed.add((spec.sr, spec.sc))
            if advance is not None:
                advance()

    # Not a `with` block: on Ctrl-C the context manager's shutdown(wait=True)
    # blocks until every in-flight shard finishes, which with a pool of
    # multi-gigabyte workers looks like a hang and invites a second Ctrl-C
    # (and a second traceback). Cancel what has not started and leave.
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_shard_worker,
        initargs=initargs,
        mp_context=mp_context,
    )
    try:
        if console:
            from rich.progress import (
                Progress,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
                SpinnerColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(label, total=len(shard_specs))
                _drain(pool, advance=lambda: progress.advance(task))
        else:
            _drain(pool)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    return written_count, failed, stats_results


def consolidate_store(
    store_path: "str | Path | StoreLocation",
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    merge_registry: bool = True,
    state_url: Optional[str] = None,
) -> int:
    """Re-consolidate a store's root metadata after in-place changes.

    ``fill_store`` skips consolidation for zone-restricted fills so parallel
    zone jobs never contend for the root ``zarr.json``, and a metadata-only
    change to an existing store (e.g. rewriting an array with a different
    compressor) leaves the consolidated metadata stale too.  HTTP readers
    cannot list a store and trust consolidated metadata exclusively, so a
    stale root breaks them.  This is the single-writer step that fixes both.

    Accepts a local store path or a remote fsspec URL such as
    ``s3://bucket/store.zarr``.  Remote URLs need the matching fsspec
    backend installed (``s3fs`` for S3) and write credentials for the
    final root ``zarr.json`` upload.

    Args:
        merge_registry: Also fold the per-zone ingestion files under
            ``_registry/`` into the root ``_registry.parquet``.

    Returns the number of consolidated nodes.
    """
    import warnings
    import zarr

    store = StoreLocation.resolve(store_path, storage_options, state_url)
    if not store.is_remote and not Path(store.url).exists():
        raise FileNotFoundError(f"store not found: {store.url}")

    if console:
        console.print(f"Consolidating metadata at [bold]{store}[/bold]")

    if merge_registry:
        merge_tile_registry(store, console=console)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        # Tessera stores carry non-zarr marker objects (tile registry,
        # zone-completion flags) that the consolidation walk would warn about.
        warnings.filterwarnings("ignore", message="Object at .* is not recognized")
        group = zarr.consolidate_metadata(store.as_zarr_store())

    return len(group.metadata.consolidated_metadata.flattened_metadata)


# ---------------------------------------------------------------------------
# RGB preview generation (NCHW layout)
# ---------------------------------------------------------------------------

RGB_PREVIEW_BANDS = (0, 1, 2)


def _compute_rgb_chunk(
    emb_bhw: np.ndarray,
    scales_hw: np.ndarray,
    band_indices: tuple,
    stretch_min: List[float],
    stretch_max: List[float],
    cdf: Optional[List[List[float]]] = None,
    gamma: float = 1.0,
    saturation: float = 1.0,
    pca_components: Optional[List[List[float]]] = None,
    pca_mean: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute RGBA preview from NCHW-layout embedding + scales.

    Args:
        emb_bhw: int8 (B, H, W). When ``pca_components`` is provided this is
            the full 128-band slice; otherwise it's the bands referenced by
            ``band_indices`` (typically 3).
        scales_hw: float32 (H, W)
        cdf: Optional per-channel CDF breakpoints — when present, pixels are
            mapped through ``searchsorted`` to uniformly cover [0, 255]
            instead of the linear ``(x - lo) / (hi - lo)`` stretch. In PCA
            mode the CDF is in *PC space*, not band space.
        gamma: Power-law applied to the [0, 1] channel value before
            quantisation to uint8. ``< 1.0`` brightens midtones.
        saturation: Final chroma multiplier (default 1.0). Each pixel is
            decomposed into ``grey + chroma`` where ``grey`` is its Rec.601
            luminance; ``chroma`` is then scaled by this factor before
            re-quantising. ``1.5`` is a noticeable pop, ``2.0`` is vivid,
            beyond ~3 most pixels start clipping.
        pca_components: Optional ``(K, n_bands)`` projection matrix learned
            in :func:`compute_global_stretch` with ``mode='pca'``. When
            present, all ``n_bands`` channels of ``emb_bhw`` are projected
            into ``K`` orthogonal directions before stretching — eliminates
            the channel-correlation-driven "washed out" look.
        pca_mean: Companion ``(n_bands,)`` centring vector for the PCA.

    Returns:
        uint8 (4, H, W) — RGBA in channels-first layout.
    """
    h, w = scales_hw.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    valid = np.isfinite(scales_hw)
    scales_safe = np.where(valid, scales_hw, 0.0)

    # Build the per-channel float arrays. Two paths:
    #   - PCA: dequantise all bands, centre, project (K, n_bands) @ (n_bands, H*W)
    #   - bands: dequantise per RGB band as before
    if pca_components is not None and pca_mean is not None:
        components = np.asarray(pca_components, dtype=np.float32)
        mean = np.asarray(pca_mean, dtype=np.float32)
        n_bands_in = emb_bhw.shape[0]
        if components.shape[1] != n_bands_in:
            raise ValueError(
                f"PCA components have {components.shape[1]} bands but "
                f"input has {n_bands_in}"
            )
        # Dequantise the whole stack at once. Memory: n_bands × h × w × 4 B.
        dequant_full = emb_bhw.astype(np.float32) * scales_safe[np.newaxis, ...]
        flat = dequant_full.reshape(n_bands_in, -1)  # (n_bands, H*W)
        flat -= mean[:, np.newaxis]
        pcs_flat = components @ flat  # (K, H*W)
        pcs = pcs_flat.reshape(components.shape[0], h, w)
        per_channel_input = list(pcs)
        n_iter = components.shape[0]
    else:
        per_channel_input = [
            emb_bhw[i].astype(np.float32) * scales_safe for i in band_indices
        ]
        n_iter = len(band_indices)

    # Keep float channels until after the saturation step to avoid two
    # quantise-then-dequantise rounds.
    float_channels: List[np.ndarray] = []
    for i in range(n_iter):
        dequant = per_channel_input[i]
        if cdf is not None:
            breaks = np.asarray(cdf[i], dtype=np.float32)
            n_break = len(breaks)
            # searchsorted → bin index 0..n_break; renormalise to [0, 1].
            idx = np.searchsorted(breaks, dequant.ravel()).astype(np.float32)
            normalised = (
                (idx / max(n_break - 1, 1)).clip(0.0, 1.0).reshape(dequant.shape)
            )
        else:
            lo, hi = stretch_min[i], stretch_max[i]
            normalised = np.clip((dequant - lo) / max(hi - lo, 1e-10), 0.0, 1.0)
        if gamma != 1.0:
            normalised = np.power(normalised, gamma, dtype=np.float32)
        float_channels.append(normalised)

    if saturation != 1.0 and len(float_channels) >= 3:
        # Rec.601 luma; chroma is (channel − luma). Scaling chroma and
        # re-adding to the original luma preserves brightness while
        # spreading colours away from the grey diagonal.
        r, g, b = float_channels[0], float_channels[1], float_channels[2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        float_channels[0] = np.clip(luma + (r - luma) * saturation, 0.0, 1.0)
        float_channels[1] = np.clip(luma + (g - luma) * saturation, 0.0, 1.0)
        float_channels[2] = np.clip(luma + (b - luma) * saturation, 0.0, 1.0)

    for i, normalised in enumerate(float_channels):
        rgba[i] = (normalised * 255).astype(np.uint8)

    rgba[:3, ~valid] = 0
    rgba[3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _sample_chunk_stats(
    emb_arr,
    scales_arr,
    time_index: int,
    ci: int,
    cj: int,
    shard_size: int,
    spatial_shape: Tuple[int, int],
    band_indices: tuple = RGB_PREVIEW_BANDS,
    max_per_chunk: int = 10_000,
) -> Optional[np.ndarray]:
    """Sample dequantised values from one shard for stretch estimation.

    Reads from NCHW layout: emb_arr[t, bands, r0:r1, c0:c1].
    """
    H, W = spatial_shape
    r0, r1 = ci * shard_size, min(ci * shard_size + shard_size, H)
    c0, c1 = cj * shard_size, min(cj * shard_size + shard_size, W)

    scales_chunk = np.asarray(scales_arr[time_index, r0:r1, c0:c1])
    valid = np.isfinite(scales_chunk)
    if not np.any(valid):
        return None

    # Read only the RGB bands
    band_list = list(band_indices)
    emb_chunk = np.asarray(
        emb_arr[time_index, band_list[0] : band_list[-1] + 1, r0:r1, c0:c1]
    )  # (n_rgb_bands, h, w)

    # Dequantise only valid pixels (avoid inf/nan multiply warnings)
    scales_safe = np.where(valid, scales_chunk, 0.0)
    vals_all = emb_chunk.astype(np.float32) * scales_safe[np.newaxis, :, :]
    # Reshape to (n_pixels, n_bands) and keep only valid
    n_bands = vals_all.shape[0]
    vals_flat = vals_all.reshape(n_bands, -1).T  # (n_pixels, n_bands)
    valid_flat = valid.ravel()
    vals = vals_flat[valid_flat]

    if vals.shape[0] > max_per_chunk:
        rng = np.random.default_rng(ci * 10007 + cj)
        idx = rng.choice(vals.shape[0], max_per_chunk, replace=False)
        vals = vals[idx]

    return vals


def compute_stretch(
    store: "zarr.Group",
    time_index: int,
    p_low: float = 2,
    p_high: float = 98,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
    sample_fraction: float = 0.1,
) -> dict:
    """Compute percentile stretch for RGB bands at one time step."""

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    _, _, H, W = emb_arr.shape

    n_rows = math.ceil(H / SHARD_SIZE)
    n_cols = math.ceil(W / SHARD_SIZE)
    all_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    n_sample = max(1, int(len(all_indices) * sample_fraction))
    rng = np.random.default_rng(42)
    sample_indices = [
        all_indices[i] for i in rng.choice(len(all_indices), n_sample, replace=False)
    ]

    results = _run_parallel(
        lambda idx: _sample_chunk_stats(
            emb_arr,
            scales_arr,
            time_index,
            idx[0],
            idx[1],
            SHARD_SIZE,
            (H, W),
        ),
        sample_indices,
        workers,
        console,
        label=f"Sampling stretch ({n_sample}/{len(all_indices)} shards)",
    )

    samples = [r for _, r in results if r is not None]
    if not samples:
        return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}

    all_rgb = np.concatenate(samples, axis=0)
    stretch_min = [float(np.percentile(all_rgb[:, i], p_low)) for i in range(3)]
    stretch_max = [float(np.percentile(all_rgb[:, i], p_high)) for i in range(3)]

    for i in range(3):
        if stretch_max[i] <= stretch_min[i]:
            stretch_max[i] = stretch_min[i] + 1.0

    return {"min": stretch_min, "max": stretch_max}


# ---------------------------------------------------------------------------
# Global cross-zone stretch (sparsity-aware sampler)
# ---------------------------------------------------------------------------
# Independently of `compute_stretch` (which works per-zone), this gives us
# ONE set of (min, max) percentile bounds derived from a random sample of
# valid pixels drawn from every UTM zone. Storing the result on the store's
# root attrs lets `build_global_preview` reuse it across zones and produce
# a seamless mosaic instead of one stretch per zone.

_GLOBAL_STRETCH_ATTR = "geoemb:stretch"


def _parse_pca_perm(pca_rgb_order: str, pca_components: int) -> List[int]:
    """Validate a PC→RGB permutation like '213' and return 0-based indices."""
    if len(pca_rgb_order) != pca_components or set(pca_rgb_order) != {
        str(i + 1) for i in range(pca_components)
    }:
        raise ValueError(
            f"pca_rgb_order must be a permutation of the digits "
            f"1..{pca_components} (e.g. '123' or '213'), got {pca_rgb_order!r}"
        )
    return [int(c) - 1 for c in pca_rgb_order]


def compute_stretch_from_stats(
    store_path: "str | Path | StoreLocation",
    year: int,
    zones: Optional[List[int]] = None,
    p_low: float = 2.0,
    p_high: float = 98.0,
    equalise: bool = True,
    equalise_breakpoints: int = 257,
    mode: str = "pca",
    pca_components: int = 3,
    pca_rgb_order: str = "123",
    drift_threshold: float = 0.25,
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Derive the global stretch from the per-zone ``stretch_*`` arrays.

    The fast path of ``zarr-stretch`` (docs/specs/zarr-stretch-stats.md):
    reads a few MiB of per-zone summaries instead of terabytes of shards.
    The PCA comes from the summed sufficient statistics and is exact — every
    valid pixel in the store contributes. Quantiles come from the pooled
    weighted samples, projected into PC space.

    Writes the result to the root ``geoemb:stretch.{year}`` attribute with
    the same keys the legacy shard-sampling path produces, so
    ``build_global_preview`` and other readers are unaffected. Works against
    local and remote stores alike.
    """
    if mode not in ("bands", "pca"):
        raise ValueError(f"mode must be 'bands' or 'pca', got {mode!r}")
    pca_perm = _parse_pca_perm(pca_rgb_order, pca_components) if mode == "pca" else None

    store = StoreLocation.resolve(store_path, storage_options)
    zone_names = _zone_group_names(store, zones)
    if not zone_names:
        raise ValueError(f"No UTM zone groups found in {store}")

    n_total = 0
    s_total = np.zeros(N_BANDS, dtype=np.float64)
    m_total = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
    sample_parts: List[Tuple[np.ndarray, np.ndarray, float]] = []
    zones_used: List[str] = []
    zones_missing: List[str] = []

    for name in zone_names:
        group = store.open_group(mode="r", path=name)
        if "stretch_stats_count" not in group:
            zones_missing.append(name)
            continue
        try:
            zone_years = [int(v) for v in group["time"][:]]
            t = zone_years.index(year)
        except (ValueError, KeyError):
            continue

        n_z = int(group["stretch_stats_count"][t])
        if n_z == 0:
            continue
        n_total += n_z
        s_total += np.asarray(group["stretch_stats_sum"][t])
        m_total += np.asarray(group["stretch_stats_prod"][t])

        k_z = int(group["stretch_sample_count"][t])
        if k_z > 0:
            sample_parts.append(
                (
                    np.asarray(group["stretch_sample"][t, :k_z]),
                    np.asarray(group["stretch_sample_scales"][t, :k_z]),
                    n_z / k_z,
                )
            )
        zones_used.append(name)

    if zones_missing:
        raise ValueError(
            f"{len(zones_missing)} zone(s) have no stretch-statistics arrays "
            f"({', '.join(zones_missing[:5])}{'...' if len(zones_missing) > 5 else ''}). "
            f"Run `zarr-fill --backfill-stretch-stats` for them, or use "
            f"--from-shards for the legacy path."
        )
    if n_total == 0 or not sample_parts:
        raise RuntimeError(
            f"No stretch statistics recorded for year {year} — have the "
            f"zone fills for this year run with stats collection on?"
        )

    if console:
        console.print(
            f"Stretch from stored statistics: {len(zones_used)} zone(s), "
            f"{n_total:,} pixels in the exact covariance, "
            f"{sum(len(p[0]) for p in sample_parts):,} sampled pixels for "
            f"quantiles"
        )

    # Exact global mean/covariance from the summed sufficient statistics.
    mu = s_total / n_total
    cov = m_total / n_total - np.outer(mu, mu)

    # Pooled weighted sample, dequantised.
    emb = np.concatenate([p[0] for p in sample_parts], axis=0)
    scales = np.concatenate([p[1] for p in sample_parts], axis=0)
    weights = np.concatenate(
        [np.full(len(p[0]), p[2], dtype=np.float64) for p in sample_parts]
    )
    x = emb.astype(np.float32) * scales[:, None]  # (n, 128)

    pca_proj_components = None
    pca_proj_mean = None
    pca_evr = None
    if mode == "pca":
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1][:pca_components]
        components = eigvecs[:, order].T  # (k, 128), eigenvalue-descending
        evr = eigvals[order] / max(eigvals.sum(), 1e-30)

        # Drift check: re-estimate the covariance from the (independent)
        # stored sample and compare. Rewritten shards double-count into the
        # sums but replace sample slots, so divergence here flags stale
        # statistics. The metric is relative Frobenius distance between the
        # two covariances — comparing eigenvectors instead would false-alarm
        # whenever eigenvalues are close, where the vectors are arbitrary.
        cov_s = np.cov(x.T, aweights=weights)
        drift = float(
            np.linalg.norm(cov - cov_s) / max(np.linalg.norm(cov), 1e-30)
        )
        # The sample covariance itself carries ~sqrt(d/n_eff) relative error,
        # so the alarm floor scales with the effective sample size — a small
        # sample must not read as drift.
        n_eff = float(weights.sum() ** 2 / (weights**2).sum())
        noise = math.sqrt(N_BANDS / max(n_eff, 1.0))
        limit = max(drift_threshold, 3.0 * noise)
        if console:
            colour = "green" if drift <= limit else "red"
            console.print(
                f"  Drift check: |cov_stats − cov_sample|/|cov_stats| = "
                f"[{colour}]{drift:.4f}[/{colour}] "
                f"(limit {limit:.2f}; sample noise floor {noise:.2f})"
            )
        if drift > limit:
            logger.warning(
                f"Stretch statistics drift: {drift:.4f} > {limit:.2f}. "
                f"The additive sums likely double-counted rewritten shards; "
                f"rebuild with `zarr-fill --backfill-stretch-stats`, or "
                f"cross-check with `zarr-stretch --from-shards`."
            )

        # Bake the PC→RGB permutation into the stored matrix, as the legacy
        # path does, so the render path needs no extra swapping.
        components = components[pca_perm]
        evr = evr[pca_perm]

        channels = (x - mu.astype(np.float32)) @ components.T.astype(np.float32)
        pca_proj_components = [[float(v) for v in row] for row in components]
        pca_proj_mean = [float(v) for v in mu]
        pca_evr = [float(v) for v in evr]
        band_indices: Tuple[int, ...] = tuple(range(N_BANDS))
    else:
        band_indices = RGB_PREVIEW_BANDS
        channels = x[:, list(band_indices)]

    n_ch = channels.shape[1]
    stretch_min = [
        float(weighted_percentile(channels[:, i], weights, np.array([p_low]))[0])
        for i in range(n_ch)
    ]
    stretch_max = [
        float(weighted_percentile(channels[:, i], weights, np.array([p_high]))[0])
        for i in range(n_ch)
    ]
    for i in range(n_ch):
        if stretch_max[i] <= stretch_min[i]:
            stretch_max[i] = stretch_min[i] + 1.0

    cdf_breaks = None
    if equalise:
        n_break = max(64, int(equalise_breakpoints))
        qs = np.linspace(0.0, 100.0, n_break)
        cdf_breaks = []
        for i in range(n_ch):
            bks = weighted_percentile(channels[:, i], weights, qs)
            for j in range(1, len(bks)):
                if bks[j] <= bks[j - 1]:
                    bks[j] = bks[j - 1] + 1e-9
            cdf_breaks.append([float(v) for v in bks])

    # Persist with the same key set as the legacy path so readers
    # (_load_global_stretch, build_global_preview) are unaffected.
    root_rw = store.open_group(mode="r+")
    stretch_map = dict(root_rw.attrs.get(_GLOBAL_STRETCH_ATTR, {}))
    method_prefix = "zone_stats_pca" if mode == "pca" else "zone_stats_percentile"
    entry: Dict[str, Any] = {
        "min": stretch_min,
        "max": stretch_max,
        "p_low": p_low,
        "p_high": p_high,
        "samples": int(channels.shape[0]),
        "stats_pixels": int(n_total),
        "zones_used": len(zones_used),
        "bands": list(band_indices),
        "method": f"{method_prefix}{'_equalised' if equalise else ''}",
        "mode": mode,
    }
    if cdf_breaks is not None:
        entry["cdf"] = cdf_breaks
    if pca_proj_components is not None:
        entry["pca_components"] = pca_proj_components
        entry["pca_mean"] = pca_proj_mean
        entry["pca_explained_variance_ratio"] = pca_evr
    stretch_map[str(year)] = entry
    root_rw.attrs[_GLOBAL_STRETCH_ATTR] = stretch_map

    if console:
        console.print(
            f"[green]Saved to {_GLOBAL_STRETCH_ATTR}.{year} on the store "
            f"root.[/green] Run zarr-consolidate so consolidated-metadata "
            f"readers see it."
        )

    return {"min": stretch_min, "max": stretch_max, "samples": int(channels.shape[0])}


def backfill_stretch_stats(
    store_path: "str | Path | StoreLocation",
    zones: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    sample_k: int = STRETCH_SAMPLE_K,
    console: Optional["rich.console.Console"] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    state_url: Optional[str] = None,
    force_lock: bool = False,
) -> int:
    """Rebuild a zone's stretch statistics by scanning its existing shards.

    The repair path for stores filled before fill-time collection existed,
    for interrupted fills, and for suspected double-counting: it re-reads
    the zone's shards once (the only stats path that touches embeddings) and
    *sets* the arrays from what is actually in the store. Creates the arrays
    if the zone predates them. Per-zone, so it composes with fills of other
    zones; takes the same (zone, year) lock a fill would.

    Returns the number of (zone, year) slots rebuilt.
    """
    store = StoreLocation.resolve(store_path, storage_options, state_url)
    zone_names = _zone_group_names(store, zones)
    if not zone_names:
        raise ValueError(f"No UTM zone groups found in {store}")

    rebuilt = 0
    for name in zone_names:
        group = store.open_group(mode="r+", path=name, zarr_format=3)
        zone_years = [int(v) for v in group["time"][:]]
        T = len(zone_years)

        if "stretch_sample" not in group:
            create_stretch_arrays(group, T, sample_k)
            if console:
                console.print(f"  {name}: created stretch-statistics arrays")
        k_slots = group["stretch_sample"].shape[1]

        emb_arr = group["embeddings"]
        scales_arr = group["scales"]
        H, W = emb_arr.shape[2], emb_arr.shape[3]
        all_coords = {
            (sr, sc)
            for sr in range(math.ceil(H / SHARD_SIZE))
            for sc in range(math.ceil(W / SHARD_SIZE))
        }

        for fill_year in years or zone_years:
            if fill_year not in zone_years:
                continue
            t = zone_years.index(fill_year)
            present = _existing_shards(store, name, t, all_coords, console=None)
            if not present:
                continue

            cap = _shard_sample_cap(k_slots, len(present))
            zone_num = int(name[3:])
            _acquire_zone_lock(store, zone_num, fill_year, force=force_lock)
            try:
                n_total, s_total = 0, np.zeros(N_BANDS, dtype=np.float64)
                m_total = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
                candidates: List[Tuple[np.ndarray, np.ndarray, float]] = []
                for i, (sr, sc) in enumerate(sorted(present)):
                    r0, c0 = sr * SHARD_SIZE, sc * SHARD_SIZE
                    r1, c1 = min(r0 + SHARD_SIZE, H), min(c0 + SHARD_SIZE, W)
                    st = shard_stretch_stats(
                        np.asarray(emb_arr[t, :, r0:r1, c0:c1]),
                        np.asarray(scales_arr[t, r0:r1, c0:c1]),
                        cap,
                    )
                    if st is None:
                        continue
                    n_total += st["n"]
                    s_total += st["sum"]
                    m_total += st["prod"]
                    candidates.append(
                        (st["sample_emb"], st["sample_scales"], st["sample_weight"])
                    )
                    if console:
                        console.print(
                            f"  {name} {fill_year}: shard {i + 1}/{len(present)} "
                            f"({st['n']:,} px)",
                            end="\r",
                        )

                emb_s, scales_s = merge_stretch_samples(candidates, k_slots)
                # Backfill SETS from actual contents (it is the repair for
                # double-counting), unlike the fill's additive fold.
                group["stretch_stats_count"][t] = n_total
                group["stretch_stats_sum"][t] = s_total
                group["stretch_stats_prod"][t] = m_total
                full_emb = np.zeros((k_slots, N_BANDS), dtype=np.int8)
                full_sc = np.full(k_slots, np.float32("inf"), dtype=np.float32)
                full_emb[: len(emb_s)] = emb_s
                full_sc[: len(emb_s)] = scales_s
                group["stretch_sample"][t] = full_emb
                group["stretch_sample_scales"][t] = full_sc
                group["stretch_sample_count"][t] = len(emb_s)
                rebuilt += 1
                if console:
                    console.print(
                        f"  {name} {fill_year}: rebuilt from "
                        f"{len(present)} shard(s), {n_total:,} pixels      "
                    )
            finally:
                _release_zone_lock(store, zone_num, fill_year)

    return rebuilt


def _sample_shard_task(
    store_path_str: str,
    zone_group: str,
    time_index: int,
    ci: int,
    cj: int,
    band_indices: tuple,
    max_per_chunk: int,
) -> Optional[np.ndarray]:
    """Worker-side: open the zone arrays and return a valid-pixel sample.

    Lives at module level so ProcessPoolExecutor can pickle it.
    """
    import zarr

    root = zarr.open_group(store_path_str, mode="r", use_consolidated=False)
    zone_store = root[zone_group]
    emb_arr = zone_store["embeddings"]
    scales_arr = zone_store["scales"]
    _, _, H, W = emb_arr.shape
    return _sample_chunk_stats(
        emb_arr,
        scales_arr,
        time_index,
        ci,
        cj,
        SHARD_SIZE,
        (H, W),
        band_indices=band_indices,
        max_per_chunk=max_per_chunk,
    )


def compute_global_stretch(
    store_path: Path,
    year: int,
    target_samples: int = 2_000_000,
    max_shards: Optional[int] = None,
    p_low: float = 2.0,
    p_high: float = 98.0,
    workers: int = 8,
    zones: Optional[List[int]] = None,
    band_indices: Tuple[int, ...] = RGB_PREVIEW_BANDS,
    max_per_chunk: int = 50_000,
    equalise: bool = True,
    equalise_breakpoints: int = 257,
    mode: str = "bands",
    pca_components: int = 3,
    pca_total_bands: int = 128,
    pca_rgb_order: str = "123",
    console: Optional["rich.console.Console"] = None,
) -> dict:
    """Sample valid pixels across every UTM zone until ``target_samples`` are
    collected, then derive one ``(min, max)`` percentile stretch shared by
    the whole world.

    Sparsity-aware: the per-shard sampler (``_sample_chunk_stats``) already
    filters out non-finite scales — both NaN (water) and +inf (land tile
    with no embedding written yet) — so we count only real, dequantisable
    pixels toward the target. Shards are visited in a deterministic shuffled
    order with bounded-batch parallelism; once the target is reached we stop
    draining the queue. The resulting stretch is persisted to the store
    root's ``geoemb:stretch`` attribute under the requested year so
    :func:`build_global_preview` can pick it up.

    Args:
        store_path: Path to the tessera Zarr store.
        year: Time slice the stretch applies to.
        target_samples: Stop after this many valid pixels are collected.
        max_shards: Optional hard cap on shards visited (default: unbounded,
            but the shuffled order means we usually hit ``target_samples``
            in a few hundred shards even on sparse stores).
        p_low/p_high: Percentile bounds for the stretch (default 2/98).
        workers: Parallel I/O threads. Sampling is I/O-bound.
        zones: Optional zone-number filter (e.g. ``[30, 31]``); defaults to
            every UTM zone present in the store.
        band_indices: Bands to compute stretch for (default: RGB triple).
        max_per_chunk: Per-shard cap on sampled pixels.
        console: Optional Rich console for progress prints.

    Returns:
        ``{"min": [..], "max": [..], "samples": int}`` — also written to
        ``<store>/zarr.json``'s ``geoemb:stretch.{year}`` attribute.
    """
    import re
    import zarr
    from concurrent.futures import ThreadPoolExecutor

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)

    if mode not in ("bands", "pca"):
        raise ValueError(f"mode must be 'bands' or 'pca', got {mode!r}")
    # PCA mode samples all 128 bands and learns 3 orthogonal axes by
    # diagonalising the covariance. The output channels are mathematically
    # uncorrelated — fixes the "everything along the grey diagonal" look
    # that hits when the chosen RGB bands are statistically dependent.
    if mode == "pca":
        band_indices = tuple(range(pca_total_bands))

    # Parse the pca_rgb_order permutation now so we fail fast on bad input.
    # pca_perm[k] = which PC ends up in output channel k ("213" swaps R/G).
    pca_perm: Optional[List[int]] = (
        _parse_pca_perm(pca_rgb_order, pca_components) if mode == "pca" else None
    )

    # Find time_index for the requested year via the first zone's time coord.
    time_index = None
    for member_name in _member_names(root):
        if member_name.startswith("utm"):
            try:
                time_arr = root[member_name]["time"][:]
                years_list = [int(v) for v in time_arr]
                if year in years_list:
                    time_index = years_list.index(year)
                    break
            except Exception:
                continue
    if time_index is None:
        raise ValueError(f"Year {year} not present in store {store_path}")

    # Enumerate every (zone, ci, cj) shard.
    zone_pattern = re.compile(r"^utm(\d{2})$")
    all_shards: List[Tuple[str, int, int]] = []
    zones_visited = set()
    for name in _member_names(root):
        m = zone_pattern.match(name)
        if not m:
            continue
        zone_num = int(m.group(1))
        if zones is not None and zone_num not in zones:
            continue
        try:
            emb_arr = root[name]["embeddings"]
            _, _, H, W = emb_arr.shape
        except (KeyError, ValueError):
            continue
        n_rows = math.ceil(H / SHARD_SIZE)
        n_cols = math.ceil(W / SHARD_SIZE)
        zones_visited.add(zone_num)
        for ci in range(n_rows):
            for cj in range(n_cols):
                all_shards.append((name, ci, cj))

    if not all_shards:
        raise RuntimeError(f"No UTM zones with embedding data found in {store_path}")

    # Deterministic shuffle seeded by year so reruns reproduce.
    rng = np.random.default_rng(year)
    rng.shuffle(all_shards)
    if max_shards is not None:
        all_shards = all_shards[:max_shards]

    if console:
        console.print(
            f"Sampling stretch from {len(zones_visited)} zones, "
            f"{len(all_shards):,} shards available, target {target_samples:,} "
            f"valid pixels, workers={workers}"
        )

    # Process in batches so we can stop early once the target is reached.
    batch_size = max(workers * 4, 32)
    collected: List[np.ndarray] = []
    total_valid = 0
    shards_visited = 0
    shards_with_data = 0
    band_tuple = tuple(band_indices)
    store_path_str = str(store_path)

    i = 0
    while total_valid < target_samples and i < len(all_shards):
        batch = all_shards[i : i + batch_size]
        i += batch_size
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _sample_shard_task,
                    store_path_str,
                    zone_group,
                    time_index,
                    ci,
                    cj,
                    band_tuple,
                    max_per_chunk,
                )
                for (zone_group, ci, cj) in batch
            ]
            for fut in futures:
                vals = fut.result()
                shards_visited += 1
                if vals is None or len(vals) == 0:
                    continue
                collected.append(vals)
                shards_with_data += 1
                total_valid += len(vals)
        if console:
            console.print(
                f"  …{total_valid:,}/{target_samples:,} pixels "
                f"({shards_with_data}/{shards_visited} shards had data)"
            )

    if not collected:
        raise RuntimeError(
            "No valid pixels found across any shard — is this store filled?"
        )

    all_vals = np.concatenate(collected, axis=0)
    if all_vals.shape[0] > target_samples * 2:
        # Trim oversize (last batch overshot) for a clean percentile.
        rng2 = np.random.default_rng(year + 1)
        idx = rng2.choice(all_vals.shape[0], target_samples, replace=False)
        all_vals = all_vals[idx]

    # PCA: diagonalise the (128, 128) covariance and keep the top-K
    # eigenvectors. The projected samples (n_samples, K) are mathematically
    # decorrelated, so the K output channels paint orthogonal colour axes
    # rather than redundantly tracking the same model feature.
    pca_proj_components: Optional[List[List[float]]] = None
    pca_proj_mean: Optional[List[float]] = None
    pca_explained_variance_ratio: Optional[List[float]] = None
    if mode == "pca":
        from sklearn.decomposition import PCA

        k = max(1, int(pca_components))
        if all_vals.shape[1] < k:
            raise ValueError(
                f"PCA needs at least {k} bands but the sample has only "
                f"{all_vals.shape[1]}. Set --pca-total-bands lower or use "
                f"--mode bands."
            )
        if console:
            console.print(
                f"Fitting PCA on {all_vals.shape[0]:,} × {all_vals.shape[1]} "
                f"sample matrix (n_components={k})..."
            )
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(all_vals.astype(np.float32, copy=False))

        # Reorder the components by pca_perm so the stored matrix already
        # bakes in the user's preferred PC→RGB mapping. After this, row 0
        # of components_ projects onto the R channel, row 1 onto G, row 2
        # onto B — regardless of which PC originally lived there. The
        # render path doesn't need any extra swapping.
        components_ordered = pca.components_[pca_perm]
        evr_ordered = pca.explained_variance_ratio_[pca_perm]

        # Project all samples through the reordered PCA so the downstream
        # percentile + CDF math is computed in the same channel ordering
        # used at render time.
        centred = all_vals.astype(np.float32, copy=False) - pca.mean_
        all_vals = centred @ components_ordered.T
        pca_proj_components = [[float(v) for v in row] for row in components_ordered]
        pca_proj_mean = [float(v) for v in pca.mean_]
        pca_explained_variance_ratio = [float(v) for v in evr_ordered]
        if console:
            evr_str = ", ".join(f"{v * 100:.1f}%" for v in pca_explained_variance_ratio)
            label = "->".join(["R", "G", "B"][:k])  # channels R, G, B (in store order)
            pc_label = "->".join(
                [f"PC{p + 1}" for p in pca_perm]
            )  # which PCs ended up there
            console.print(
                f"PCA fitted: {pc_label} -> {label}; "
                f"explained variance ratio = [{evr_str}]"
            )

    n_bands = all_vals.shape[1]
    stretch_min = [float(np.percentile(all_vals[:, k], p_low)) for k in range(n_bands)]
    stretch_max = [float(np.percentile(all_vals[:, k], p_high)) for k in range(n_bands)]
    for k in range(n_bands):
        if stretch_max[k] <= stretch_min[k]:
            stretch_max[k] = stretch_min[k] + 1.0

    if console:
        space = "PC" if mode == "pca" else "band"
        console.print(
            f"Stretch in {space} space: "
            f"min={[f'{v:.3f}' for v in stretch_min]}, "
            f"max={[f'{v:.3f}' for v in stretch_max]} "
            f"(from {all_vals.shape[0]:,} pixels)"
        )

    # Optional CDF for histogram equalisation. Each channel's breakpoints
    # are the values at evenly-spaced quantiles 0%, q, 2q, …, 100% with
    # ``equalise_breakpoints`` points total. At render time
    # ``np.searchsorted(breaks, pixel)`` maps a pixel value into a bin
    # index 0..(n_breaks-1), which scales linearly to uint8 — guaranteeing
    # output bytes are uniformly distributed across 0..255.
    cdf_breaks: Optional[List[List[float]]] = None
    if equalise:
        n_break = max(64, int(equalise_breakpoints))
        quantiles = np.linspace(0.0, 100.0, n_break)
        cdf_breaks = []
        for k in range(n_bands):
            bks = np.percentile(all_vals[:, k], quantiles)
            # Ensure strictly increasing so searchsorted is well-defined.
            for j in range(1, len(bks)):
                if bks[j] <= bks[j - 1]:
                    bks[j] = bks[j - 1] + 1e-9
            cdf_breaks.append([float(v) for v in bks])
        if console:
            console.print(
                f"Computed {n_break}-point CDF per channel for histogram equalisation."
            )

    # Persist to root attrs under year-keyed subdict.
    root_rw = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)
    stretch_map = dict(root_rw.attrs.get(_GLOBAL_STRETCH_ATTR, {}))
    method_suffix = "_equalised" if equalise else ""
    method_prefix = "global_pca" if mode == "pca" else "global_percentile"
    entry = {
        "min": stretch_min,
        "max": stretch_max,
        "p_low": p_low,
        "p_high": p_high,
        "samples": int(all_vals.shape[0]),
        "shards_visited": int(shards_visited),
        "shards_with_data": int(shards_with_data),
        "bands": list(band_indices),
        "method": f"{method_prefix}{method_suffix}",
        "mode": mode,
    }
    if cdf_breaks is not None:
        entry["cdf"] = cdf_breaks
    if pca_proj_components is not None:
        entry["pca_components"] = pca_proj_components
        entry["pca_mean"] = pca_proj_mean
        entry["pca_explained_variance_ratio"] = pca_explained_variance_ratio
    stretch_map[str(year)] = entry
    root_rw.attrs[_GLOBAL_STRETCH_ATTR] = stretch_map

    if console:
        console.print(
            f"[green]Saved to {_GLOBAL_STRETCH_ATTR}.{year} on store root.[/green]"
        )

    return {
        "min": stretch_min,
        "max": stretch_max,
        "samples": int(all_vals.shape[0]),
        "cdf": cdf_breaks,
    }


def _load_global_stretch(store_path: Path, year: int) -> Optional[dict]:
    """Look up a previously-computed global stretch for ``year``.

    Returns ``{"min": [..], "max": [..], "cdf": [[..], ..],
    "pca_components": [[..], ...], "pca_mean": [..]}`` — the PCA fields are
    only populated when the stretch was computed in ``mode='pca'``.
    Returns ``None`` if no stretch is stored for the year.
    """
    import zarr

    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)
    stretch_map = root.attrs.get(_GLOBAL_STRETCH_ATTR, {})
    if not isinstance(stretch_map, dict):
        return None
    entry = stretch_map.get(str(year))
    if not entry:
        return None
    out = {"min": list(entry["min"]), "max": list(entry["max"])}
    if "cdf" in entry and entry["cdf"] is not None:
        out["cdf"] = [list(c) for c in entry["cdf"]]
    if "pca_components" in entry and entry["pca_components"] is not None:
        out["pca_components"] = [list(r) for r in entry["pca_components"]]
        out["pca_mean"] = list(entry["pca_mean"])
    out["mode"] = entry.get("mode", "bands")
    return out


# ---------------------------------------------------------------------------
# Global RGB preview pyramid
# ---------------------------------------------------------------------------
# Reprojects per-zone UTM embeddings into a single EPSG:4326 RGB pyramid,
# computing RGB on the fly from bands 0-2 + scales (no stored rgb array).
# ProcessPoolExecutor for reprojection, ThreadPoolExecutor for pyramid
# coarsening.


def _ensure_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global_rgb/ pyramid group within the store."""
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )

    # Check if already exists with correct shape
    if "global_rgb/0/rgb" in root:
        shape = root["global_rgb/0/rgb"].shape
        if shape == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS):
            return
        import shutil

        shutil.rmtree(str(store_path / "global_rgb"))
        root = zarr.open_group(
            str(store_path), mode="r+", zarr_format=3, use_consolidated=False
        )

    # Create pyramid levels via zarr API
    global_grp = root.create_group("global_rgb")
    h, w = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    band_data = np.arange(GLOBAL_NUM_BANDS, dtype=np.int32)

    for lvl in range(num_levels):
        if h < 1 or w < 1:
            break
        lvl_grp = global_grp.create_group(str(lvl))
        lvl_grp.create_array(
            "rgb",
            shape=(h, w, GLOBAL_NUM_BANDS),
            chunks=(GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=["lat", "lon", "band"],
        )
        lvl_grp.create_array(
            "band",
            data=band_data,
            chunks=(GLOBAL_NUM_BANDS,),
            dimension_names=["band"],
        )
        h //= 2
        w //= 2

    # Re-open the global_rgb group to ensure attrs write to the correct handle
    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )
    global_grp = root["global_rgb"]

    # Build multiscale + spatial + proj metadata directly
    # (avoids depending on unstable topozarr API)
    from geozarr_toolkit import (
        create_geozarr_attrs,
        create_multiscales_layout,
    )
    from geozarr_toolkit.conventions.multiscales import MultiscalesConventionMetadata

    west, south, east, north_ = GLOBAL_BOUNDS
    actual_levels = len([k for k in global_grp.keys() if k.isdigit()])

    # Build multiscale layout
    h_lvl, w_lvl = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    res = GLOBAL_BASE_RES
    levels = []
    for lvl in range(actual_levels):
        entry: Dict[str, Any] = {"asset": str(lvl)}
        if lvl > 0:
            entry["derived_from"] = str(lvl - 1)
            entry["transform"] = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
            entry["resampling_method"] = "mean"
        else:
            entry["transform"] = {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}
        entry["spatial:shape"] = [h_lvl, w_lvl]
        entry["spatial:transform"] = [res, 0.0, west, 0.0, -res, north_]
        levels.append(entry)
        h_lvl //= 2
        w_lvl //= 2
        res *= 2.0

    ms_layout = create_multiscales_layout(levels, resampling_method="mean")

    # Geospatial attrs (proj + spatial)
    geozarr_attrs = create_geozarr_attrs(
        dimensions=["lat", "lon"],
        crs="EPSG:4326",
        bbox=[west, south, east, north_],
    )

    # Fix spatial description bug in geozarr-toolkit
    for conv in geozarr_attrs.get("zarr_conventions", []):
        if conv.get("uuid") == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4":
            conv["description"] = "Spatial coordinate information"

    # Add multiscales convention registration
    ms_conv = MultiscalesConventionMetadata()
    geozarr_attrs["zarr_conventions"].insert(0, ms_conv.model_dump(exclude_none=True))

    # Merge all attrs
    geozarr_attrs.update(ms_layout)
    global_grp.attrs.update(geozarr_attrs)


# Per-worker state for reprojection
_reproj_global_arr = None
_reproj_emb_arr = None
_reproj_scales_arr = None
_reproj_to_utm = None
_reproj_time_index = None
_reproj_stretch = None


def _init_reproj_worker(
    store_path: str,
    zone_group: str,
    zone_epsg: int,
    time_index: int,
    stretch: dict,
) -> None:
    """Process pool initializer: open stores and create transformer."""
    global _reproj_global_arr, _reproj_emb_arr, _reproj_scales_arr
    global _reproj_to_utm, _reproj_time_index, _reproj_stretch
    import zarr
    from pyproj import Transformer

    root = zarr.open_group(store_path, mode="r+", zarr_format=3, use_consolidated=False)
    _reproj_global_arr = root["global_rgb/0/rgb"]
    zone = root[zone_group]
    _reproj_emb_arr = zone["embeddings"]
    _reproj_scales_arr = zone["scales"]
    _reproj_to_utm = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{zone_epsg}",
        always_xy=True,
    )
    _reproj_time_index = time_index
    _reproj_stretch = stretch


def _reproject_chunk_worker(args) -> bool:
    """Process pool worker for reprojection."""
    (
        chunk_row,
        chunk_col,
        src_epsg,
        src_pixel,
        src_origin_e,
        src_origin_n,
        src_h,
        src_w,
    ) = args
    return _reproject_chunk(
        _reproj_global_arr,
        chunk_row,
        chunk_col,
        _reproj_emb_arr,
        _reproj_scales_arr,
        _reproj_time_index,
        _reproj_stretch,
        src_epsg,
        src_pixel,
        src_origin_e,
        src_origin_n,
        src_h,
        src_w,
        _reproj_to_utm,
    )


def _reproject_chunk(
    global_arr,
    chunk_row: int,
    chunk_col: int,
    emb_arr,
    scales_arr,
    time_index: int,
    stretch: dict,
    src_epsg: int,
    src_pixel: float,
    src_origin_e: float,
    src_origin_n: float,
    src_h: int,
    src_w: int,
    to_utm,
) -> bool:
    """Reproject one 512x512 global chunk, computing RGB from embeddings on the fly."""
    import warnings
    from affine import Affine
    from rasterio.enums import Resampling
    import rasterio.warp
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    west, _south, _east, north_ = GLOBAL_BOUNDS
    row0 = chunk_row * GLOBAL_CHUNK
    col0 = chunk_col * GLOBAL_CHUNK
    tile_h = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_H - row0)
    tile_w = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_W - col0)
    if tile_h <= 0 or tile_w <= 0:
        return False

    tile_west = west + col0 * GLOBAL_BASE_RES
    tile_north = north_ - row0 * GLOBAL_BASE_RES
    tile_east = tile_west + tile_w * GLOBAL_BASE_RES
    tile_south = tile_north - tile_h * GLOBAL_BASE_RES

    dst_transform = Affine(
        GLOBAL_BASE_RES,
        0,
        tile_west,
        0,
        -GLOBAL_BASE_RES,
        tile_north,
    )

    # Sample corners to check zone coverage
    sample_lons = [
        tile_west,
        tile_east,
        tile_west,
        tile_east,
        (tile_west + tile_east) / 2,
    ]
    sample_lats = [
        tile_north,
        tile_north,
        tile_south,
        tile_south,
        (tile_north + tile_south) / 2,
    ]
    try:
        utm_xs, utm_ys = to_utm.transform(sample_lons, sample_lats)
    except Exception:
        return False

    if any(not math.isfinite(v) for v in list(utm_xs) + list(utm_ys)):
        return False

    # Compute source window in UTM pixel coords
    pad = 16
    r_min = max(0, int((src_origin_n - max(utm_ys)) / src_pixel) - pad)
    r_max = min(src_h, int(math.ceil((src_origin_n - min(utm_ys)) / src_pixel)) + pad)
    c_min = max(0, int((min(utm_xs) - src_origin_e) / src_pixel) - pad)
    c_max = min(src_w, int(math.ceil((max(utm_xs) - src_origin_e) / src_pixel)) + pad)

    if r_max <= r_min or c_max <= c_min:
        return False

    # Compute RGB on the fly from embeddings + scales (no stored rgb array needed)
    scales_chunk = np.asarray(scales_arr[time_index, r_min:r_max, c_min:c_max])
    valid = np.isfinite(scales_chunk)
    if not np.any(valid):
        return False

    # In PCA mode we need every band the projection matrix expects (usually
    # all 128); in linear/bands mode we only need the contiguous RGB slice.
    pca_components = stretch.get("pca_components")
    pca_mean = stretch.get("pca_mean")
    if pca_components is not None:
        n_pca_bands = len(pca_components[0])
        emb_chunk = np.asarray(
            emb_arr[time_index, 0:n_pca_bands, r_min:r_max, c_min:c_max]
        )
        band_tuple: Tuple[int, ...] = tuple(range(n_pca_bands))
    else:
        b0, b1 = RGB_PREVIEW_BANDS[0], RGB_PREVIEW_BANDS[-1] + 1
        emb_chunk = np.asarray(emb_arr[time_index, b0:b1, r_min:r_max, c_min:c_max])
        band_tuple = tuple(range(b1 - b0))
    rgba = _compute_rgb_chunk(
        emb_chunk,
        scales_chunk,
        band_tuple,
        stretch["min"],
        stretch["max"],
        cdf=stretch.get("cdf"),
        gamma=stretch.get("gamma", 1.0),
        saturation=stretch.get("saturation", 1.0),
        pca_components=pca_components,
        pca_mean=pca_mean,
    )  # (4, h, w) uint8

    src_data = rgba.astype(np.float32)
    del emb_chunk, scales_chunk

    win_transform = Affine(
        src_pixel,
        0,
        src_origin_e + c_min * src_pixel,
        0,
        -src_pixel,
        src_origin_n - r_min * src_pixel,
    )

    # Mask invalid pixels (alpha < 128) as NaN
    alpha_band = src_data[3]
    invalid = alpha_band < 128
    for b in range(3):
        src_data[b][invalid] = np.nan
    rgb_src = src_data[:3]
    del src_data

    rgb_dst = np.full((3, tile_h, tile_w), np.nan, dtype=np.float32)

    try:
        rasterio.warp.reproject(
            source=rgb_src,
            destination=rgb_dst,
            src_transform=win_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    except Exception:
        return False

    del rgb_src

    # Derive alpha from valid reprojected RGB
    has_data = np.any(np.isfinite(rgb_dst) & (rgb_dst != 0), axis=0)
    rgb_dst = np.nan_to_num(rgb_dst, nan=0.0)
    rgb_dst = np.clip(rgb_dst, 0, 255).astype(np.uint8)
    rgb_out = np.transpose(rgb_dst, (1, 2, 0))  # (h, w, 3)
    del rgb_dst

    out = np.zeros((tile_h, tile_w, GLOBAL_NUM_BANDS), dtype=np.uint8)
    out[:, :, :3] = rgb_out
    out[:, :, 3] = np.where(has_data, 255, 0).astype(np.uint8)
    del rgb_out

    if not out.any():
        return False

    # Composite: only overwrite pixels where new zone has data
    mask = out.any(axis=2)
    if mask.all():
        global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = out
    else:
        existing = np.asarray(global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :])
        existing[mask] = out[mask]
        global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = existing
    return True


def _reproject_zone(
    store_path: Path,
    zone_num: int,
    zone_group: str,
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
    time_index: int,
    stretch: dict,
    workers: int,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> Tuple[int, int, int, int, bool]:
    """Reproject one zone's embeddings into global level 0 (computing RGB on the fly)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    row_start, row_end, col_start, col_end = _zone_output_bounds(
        zone_epsg=zone_epsg,
        zone_transform=zone_transform,
        zone_shape=(src_h, src_w),
    )

    if col_end <= col_start or row_end <= row_start:
        if console:
            console.print(f"    [yellow]Zone {zone_num}: no output region[/yellow]")
        return (0, 0, 0, 0, False)

    n_chunk_rows = (row_end - row_start) // GLOBAL_CHUNK
    n_chunk_cols = (col_end - col_start) // GLOBAL_CHUNK
    chunk_row_start = row_start // GLOBAL_CHUNK
    chunk_col_start = col_start // GLOBAL_CHUNK

    # Resume check. The marker lives in the state sibling, not the store, so
    # the published hierarchy stays free of non-Zarr objects.
    marker = _preview_marker_path(store_path, zone_num)
    if marker.exists():
        if force:
            marker.unlink()
        else:
            if console:
                console.print(f"    Zone {zone_num:02d}: already complete, skipping")
            return (row_start, row_end, col_start, col_end, False)

    chunks_total = n_chunk_rows * n_chunk_cols
    if console:
        console.print(
            f"    Zone {zone_num:02d}: {n_chunk_rows}x{n_chunk_cols} "
            f"= {chunks_total} chunks"
        )

    work_items = [
        (
            chunk_row_start + cr,
            chunk_col_start + cc,
            zone_epsg,
            src_pixel,
            src_origin_e,
            src_origin_n,
            src_h,
            src_w,
        )
        for cr in range(n_chunk_rows)
        for cc in range(n_chunk_cols)
    ]

    chunks_written = 0

    if console:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            MofNCompleteColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task(
                f"Reprojecting zone {zone_num:02d}",
                total=len(work_items),
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_reproj_worker,
                initargs=(str(store_path), zone_group, zone_epsg, time_index, stretch),
            ) as pool:
                futures = {
                    pool.submit(_reproject_chunk_worker, item): item
                    for item in work_items
                }
                for future in as_completed(futures):
                    try:
                        if future.result():
                            chunks_written += 1
                    except Exception as e:
                        logger.warning(f"Reproject chunk failed: {e}")
                    progress.advance(ptask)
        console.print(f"    {chunks_written} chunks with data")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_reproj_worker,
            initargs=(str(store_path), zone_group, zone_epsg, time_index, stretch),
        ) as pool:
            futures = {
                pool.submit(_reproject_chunk_worker, item): item for item in work_items
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        chunks_written += 1
                except Exception as e:
                    logger.warning(f"Reproject chunk failed: {e}")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"zone={zone_num} chunks={chunks_total} written={chunks_written}\n"
    )

    return (row_start, row_end, col_start, col_end, True)


def build_global_preview(
    store_path: Path,
    year: int = 2024,
    zones: Optional[List[int]] = None,
    num_levels: int = GLOBAL_DEFAULT_LEVELS,
    workers: int = 4,
    gamma: float = 1.0,
    saturation: float = 1.0,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> None:
    """Build the global EPSG:4326 RGB pyramid from zone-level embeddings.

    Computes RGB from embeddings+scales (bands 0-2) for the specified year,
    reprojects from UTM to geographic coordinates and composites into the
    pyramid. No pre-computed rgb array needed.

    Args:
        gamma: Per-channel gamma applied after normalisation. ``< 1.0``
            brightens midtones (standard EO preview is 0.6–0.8); ``1.0``
            (default) leaves the linear/equalised mapping alone.
        saturation: Multiplier on the chroma component (distance from
            per-pixel luma). ``> 1.0`` makes colours pop, ``2.0`` is vivid,
            ``1.0`` (default) is unchanged. Useful when bands 0-2 are
            correlated and the RGB clusters along the grey diagonal.
    """
    import re
    import warnings
    import zarr
    import gc

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)

    # Derive years from first zone's time coordinate
    all_years: list[int] = []
    for member_name in _member_names(root):
        if member_name.startswith("utm"):
            try:
                time_arr = root[member_name]["time"][:]
                all_years = [int(v) for v in time_arr]
                break
            except Exception:
                continue

    if not all_years:
        if console:
            console.print("[red]Error: no years found in store[/red]")
        return

    if year not in all_years:
        if console:
            console.print(
                f"[red]Error: year {year} not in store (available: {all_years})[/red]"
            )
        return

    time_index = all_years.index(year)

    if console:
        console.print(f"Building global preview for year {year} (t={time_index})")

    # Discover zones with embedding data
    zone_pattern = re.compile(r"^utm(\d{2})$")
    zone_infos: Dict[int, dict] = {}

    for name in _member_names(root):
        m = zone_pattern.match(name)
        if not m:
            continue
        zone_num = int(m.group(1))
        if zones is not None and zone_num not in zones:
            continue

        zone_store = root[name]
        attrs = dict(zone_store.attrs)

        try:
            emb_arr = zone_store["embeddings"]
            _, _, zone_h, zone_w = emb_arr.shape
        except (KeyError, ValueError):
            continue

        zone_infos[zone_num] = {
            "zone_group": name,
            "epsg": int(attrs["proj:code"].split(":")[1]),
            "transform": list(attrs["spatial:transform"]),
            "shape": (zone_h, zone_w),
        }

    if not zone_infos:
        if console:
            console.print("[yellow]No zones with embedding data found[/yellow]")
        return

    if console:
        console.print(f"  {len(zone_infos)} zone(s) with data")

    # Ensure global pyramid structure exists
    _ensure_global_store(store_path, num_levels)

    # Prefer a pre-computed cross-zone stretch (written by `zarr-stretch`).
    # Using one shared stretch eliminates inter-zone colour discontinuities.
    global_stretch = _load_global_stretch(store_path, year)
    if global_stretch is not None:
        global_stretch["gamma"] = gamma
        global_stretch["saturation"] = saturation
        if console:
            has_cdf = "cdf" in global_stretch
            console.print(
                f"[cyan]Using global stretch from store attrs "
                f"(mode={'CDF-equalised' if has_cdf else 'linear'}, "
                f"gamma={gamma}, saturation={saturation})[/cyan]"
            )
    elif console:
        console.print(
            "[yellow]No global stretch attribute found. Falling back to "
            "per-zone stretch — the mosaic may show colour seams. "
            "Run `geotessera-registry zarr-stretch <store> --year "
            f"{year}` first for seamless colours.[/yellow]"
        )

    # Reproject each zone and build pyramid
    for zone_num, info in sorted(zone_infos.items()):
        if console:
            console.print(f"\n  Zone {zone_num:02d}:")

        if global_stretch is not None:
            stretch = global_stretch
        else:
            # Fallback: per-zone stretch (produces seams at zone boundaries).
            zone_store = zarr.open_group(
                str(store_path),
                mode="r",
                path=info["zone_group"],
                zarr_format=3,
                use_consolidated=False,
            )
            if console:
                console.print("    Sampling stretch...")
            stretch = compute_stretch(
                zone_store,
                time_index,
                workers=workers,
                console=console,
            )
            stretch["gamma"] = gamma
            stretch["saturation"] = saturation
            if console:
                console.print(
                    f"    Stretch: min={[f'{v:.3f}' for v in stretch['min']]}, "
                    f"max={[f'{v:.3f}' for v in stretch['max']]}, "
                    f"gamma={gamma}, saturation={saturation}"
                )

        row_start, row_end, col_start, col_end, did_work = _reproject_zone(
            store_path=store_path,
            zone_num=zone_num,
            zone_group=info["zone_group"],
            zone_epsg=info["epsg"],
            zone_transform=info["transform"],
            zone_shape=info["shape"],
            time_index=time_index,
            stretch=stretch,
            workers=workers,
            console=console,
            force=force,
        )

        if did_work:
            if console:
                console.print("    Building pyramid...")
            _coarsen_zone_pyramid(
                store_path=store_path,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                num_levels=num_levels,
                workers=workers,
                console=console,
            )

        gc.collect()

    # Consolidate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(str(store_path))

    if console:
        console.print("\n  [green]Global preview complete[/green]")
