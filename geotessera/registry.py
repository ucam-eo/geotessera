"""Registry management for Tessera data files.

This module handles all registry-related operations including loading and querying
the Parquet registry, and direct HTTP downloads with local caching.

Also includes utilities for block-based registry management, organizing global grid
data into 5x5 degree blocks for efficient data access.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Iterator, Callable
import os
import math
import re
import errno
import hashlib
import logging
import numpy as np
import time
from datetime import datetime, timezone
from email.utils import formatdate, parsedate_to_datetime

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for registry operations")

try:
    import geopandas as gpd
except ImportError:
    raise ImportError(
        "geopandas is required for spatial operations. Install with: pip install geopandas"
    )

# Constants for block-based registry management
BLOCK_SIZE = 5  # 5x5 degree blocks

# Default dataset variant. The bare ``global_0.1_degree_representation``
# subdirectory corresponds to this variant; named variants get a ``.<name>``
# suffix.
DEFAULT_VARIANT = "vultr"

# Known (version, variant) datasets and their directory names in the npy/
# tree, one row per pair: (normalised version, variant, npy/ directory).
# A ``None`` directory reserves a variant that is not yet published
# ("coming soon"). The v1 series predates the variant-suffix scheme, so
# every 1.0 variant shares the bare ``v1/`` directory; later versions get
# one directory per variant, named ``{version_path}-{suffix}``.
KNOWN_DATASETS = (
    ("1.0", "vultr", "v1"),
    ("1.1", "cambridge", "v1.1-cam"),
    ("1.1", "dclimate", None),  # complete global v1.1 run — coming soon
    ("2.0", "2B-L~beta1", "v2-2B-L~beta1"),
    ("2.0", "2B-L~beta2", "v2-2B-L~beta2"),
)

# Each version's default variant is the first *published* (non-None)
# variant listed for it in KNOWN_DATASETS. Reorder the rows to change a
# default when a newer variant (e.g. the v1.1 ``dclimate`` run) becomes
# the preferred one.
VERSION_DEFAULT_VARIANTS: Dict[str, str] = {}
for _version, _variant, _dir in KNOWN_DATASETS:
    if _dir is not None:
        VERSION_DEFAULT_VARIANTS.setdefault(_version, _variant)


def default_variant(version_norm: str) -> str:
    """Default variant for *version_norm* (e.g. ``"1.1"`` → ``"cambridge"``)."""
    return VERSION_DEFAULT_VARIANTS.get(version_norm, DEFAULT_VARIANT)


def known_variants(version_norm: str) -> List[Tuple[str, Optional[str]]]:
    """Known variants for *version_norm* as ``(variant, npy_directory)`` pairs.

    A ``None`` directory marks a variant that is reserved but not yet
    published ("coming soon").
    """
    return [(var, d) for v, var, d in KNOWN_DATASETS if v == version_norm]


def published_datasets() -> List[Tuple[str, str, str]]:
    """Published ``(version_norm, variant, npy_directory)`` triples.

    Skips reserved coming-soon variants. Multi-dataset operations (e.g.
    ``coverage --by-source --dataset-version=all``) enumerate manifests
    from this list rather than issuing a bucket listing call.
    """
    return [(v, var, d) for v, var, d in KNOWN_DATASETS if d is not None]


def _parse_dataset_version(spec: str) -> Tuple[str, str]:
    """Parse a flexible dataset-version spec.

    Returns ``(version_path, normalized_version)``. Accepts inputs like
    ``"v1"``, ``"1"``, ``"1.0"``, ``"v1.0"`` (all → ``("v1", "1.0")``) and
    ``"v1.1"``, ``"1.1"`` (→ ``("v1.1", "1.1")``). The repository uses
    ``v1/`` for the 1.0 series, so ``.0`` minors are stripped from the path.
    """
    s = spec.strip()
    if s.startswith("v"):
        s = s[1:]
    parts = s.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    norm = f"{major}.{minor}"
    path = f"v{major}" if minor == "0" else f"v{major}.{minor}"
    return path, norm


def _variant_subdir(variant: str) -> str:
    """Map a variant name to its embeddings subdirectory name."""
    if variant == DEFAULT_VARIANT:
        return EMBEDDINGS_DIR_NAME
    return f"{EMBEDDINGS_DIR_NAME}.{variant}"


def _version_path_from_norm(norm: str) -> str:
    """Inverse of ``_parse_dataset_version`` for the path component.

    ``"1.0"`` → ``"v1"`` (the repository uses ``v1/`` for the 1.0 series);
    ``"1.1"`` → ``"v1.1"``; ``"2.0"`` → ``"v2"``; etc.
    """
    major, _, minor = norm.partition(".")
    if not minor or minor == "0":
        return f"v{major}"
    return f"v{major}.{minor}"


def dataset_path(version_norm: str, variant: str) -> str:
    """Directory in the ``npy/`` tree for a (version, variant) pair.

    The v1 series predates the variant-suffix scheme: every 1.0 variant
    (including the default ``vultr``) maps to the bare ``v1`` directory.
    Later versions get one directory per variant, e.g.
    ``("1.1", "cambridge")`` → ``"v1.1-cam"`` and
    ``("2.0", "2B-L~beta1")`` → ``"v2-2B-L~beta1"``.

    Raises:
        ValueError: If the variant is reserved but not yet published
            (e.g. ``("1.1", "dclimate")``).
    """
    if version_norm == "1.0":
        return "v1"
    for v, var, dirname in KNOWN_DATASETS:
        if v == version_norm and var == variant:
            if dirname is None:
                published = [
                    var2 for var2, d in known_variants(version_norm) if d is not None
                ]
                raise ValueError(
                    f"Dataset variant {variant!r} for version {version_norm} "
                    f"is coming soon but not yet published. Currently "
                    f"available variant(s) for {version_norm}: "
                    f"{', '.join(published) or 'none'}. Run 'geotessera info' "
                    f"to list all datasets."
                )
            return dirname
    # Unknown pair: assume the generic ``{version_path}-{variant}`` naming
    # so freshly published datasets work without a code change.
    return f"{_version_path_from_norm(version_norm)}-{variant}"


# Dataset directory names: a version path with an optional -variant suffix
# (``v1``, ``v1.1-cam``, ``v2-2B-L~beta1``).
_DATASET_DIR_RE = re.compile(r"^v(\d+)(?:\.(\d+))?(?:-(.+))?$")


def dataset_from_path(dirname: str) -> Optional[Tuple[str, str]]:
    """Inverse of :func:`dataset_path`: ``"v1.1-cam"`` → ``("1.1", "cambridge")``.

    Bare version directories resolve to the version's default variant
    (``"v1"`` → ``("1.0", "vultr")``); unknown suffixes are taken verbatim
    as the variant name. Returns ``None`` if *dirname* does not look like
    a dataset directory.
    """
    for v, var, d in KNOWN_DATASETS:
        if d == dirname:
            return v, var
    m = _DATASET_DIR_RE.match(dirname)
    if not m:
        return None
    version_norm = f"{m.group(1)}.{m.group(2) or '0'}"
    variant = m.group(3) or default_variant(version_norm)
    return version_norm, variant


# Sidecar filename written into output directories alongside downloaded tiles
# (NPY format) to record which dataset version/variant produced the files.
TESSERA_METADATA_FILENAME = "tessera_metadata.json"


def write_tessera_metadata(
    output_dir: Union[str, Path],
    dataset_version: str,
    dataset_variant: str,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write a ``tessera_metadata.json`` sidecar describing a download.

    The local layout always uses ``global_0.1_degree_representation/`` for NPY
    tiles regardless of variant; this sidecar records the provenance so
    downstream tools can recover which dataset was downloaded.
    """
    import json

    version_path, version_norm = _parse_dataset_version(dataset_version)
    payload: Dict[str, object] = {
        "dataset_version": version_norm,
        "dataset_version_path": version_path,
        "dataset_variant": dataset_variant,
        # Directory in the npy/ tree this (version, variant) pair was
        # downloaded from (e.g. "v1", "v1.1-cam", "v2-2B-L~beta1").
        "dataset_path": dataset_path(version_norm, dataset_variant),
        "embeddings_subdir": EMBEDDINGS_DIR_NAME,
        # Variant-aware subdirectory name, kept for downstream readers. The
        # repository layout itself has no variant level.
        "s3_embeddings_subdir": _variant_subdir(dataset_variant),
        "source_url_prefix": f"{TESSERA_NPY_MIRROR_URL}/{dataset_path(version_norm, dataset_variant)}/",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    out = Path(output_dir) / TESSERA_METADATA_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


# ==============================================================================
# COORDINATE SYSTEM HIERARCHY
# ==============================================================================
# This module uses a three-level coordinate hierarchy:
#
# 1. BLOCKS (5×5 degrees): Registry files are organized into blocks for efficient
#    loading. Each block contains up to 2,500 tiles (50×50 grid).
#
# 2. TILES (0.1×0.1 degrees): Individual data files containing embeddings or
#    landmasks. Tiles are centered at 0.05-degree offsets (e.g., 0.05, 0.15, 0.25).
#
# 3. WORLD: Arbitrary decimal degree coordinates provided by users.
#
# Function naming convention:
# - block_* : Operations on 5-degree registry blocks
# - tile_*  : Operations on 0.1-degree data tiles
# - *_from_world : Convert from arbitrary coordinates to block/tile coords
# ==============================================================================


# Block-level functions (5-degree registry organization)
def block_from_world(lon: float, lat: float) -> Tuple[int, int]:
    """Convert world coordinates to containing registry block coordinates.

    Registry blocks are 5×5 degree squares used to organize registry files.
    Each block can contain up to 2,500 tiles.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        tuple: (block_lon, block_lat) lower-left corner of the containing block

    Raises:
        ValueError: If coordinates are out of bounds or not finite

    Examples:
        >>> block_from_world(3.2, 52.7)
        (0, 50)
        >>> block_from_world(-7.8, -23.4)
        (-10, -25)
    """
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of bounds [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of bounds [-90, 90]")
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Coordinates must be finite numbers")

    block_lon = math.floor(lon / BLOCK_SIZE) * BLOCK_SIZE
    block_lat = math.floor(lat / BLOCK_SIZE) * BLOCK_SIZE
    return int(block_lon), int(block_lat)


def block_to_embeddings_registry_filename(
    year: str, block_lon: int, block_lat: int
) -> str:
    """Generate registry filename for an embeddings block.

    Args:
        year: Year string (e.g., "2024")
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)

    Returns:
        str: Registry filename like "embeddings_2024_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"embeddings_{year}_{lon_str}_{lat_str}.txt"


def block_to_landmasks_registry_filename(block_lon: int, block_lat: int) -> str:
    """Generate registry filename for a landmasks block.

    Args:
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)

    Returns:
        str: Registry filename like "landmasks_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"landmasks_{lon_str}_{lat_str}.txt"


# Tile-level functions (0.1-degree data tiles)
def tile_from_world(lon: float, lat: float) -> Tuple[float, float]:
    """Convert world coordinates to containing tile center coordinates.

    Tiles are 0.1×0.1 degree squares centered at 0.05-degree offsets
    (e.g., -0.05, 0.05, 0.15, 0.25, etc.).

    Args:
        lon: World longitude in decimal degrees
        lat: World latitude in decimal degrees

    Returns:
        Tuple of (tile_lon, tile_lat) representing the tile center

    Raises:
        ValueError: If coordinates are out of bounds or not finite

    Examples:
        >>> tile_from_world(0.17, 52.23)
        (0.15, 52.25)
        >>> tile_from_world(-0.12, -0.03)
        (-0.15, -0.05)
    """
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of bounds [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of bounds [-90, 90]")
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Coordinates must be finite numbers")

    tile_lon = np.floor(lon * 10) / 10 + 0.05
    tile_lat = np.floor(lat * 10) / 10 + 0.05
    return round(float(tile_lon), 2), round(float(tile_lat), 2)


def parse_grid_name(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract tile coordinates from a grid filename.

    Args:
        filename: Grid filename like "grid_-50.55_-20.65"

    Returns:
        tuple: (lon, lat) as floats, or (None, None) if parsing fails
    """
    match = re.match(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def coord_to_grid_int(coord: float) -> np.int32:
    """Convert a float coordinate to an integer grid index.

    Multiplies by 100 and rounds to get an integer index.
    This is the inverse of grid_int_to_coord().

    Args:
        coord: Longitude or latitude coordinate (e.g., -0.15, 51.55)

    Returns:
        Integer grid index (e.g., -15, 5155) as numpy.int32
    """
    return np.int32(np.round(coord * 100))


def tile_to_grid_name(lon: float, lat: float) -> str:
    """Generate grid name for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        str: Grid name like "grid_-50.55_-20.65"
    """
    return f"grid_{lon:.2f}_{lat:.2f}"


def tile_to_embedding_paths(lon: float, lat: float, year: int) -> Tuple[Path, Path]:
    """Generate embedding and scales file paths for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude
        year: Year of embeddings

    Returns:
        Tuple of (embedding_path, scales_path) as Path objects
    """
    grid_name = tile_to_grid_name(lon, lat)
    embedding_path = Path(str(year)) / grid_name / f"{grid_name}.npy"
    scales_path = Path(str(year)) / grid_name / f"{grid_name}_scales.npy"
    return embedding_path, scales_path


def tile_to_geotiff_path(lon: float, lat: float, year: int) -> Path:
    """Generate GeoTIFF file path for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude
        year: Year of embeddings

    Returns:
        Path: Relative path like "{year}/grid_{lon}_{lat}/grid_{lon}_{lat}_{year}.tiff"
    """
    grid_name = tile_to_grid_name(lon, lat)
    return Path(str(year)) / grid_name / f"{grid_name}_{year}.tiff"


def tile_to_landmask_filename(lon: float, lat: float) -> str:
    """Generate landmask filename for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        Landmask filename like "grid_0.15_52.25.tiff"
    """
    return f"{tile_to_grid_name(lon, lat)}.tiff"


def tile_to_bounds(lon: float, lat: float) -> Tuple[float, float, float, float]:
    """Get geographic bounds for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        Tuple of (west, south, east, north) bounds
    """
    return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)


# All Tessera data is served over plain HTTPS from the public Source
# Cooperative repository. The npy/ tree has one directory per dataset —
# a (version, variant) pair (see KNOWN_DATASETS / dataset_path); the
# landmasks/ and zarr/ trees are keyed by plain version:
#   npy/{dataset_path}/{year}/grid_.../grid_....npy    embeddings and scales
#   npy/{dataset_path}/manifest.parquet                per-dataset manifest
#   landmasks/{version_path}/grid_....tiff             landmask TIFFs
#   landmasks/{version_path}/landmasks.parquet         landmask registry
#   zarr/{version_path}/                               zarr store root
# Each dataset directory holds a complete embedding tree with no variant
# subdirectory. New datasets appear as they are uploaded. The repository is
# also reachable as an S3-compatible endpoint at TESSERA_MIRROR_ENDPOINT
# with bucket path TESSERA_MIRROR_REPO.
TESSERA_MIRROR_ENDPOINT = "https://data.source.coop"
TESSERA_MIRROR_REPO = "tessera/tessera"
TESSERA_MIRROR_URL = f"{TESSERA_MIRROR_ENDPOINT}/{TESSERA_MIRROR_REPO}"
TESSERA_NPY_MIRROR_URL = f"{TESSERA_MIRROR_URL}/npy"
TESSERA_LANDMASKS_MIRROR_URL = f"{TESSERA_MIRROR_URL}/landmasks"

# Subdirectory names used in the local embeddings_dir layout.
EMBEDDINGS_DIR_NAME = "global_0.1_degree_representation"  # NPY embeddings and scales
LANDMASKS_DIR_NAME = "global_0.1_degree_tiff_all"  # Landmask TIFFs


def manifest_url(dataset_dir: str) -> str:
    """Default URL of the embeddings manifest for *dataset_dir*.

    *dataset_dir* is an npy/ tree directory name from :func:`dataset_path`
    (e.g. ``"v1"``, ``"v1.1-cam"``, ``"v2-2B-L~beta1"``).
    """
    return f"{TESSERA_NPY_MIRROR_URL}/{dataset_dir}/manifest.parquet"


def landmasks_parquet_url(version_path: str) -> str:
    """Default URL of the landmasks registry for *version_path* (e.g. ``"v1"``)."""
    return f"{TESSERA_LANDMASKS_MIRROR_URL}/{version_path}/landmasks.parquet"


def embedding_url(dataset_dir: str, path: str) -> str:
    """URL of the embedding file at *path* within the npy/ *dataset_dir*."""
    return f"{TESSERA_NPY_MIRROR_URL}/{dataset_dir}/{path}"


def landmask_url(version_path: str, filename: str) -> str:
    """URL of the landmask TIFF *filename* for *version_path*."""
    return f"{TESSERA_LANDMASKS_MIRROR_URL}/{version_path}/{filename}"


def zarr_store_url(version: str) -> str:
    """Default URL of the zarr store for *version*.

    Accepts a version name (``"v1"``, ``"v2"``), resolved through the
    version's default variant, or an explicit store path such as
    ``"v2-2B-L~beta1"``.
    """
    version_path, norm = _parse_dataset_version(version)
    if norm in VERSION_DEFAULT_VARIANTS:
        variant = default_variant(norm)
        if norm == "1.1":
            version = version_path
        else:
            version = dataset_path(norm, variant)
    return f"{TESSERA_MIRROR_URL}/zarr/{version}"


def format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. ``"1.5 GB"``)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


# Write failures that say the destination itself is at fault, so that no
# number of retries will help.
_PERMANENT_WRITE_ERRNOS = frozenset(
    {
        errno.EROFS,  # read-only file system
        errno.EACCES,  # permission denied
        errno.EPERM,  # operation not permitted
        errno.ENOSPC,  # no space left on device
        errno.EDQUOT,  # disk quota exceeded
    }
)


class UnwritableDestination(OSError):
    """A download destination that cannot be written, and will stay so.

    Distinguishes an ``embeddings_dir`` that is read-only, unwritable or
    full from a transfer that merely failed this time, so that callers can
    abandon a whole batch of downloads rather than retry each in turn.
    """


def _unwritable(exc: OSError) -> bool:
    """Report whether *exc* is a write failure that no retry can fix."""
    return exc.errno in _PERMANENT_WRITE_ERRNOS


def _preflight_destination(cache_path: Path) -> None:
    """Check that *cache_path* can be written, before any request is made.

    Nothing is written until the response is already streaming, so an
    unwritable destination otherwise costs a full transfer to discover,
    once per file.

    Raises:
        UnwritableDestination: If the parent directory cannot be created,
            or is not writable.
    """
    parent = Path(cache_path).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if _unwritable(e):
            raise UnwritableDestination(
                e.errno, f"Cannot create {parent}: {e.strerror}"
            ) from e
        raise
    # Advisory: an existing but read-only directory survives the mkdir
    # above. The retry loop still classifies whatever the write raises.
    if not os.access(parent, os.W_OK):
        raise UnwritableDestination(
            errno.EACCES, f"Cannot write to {parent}: Permission denied"
        )


class HTTPStatusError(OSError):
    """A non-retryable HTTP status, such as 403 or 404.

    Retryable statuses never reach here: urllib3's ``Retry`` handles them and
    raises ``MaxRetryError`` once exhausted.
    """

    def __init__(self, url: str, status: int, reason: Optional[str]):
        super().__init__(f"HTTP {status} {reason or ''} for {url}".rstrip())
        self.url = url
        self.status = status


_pool = None


def _http():
    """Return the process-wide urllib3 pool used for every HTTPS request.

    One pool means the thousands of per-tile GETs in a region download reuse
    connections rather than paying a TCP and TLS handshake each. It retries
    transient failures with exponential backoff, and enforces the response
    ``Content-Length``, so a truncated body raises rather than ending the
    stream silently.
    """
    global _pool
    if _pool is None:
        import urllib3

        from . import __version__

        _pool = urllib3.PoolManager(
            # The CDN answers 403 to some default client User-Agents.
            headers={"User-Agent": f"geotessera/{__version__}"},
            retries=urllib3.Retry(
                total=5,
                backoff_factor=1,  # 1, 2, 4, 8, 16s between attempts
                status_forcelist=(429, 500, 502, 503, 504),
                respect_retry_after_header=True,
            ),
            timeout=urllib3.Timeout(connect=15, read=60),
        )
    return _pool


def download_file_to_temp(
    url: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cache_path: Optional[Path] = None,
) -> str:
    """Download a file over HTTPS with caching, retries, and integrity checks.

    The request goes through the shared pool (see :func:`_http`). The body is
    additionally verified against an MD5 when the server's ``ETag`` is a plain
    32-hex-digit MD5. When *cache_path* already exists, an
    ``If-Modified-Since`` conditional GET keyed on its mtime returns the
    cached copy on a 304.

    Args:
        url: HTTPS URL.
        progress_callback: Optional callback(bytes_downloaded, total_bytes, status).
        cache_path: Optional destination. When given, the file is written here
            atomically and reused on later calls; when omitted it goes to a
            temporary path the caller must clean up.

    Returns:
        Path to the file: ``str(cache_path)`` on both a fresh download and a 304
        cache hit, or a temporary path when ``cache_path`` is None.

    Raises:
        HTTPStatusError: On a non-retryable HTTP status (e.g. 404).
        UnwritableDestination: If *cache_path* cannot be written.
        urllib3.exceptions.MaxRetryError: When retryable failures persist.
        OSError: On a corrupted download (after retries).
    """
    import urllib3

    # Check the destination before spending a request on it.
    if cache_path is not None:
        _preflight_destination(cache_path)

    # The pool already retries the request itself. This loop covers only
    # failures that strike after the headers, where nothing short of
    # restarting the transfer will do.
    attempts = 4
    for attempt in range(attempts):
        try:
            return _download_once(url, progress_callback, cache_path)
        except (
            HTTPStatusError,
            UnwritableDestination,
            urllib3.exceptions.MaxRetryError,
        ):
            raise
        except OSError as e:
            if _unwritable(e):
                raise UnwritableDestination(
                    e.errno, f"Cannot write {cache_path or 'download'}: {e.strerror}"
                ) from e
            if attempt == attempts - 1:
                raise
        except urllib3.exceptions.HTTPError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)


def _download_once(
    url: str,
    progress_callback: Optional[Callable[[int, int, str], None]],
    cache_path: Optional[Path],
) -> str:
    """Single download attempt (see :func:`download_file_to_temp`)."""
    headers = {}
    if cache_path and cache_path.exists():
        headers["If-Modified-Since"] = formatdate(
            cache_path.stat().st_mtime, usegmt=True
        )

    response = _http().request(
        "GET", url, headers=headers or None, preload_content=False
    )
    try:
        if response.status == 304:
            if progress_callback:
                progress_callback(0, 0, "Cache is current")
            return str(cache_path)
        if response.status != 200:
            raise HTTPStatusError(url, response.status, response.reason)
        return _stream_to_file(response, url, cache_path, progress_callback)
    finally:
        # Drain before releasing: an unread body poisons the connection for
        # whoever takes it out of the pool next.
        response.drain_conn()
        response.release_conn()


def _transfer_status(downloaded: int, total: int, elapsed: float) -> str:
    """Format a progress line, omitting the rate until elapsed time is measurable."""
    done = f"{format_bytes(downloaded)}/{format_bytes(total)}"
    if elapsed <= 0:
        return done
    return f"{done} @ {format_bytes(downloaded / elapsed)}/s"


def _stream_to_file(
    response,
    url: str,
    cache_path: Optional[Path],
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> str:
    """Stream *response* to *cache_path*, or to a temporary file, atomically.

    The body is verified against the MD5 ETag where there is one, and the file
    mtime is stamped from ``Last-Modified`` so the next ``If-Modified-Since``
    works. Truncation needs no check: the pool enforces ``Content-Length``, so
    a short body raises out of ``response.read()``. Returns the final path.
    """
    from .remote import atomic_output

    total_size = int(response.headers.get("Content-Length") or 0)

    last_modified = None
    lm_header = response.headers.get("Last-Modified")
    if lm_header:
        try:
            last_modified = parsedate_to_datetime(lm_header)
        except (TypeError, ValueError):
            pass

    # A plain 32-hex-digit ETag is the object's content MD5. Multipart
    # uploads ("-<parts>" suffix) and weak validators ("W/" prefix) are not,
    # and fail the match, so those downloads are only length-checked.
    etag = (response.headers.get("ETag") or "").strip('"')
    md5 = hashlib.md5() if re.fullmatch(r"[0-9a-f]{32}", etag) else None

    suffix = cache_path.suffix if cache_path else ".npy"
    with atomic_output(cache_path, suffix=suffix) as temp_path:
        downloaded = 0
        start_time = time.time()
        last_update_time = start_time

        if progress_callback:
            size_str = format_bytes(total_size) if total_size > 0 else "unknown size"
            progress_callback(0, total_size, f"Starting ({size_str})")

        with open(temp_path, "wb") as temp_file:
            while chunk := response.read(256 * 1024):
                if md5 is not None:
                    md5.update(chunk)
                temp_file.write(chunk)
                downloaded += len(chunk)

                if progress_callback and total_size > 0:
                    now = time.time()
                    # Report every ~100ms, and once more on the last chunk.
                    if now - last_update_time > 0.1 or downloaded == total_size:
                        last_update_time = now
                        progress_callback(
                            downloaded,
                            total_size,
                            _transfer_status(downloaded, total_size, now - start_time),
                        )

        if md5 is not None and md5.hexdigest() != etag:
            raise OSError(
                f"Checksum mismatch for {url}: MD5 {md5.hexdigest()} != ETag {etag}"
            )

        # atomic_output's replace() preserves the mtime set here.
        if last_modified is not None:
            try:
                ts = last_modified.timestamp()
                os.utime(temp_path, (ts, ts))
            except OSError as e:
                logging.getLogger(__name__).warning(f"Could not set file mtime: {e}")

    if progress_callback:
        progress_callback(
            downloaded,
            total_size or downloaded,
            f"Complete ({format_bytes(downloaded)})",
        )

    return str(cache_path if cache_path else temp_path)


class Registry:
    """Registry management for Tessera data files using Parquet.

    Handles all registry-related operations including:
    - Loading and querying Parquet registry
    - Direct HTTP downloads to embeddings_dir for persistent tile storage
    - Parsing available embeddings and landmasks

    Note: The Parquet registry itself (~few MB) is cached separately from tile data.
    Data tiles are downloaded to embeddings_dir and persist for reuse across sessions.
    """

    def __init__(
        self,
        version: str,
        variant: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        registry_url: Optional[str] = None,
        registry_path: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        landmasks_registry_url: Optional[str] = None,
        landmasks_registry_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Registry manager with optimized Parquet registries.

        Args:
            version: Dataset version. Accepts ``"v1"``/``"1.0"``,
                ``"v1.1"``/``"1.1"``, ``"v2"``/``"2.0"``, etc.
            variant: Dataset variant. Defaults to the version's default
                variant (``"vultr"`` for v1, ``"cambridge"`` for v1.1,
                ``"2B-L~beta1"`` for v2 — see ``KNOWN_DATASETS``). The
                (version, variant) pair selects the npy/ tree directory
                the manifest and tiles are fetched from. Reserved
                coming-soon variants (e.g. the v1.1 ``dclimate`` run)
                raise ``ValueError`` until published.
            cache_dir: Optional directory for caching Parquet registries only (not data files)
            embeddings_dir: Directory for storing embedding tiles (defaults to current directory).
                Expected structure: global_0.1_degree_representation[.<variant>]/{year}/
                grid_{lon}_{lat}.npy and _scales.npy, global_0.1_degree_tiff_all/landmask_{lon}_{lat}.tif.
                Tiles are downloaded here and persist for reuse across sessions.
            registry_url: URL to download embeddings manifest parquet from (default: remote)
            registry_path: Local path to existing embeddings manifest parquet file
            registry_dir: Directory containing manifest.parquet and landmasks.parquet files (alternative to individual paths)
            landmasks_registry_url: URL to download landmasks Parquet registry from (default: remote)
            landmasks_registry_path: Local path to existing landmasks Parquet registry file
            logger: Optional logger instance. If not provided, creates a new one
        """
        # Resolve version into a path component and a normalised numeric form.
        self._version_path, self._version_norm = _parse_dataset_version(version)
        self._variant = variant or default_variant(self._version_norm)
        # npy/ tree directory for this (version, variant) dataset, e.g.
        # "v1", "v1.1-cam", "v2-2B-L~beta1". Raises for variants that are
        # reserved but not yet published (e.g. the v1.1 dclimate run).
        self._dataset_path = dataset_path(self._version_norm, self._variant)
        self._embeddings_subdir = _variant_subdir(self._variant)
        # Preserve the original kwarg for callers that still read .version.
        self.version = self._version_path
        # Public read-only view of the variant-aware subdirectory name,
        # recorded in the tessera_metadata.json sidecar.
        self.embeddings_subdir = self._embeddings_subdir
        self.variant = self._variant
        # Populated by _load_registry() with the local path to the manifest
        # parquet (cache file or user-supplied). Useful for tools that need
        # the raw unfiltered manifest (e.g. multi-source coverage rendering).
        self.manifest_path: Optional[Path] = None
        self.logger = logger or logging.getLogger(__name__)

        # Set up cache directory for Parquet registries only
        if cache_dir:
            self._registry_cache_dir = Path(cache_dir)
        else:
            # Use platform-appropriate cache directory
            if os.name == "nt":
                base = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            self._registry_cache_dir = base / "geotessera"

        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up embeddings directory for tile storage (defaults to cwd)
        if embeddings_dir:
            self._embeddings_dir = Path(embeddings_dir)
        else:
            self._embeddings_dir = Path.cwd()
            # Use debug level since this is only relevant when actually fetching tiles
            self.logger.debug(
                f"embeddings_dir not specified, using current directory: {self._embeddings_dir}"
            )

        # Handle registry_dir convenience parameter
        if registry_dir:
            registry_dir_path = Path(registry_dir)
            if not registry_path:
                candidate = registry_dir_path / "manifest.parquet"
                if candidate.exists():
                    registry_path = candidate
            if not landmasks_registry_path:
                candidate = registry_dir_path / "landmasks.parquet"
                if candidate.exists():
                    landmasks_registry_path = candidate

        # Embeddings manifest (GeoDataFrame with spatial index). The wire format
        # mirrors the file-scan inventory schema; geometry is derived from
        # lon/lat at load time if the parquet was written as a plain DataFrame.
        # Manifests are per dataset (one npy/ directory per (version,
        # variant) pair); the consumer fetches the manifest matching its
        # dataset directory and filters by variant on load.
        self._registry_gdf: Optional[gpd.GeoDataFrame] = None
        self._registry_url = registry_url or manifest_url(self._dataset_path)
        self._registry_path = Path(registry_path) if registry_path else None

        # Landmasks Parquet registry (per version).
        self._landmasks_df: Optional[pd.DataFrame] = None
        self._landmasks_registry_url = landmasks_registry_url or landmasks_parquet_url(
            self._version_path
        )
        self._landmasks_registry_path = (
            Path(landmasks_registry_path) if landmasks_registry_path else None
        )

        # Memoizes validate_embeddings_dir() so per-tile fetches don't
        # re-read the sidecar.
        self._embeddings_dir_validated = False

        # Load registries
        self._load_registry()
        self._load_landmasks_registry()

    def _load_registry(self):
        """Load manifest as GeoDataFrame with If-Modified-Since refresh.

        The manifest is a plain parquet using the file-scan inventory schema
        (year, lon, lat, grid_size, scales_size, ...). A Point geometry column
        is materialised from lon/lat at load time so spatial queries via
        GeoPandas's R-tree still work.
        """
        registry_path = None

        if self._registry_path and self._registry_path.exists():
            # Load from local file (no updates check for explicit paths)
            self.logger.info(f"Loading manifest from local file: {self._registry_path}")
            registry_path = self._registry_path
        else:
            # Use cached version with If-Modified-Since to check for updates.
            # Cache layout mirrors the npy/ tree so multiple datasets coexist.
            registry_cache_path = (
                self._registry_cache_dir / self._dataset_path / "manifest.parquet"
            )
            registry_cache_path.parent.mkdir(parents=True, exist_ok=True)

            if registry_cache_path.exists():
                self.logger.info(f"Using cached manifest: {registry_cache_path}")
                # download_file_to_temp returns str(cache_path) on both a 304
                # (unchanged) and a 200 (fresh download), so compare the file
                # mtime pre/post to tell which happened: a fresh download
                # rewrites it (mtime set from the new Last-Modified), a 304
                # leaves it untouched.
                pre_mtime = registry_cache_path.stat().st_mtime
                try:
                    self.logger.info("Checking for manifest updates...")
                    result_path = download_file_to_temp(
                        self._registry_url, cache_path=registry_cache_path
                    )
                    registry_path = Path(result_path)
                    if registry_cache_path.stat().st_mtime == pre_mtime:
                        self.logger.info(
                            "Verified with server - manifest is current (no download needed)"
                        )
                    else:
                        self.logger.info("Downloaded updated manifest from server")
                except Exception as e:
                    self.logger.warning(f"Could not check for updates: {e}")
                    self.logger.info("Using existing cached manifest")
                    registry_path = registry_cache_path
            else:
                # Download the manifest to cache for the first time
                self.logger.info(f"Downloading manifest from {self._registry_url}")
                try:
                    result_path = download_file_to_temp(
                        self._registry_url, cache_path=registry_cache_path
                    )
                    registry_path = Path(result_path)
                    self.logger.info("Manifest downloaded successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to download manifest: {e}") from e

        # Load as plain parquet first; promote to GeoDataFrame if needed.
        try:
            df = pd.read_parquet(registry_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load manifest parquet: {e}") from e

        # Record the on-disk manifest path for tools that need the unfiltered
        # parquet (multi-source coverage rendering, manifest introspection, …).
        self.manifest_path = Path(registry_path)

        self.logger.info(f"Loaded manifest with {len(df):,} tiles")

        # Validate required columns (file-scan inventory schema).
        required_columns = {"lat", "lon", "year", "grid_size"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Manifest is missing required columns: {missing}")

        # Filter to the requested (version, variant) if those columns are present.
        # Missing columns mean a single-(version, variant) manifest; trust it as-is.
        before = len(df)
        if "version" in df.columns:
            df = df[df["version"].astype(str) == self._version_norm]
        if "variant" in df.columns:
            df = df[df["variant"].astype(str) == self._variant]
        if len(df) != before:
            self.logger.info(
                f"Filtered manifest to version={self._version_norm}, "
                f"variant={self._variant}: {len(df):,} tiles (from {before:,})"
            )
        if df.empty:
            raise ValueError(
                f"Manifest has no rows for version={self._version_norm}, "
                f"variant={self._variant}. Check the dataset_version and "
                f"dataset_variant arguments."
            )

        # Defensive dedupe: bucket-side misfiling (e.g. a tile filed under the
        # wrong grid directory) can cause the scanner to emit two rows for the
        # same (year, lon, lat). Duplicates here would make the MultiIndex
        # non-unique and turn .loc[] lookups into multi-row Series, which
        # downstream int(row["grid_size"]) cannot handle.
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=["year", "lon", "lat"], keep="first")
        if len(df) != before_dedupe:
            self.logger.warning(
                f"Dropped {before_dedupe - len(df):,} duplicate (year, lon, lat) "
                "rows from manifest"
            )

        # Materialise geometry from lon/lat for spatial queries (.cx).
        if "geometry" in df.columns:
            self._registry_gdf = gpd.GeoDataFrame(
                df, geometry="geometry", crs="EPSG:4326"
            )
        else:
            geometry = gpd.points_from_xy(df["lon"], df["lat"])
            self._registry_gdf = gpd.GeoDataFrame(
                df, geometry=geometry, crs="EPSG:4326"
            )

        # Ensure lon_i/lat_i columns exist (backwards compat with old parquet files)
        if "lon_i" not in self._registry_gdf.columns:
            self._registry_gdf["lon_i"] = (
                (self._registry_gdf["lon"] * 100).round().astype(np.int32)
            )
        if "lat_i" not in self._registry_gdf.columns:
            self._registry_gdf["lat_i"] = (
                (self._registry_gdf["lat"] * 100).round().astype(np.int32)
            )

        # Set MultiIndex for O(1) lookups via .loc[(year, lon_i, lat_i)]
        self._registry_gdf["year"] = self._registry_gdf["year"].astype(int)
        self._registry_gdf = self._registry_gdf.set_index(["year", "lon_i", "lat_i"])

    def _load_landmasks_registry(self):
        """Load landmasks Parquet registry from local path or download from remote with If-Modified-Since refresh."""
        if self._landmasks_registry_path and self._landmasks_registry_path.exists():
            # Load from local file (no updates check for explicit paths)
            self.logger.info(
                f"Loading landmasks registry from local file: {self._landmasks_registry_path}"
            )
            self._landmasks_df = pd.read_parquet(self._landmasks_registry_path)
        else:
            # Use cached version with If-Modified-Since to check for updates.
            # Cache layout mirrors S3 so multiple versions coexist.
            landmasks_cache_path = (
                self._registry_cache_dir / self._version_path / "landmasks.parquet"
            )
            landmasks_cache_path.parent.mkdir(parents=True, exist_ok=True)

            if landmasks_cache_path.exists():
                self.logger.info(
                    f"Using cached landmasks registry: {landmasks_cache_path}"
                )
                # Compare the file mtime pre/post to tell a 304 (unchanged,
                # file untouched) from a 200 (fresh download, mtime updated
                # from the new Last-Modified).
                pre_mtime = landmasks_cache_path.stat().st_mtime
                try:
                    self.logger.info("Checking for landmasks registry updates...")
                    result_path = download_file_to_temp(
                        self._landmasks_registry_url, cache_path=landmasks_cache_path
                    )
                    landmasks_path = Path(result_path)
                    self._landmasks_df = pd.read_parquet(landmasks_path)
                    if landmasks_cache_path.stat().st_mtime == pre_mtime:
                        self.logger.info(
                            "Verified with server - landmasks registry is current (no download needed)"
                        )
                    else:
                        self.logger.info(
                            "Downloaded updated landmasks registry from server"
                        )
                except Exception as e:
                    # Landmasks are optional, if update check fails use cached version
                    self.logger.warning(f"Could not check for landmasks updates: {e}")
                    self.logger.info("Using existing cached landmasks registry")
                    try:
                        self._landmasks_df = pd.read_parquet(landmasks_cache_path)
                    except (OSError, ValueError, ImportError) as read_error:
                        # OSError: File not readable, ValueError: Invalid parquet, ImportError: Missing deps
                        self.logger.warning(
                            f"Could not read cached landmasks: {read_error}"
                        )
                        self._landmasks_df = None
                        return
            else:
                # Download the landmasks registry to cache for the first time
                self.logger.info(
                    f"Downloading landmasks registry from {self._landmasks_registry_url}"
                )
                try:
                    # Download landmasks registry with caching
                    result_path = download_file_to_temp(
                        self._landmasks_registry_url, cache_path=landmasks_cache_path
                    )
                    landmasks_path = Path(result_path)
                    self._landmasks_df = pd.read_parquet(landmasks_path)
                    self.logger.info("Landmasks registry downloaded successfully")
                except Exception as e:
                    # Landmasks are optional, so just warn instead of failing
                    self.logger.warning(f"Failed to download landmasks registry: {e}")
                    self._landmasks_df = None
                    return

        # Validate landmasks registry structure. Hash columns are not
        # required because integrity is verified against Content-Length and
        # the MD5 ETag at download time.
        if self._landmasks_df is not None:
            required_columns = {"lat", "lon", "file_size"}
            if not required_columns.issubset(self._landmasks_df.columns):
                missing = required_columns - set(self._landmasks_df.columns)
                self.logger.warning(
                    f"Landmasks registry is missing required columns: {missing}"
                )
                self._landmasks_df = None
            else:
                # Ensure lon_i/lat_i columns exist (backwards compat with old parquet files)
                if "lon_i" not in self._landmasks_df.columns:
                    self._landmasks_df["lon_i"] = (
                        (self._landmasks_df["lon"] * 100).round().astype(np.int32)
                    )
                if "lat_i" not in self._landmasks_df.columns:
                    self._landmasks_df["lat_i"] = (
                        (self._landmasks_df["lat"] * 100).round().astype(np.int32)
                    )

                # Set index for O(1) lookups via .loc[(lon_i, lat_i)]
                self._landmasks_df = self._landmasks_df.set_index(["lon_i", "lat_i"])

    def _lookup_tile(self, year: int, lon: float, lat: float) -> pd.Series:
        """Look up a tile row by year and coordinates.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            pd.Series with the tile's registry data

        Raises:
            ValueError: If tile not found in registry
        """
        lon_i = int(coord_to_grid_int(lon))
        lat_i = int(coord_to_grid_int(lat))
        try:
            return self._registry_gdf.loc[(int(year), lon_i, lat_i)]
        except KeyError:
            raise ValueError(
                f"Tile not found in registry: year={year}, lon={lon:.2f}, lat={lat:.2f}"
            )

    def _lookup_landmask(self, lon: float, lat: float) -> pd.Series:
        """Look up a landmask row by coordinates.

        Args:
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            pd.Series with the landmask's registry data

        Raises:
            ValueError: If landmask not found in registry
        """
        lon_i = int(coord_to_grid_int(lon))
        lat_i = int(coord_to_grid_int(lat))
        try:
            return self._landmasks_df.loc[(lon_i, lat_i)]
        except KeyError:
            raise ValueError(
                f"Landmask not found in registry: lon={lon:.2f}, lat={lat:.2f}"
            )

    def iter_tiles_in_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ) -> Iterator[Tuple[int, float, float]]:
        """Lazy iterator over tiles in a region using GeoPandas spatial indexing.

        GeoPandas automatically uses R-tree spatial indexing for fast queries.
        This method:
        - Starts yielding immediately (low latency)
        - Uses constant memory regardless of region size
        - Allows early termination without processing all tiles
        - Leverages GeoPandas built-in R-tree for optimal performance

        Note: Registry stores tiles as Point geometries at their centers, but tiles
        are 0.1° x 0.1° boxes. The query is expanded by 0.05° (half tile width) in
        all directions to catch tiles whose boxes intersect the bounds but whose
        centers fall outside.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load

        Yields:
            Tuples of (year, tile_lon, tile_lat) for each tile in the region

        Example:
            >>> registry = Registry('v1')
            >>> bounds = (-0.2, 51.4, 0.1, 51.6)  # London
            >>> for year, lon, lat in registry.iter_tiles_in_region(bounds, 2024):
            ...     embedding = fetch_embedding(lon, lat, year)
            ...     process(embedding)  # Start processing immediately
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Expand bounds by half a tile width (0.05°) to catch tiles whose boxes
        # intersect the query region. Registry stores tiles as Point geometries at
        # their centers, but tiles are 0.1° x 0.1° boxes, so we need to expand the
        # query to include tiles whose centers are up to 0.05° outside the bounds.
        expansion = 0.05
        tiles = self._registry_gdf.cx[
            min_lon - expansion : max_lon + expansion,
            min_lat - expansion : max_lat + expansion,
        ]

        # Filter by year using index level
        tiles = tiles[tiles.index.get_level_values("year") == year]

        # Yield unique (year, lon_i, lat_i) tuples from the index
        seen = set()
        for idx in tiles.index:
            if idx not in seen:
                seen.add(idx)
                year_val, lon_i, lat_i = idx
                yield (year_val, lon_i / 100.0, lat_i / 100.0)

    def load_blocks_for_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ) -> List[Tuple[int, float, float]]:
        """Load tiles for a region (list-returning version for backward compatibility).

        For memory-efficient streaming, use iter_tiles_in_region() instead.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load

        Returns:
            List of (year, tile_lon, tile_lat) tuples for tiles in the region
        """
        # Use iterator and materialize to list (vectorized, 10-100x faster than iterrows)
        tiles_list = list(self.iter_tiles_in_region(bounds, year))

        if tiles_list:
            self.logger.info(f"Found {len(tiles_list)} tiles for region in year {year}")

        return tiles_list

    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.

        Returns:
            List of years with available data, sorted in ascending order.
        """
        return sorted(
            self._registry_gdf.index.get_level_values("year").unique().tolist()
        )

    def get_tile_counts_by_year(self) -> Dict[int, int]:
        """Get count of tiles per year using efficient pandas operations.

        Returns:
            Dictionary mapping year to tile count
        """
        # Count unique (lon_i, lat_i) index pairs per year level
        idx = self._registry_gdf.index
        counts = (
            pd.DataFrame(
                {
                    "year": idx.get_level_values("year"),
                    "lon_i": idx.get_level_values("lon_i"),
                    "lat_i": idx.get_level_values("lat_i"),
                }
            )
            .drop_duplicates()
            .groupby("year")
            .size()
            .to_dict()
        )
        return {int(year): int(count) for year, count in counts.items()}

    def get_available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of all available embeddings with vectorized conversion.

        Returns:
            List of (year, lon, lat) tuples for all available embedding tiles
        """
        # Use unique index tuples directly - already (year, lon_i, lat_i)
        unique_idx = self._registry_gdf.index.unique()
        return [
            (int(year), lon_i / 100.0, lat_i / 100.0)
            for year, lon_i, lat_i in unique_idx
        ]

    def _sidecar_dataset_path(self) -> Optional[str]:
        """Dataset directory recorded in the embeddings_dir sidecar, if any."""
        import json

        sidecar = self._embeddings_dir / TESSERA_METADATA_FILENAME
        try:
            data = json.loads(sidecar.read_text())
            version = data.get("dataset_version_path") or data.get("dataset_version")
            variant = data.get("dataset_variant")
            if not (version and variant):
                return None
            return dataset_path(_parse_dataset_version(str(version))[1], str(variant))
        except (OSError, ValueError):
            return None

    def validate_embeddings_dir(self) -> None:
        """Raise ``ValueError`` if embeddings_dir holds a different dataset.

        Every dataset uses the same local file layout; the
        ``tessera_metadata.json`` sidecar records which one populated the
        directory.
        """
        if self._embeddings_dir_validated:
            return
        recorded = self._sidecar_dataset_path()
        if recorded is not None and recorded != self._dataset_path:
            raise ValueError(
                f"{self._embeddings_dir} holds tiles from dataset "
                f"'{recorded}', but '{self._dataset_path}' was requested. "
                f"Datasets share one local layout, so mixing them returns "
                f"wrong embeddings; use a separate embeddings_dir per dataset."
            )
        self._embeddings_dir_validated = True

    def _record_embeddings_dir_dataset(self) -> None:
        """Write the provenance sidecar after downloading into embeddings_dir."""
        if (self._embeddings_dir / TESSERA_METADATA_FILENAME).exists():
            return
        try:
            write_tessera_metadata(
                self._embeddings_dir, self._version_path, self._variant
            )
        except OSError as e:
            self.logger.warning(f"Could not write {TESSERA_METADATA_FILENAME}: {e}")

    def fetch(
        self,
        path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        refresh: bool = False,
        year: Optional[int] = None,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        is_scales: bool = False,
    ) -> str:
        """Fetch a file using local embeddings_dir or direct HTTP download.

        Args:
            path: Optional path to the file (relative to base URL or embeddings_dir).
                  If not provided, will be calculated from year/lon/lat.
            progress_callback: Optional callback for progress updates
            refresh: If True, force re-download even if local file exists
            year: Year of the tile (required if path not provided)
            lon: Longitude of the tile (required if path not provided)
            lat: Latitude of the tile (required if path not provided)
            is_scales: If True, fetch scales file instead of embedding file

        Returns:
            Path to the file in embeddings_dir
        """
        # Calculate path from coordinates if not provided
        if path is None:
            if year is None or lon is None or lat is None:
                raise ValueError(
                    "Must provide either 'path' or all of (year, lon, lat)"
                )
            embedding_path, scales_path = tile_to_embedding_paths(lon, lat, year)
            path = scales_path if is_scales else embedding_path

        # Local layout always uses the bare ``global_0.1_degree_representation``
        # subdir regardless of variant; variant/version provenance lives in
        # the ``tessera_metadata.json`` sidecar.
        local_path = self._embeddings_dir / EMBEDDINGS_DIR_NAME / path

        self.validate_embeddings_dir()

        # Check if file exists locally and not refreshing
        if local_path.exists() and not refresh:
            # Use existing local file
            return str(local_path)

        # Download to embeddings_dir. Use as_posix() so the URL uses forward
        # slashes on Windows.
        path_str = path.as_posix() if isinstance(path, Path) else path
        url = embedding_url(self._dataset_path, path_str)
        result = download_file_to_temp(
            url,
            progress_callback=progress_callback,
            cache_path=local_path,
        )
        self._record_embeddings_dir_dataset()
        return result

    def fetch_landmask(
        self,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        refresh: bool = False,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
    ) -> str:
        """Fetch a landmask file using local embeddings_dir or direct HTTP download.

        Args:
            filename: Optional name of the landmask file. If not provided, will be
                      calculated from lon/lat.
            progress_callback: Optional callback for progress updates
            refresh: If True, force re-download even if local file exists
            lon: Longitude of the tile (required if filename not provided)
            lat: Latitude of the tile (required if filename not provided)

        Returns:
            Path to the file in embeddings_dir
        """
        # Calculate filename from coordinates if not provided
        if filename is None:
            if lon is None or lat is None:
                raise ValueError("Must provide either 'filename' or both (lon, lat)")
            filename = tile_to_landmask_filename(lon, lat)

        # Determine local file path
        local_path = self._embeddings_dir / LANDMASKS_DIR_NAME / filename

        self.validate_embeddings_dir()

        # Check if file exists locally and not refreshing
        if local_path.exists() and not refresh:
            # Use existing local file
            return str(local_path)

        # Download to embeddings_dir.
        url = landmask_url(self._version_path, filename)
        result = download_file_to_temp(
            url,
            progress_callback=progress_callback,
            cache_path=local_path,
        )
        self._record_embeddings_dir_dataset()
        return result

    @property
    def available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of available embeddings."""
        return self.get_available_embeddings()

    def get_landmask_count(self) -> int:
        """Get count of unique landmask tiles using efficient pandas operations.

        Returns:
            Count of unique landmask tiles
        """
        if self._landmasks_df is not None:
            # Count unique (lon_i, lat_i) index pairs in landmasks registry
            return len(self._landmasks_df.index.unique())

        # Fallback: count unique tiles in embeddings registry
        idx = self._registry_gdf.index.droplevel("year")
        return len(idx.unique())

    @property
    def available_landmasks(self) -> List[Tuple[float, float]]:
        """Get list of available landmasks with vectorized conversion.

        Falls back to embedding tiles if landmasks registry is not available.

        Note: For performance, use get_landmask_count() if you only need the count.
        """
        # Use landmasks registry if available
        if self._landmasks_df is not None:
            unique_idx = self._landmasks_df.index.unique()
            return [(lon_i / 100.0, lat_i / 100.0) for lon_i, lat_i in unique_idx]

        # Fallback: assume landmasks are available for all embedding tiles
        unique_idx = self._registry_gdf.index.droplevel("year").unique()
        return [(lon_i / 100.0, lat_i / 100.0) for lon_i, lat_i in unique_idx]

    def get_manifest_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get manifest information (git hash and repo URL).

        For Parquet registries, this information is not stored in the registry.
        Returns empty values for API compatibility.

        Returns:
            Tuple of (git_hash, repo_url) - both None for Parquet registries
        """
        return None, None

    def get_tile_file_size(self, year: int, lon: float, lat: float) -> int:
        """Get the file size of an embedding tile from the registry.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If tile not found in registry or file_size column missing
        """
        if "grid_size" not in self._registry_gdf.columns:
            raise ValueError(
                "Manifest is missing 'grid_size' column. "
                "Please update your manifest to include file size metadata."
            )

        row = self._lookup_tile(year, lon, lat)
        return int(row["grid_size"])

    def get_scales_file_size(self, year: int, lon: float, lat: float) -> int:
        """Get the file size of a scales file from the registry.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If tile not found in registry or scales_size column missing
        """
        if "scales_size" not in self._registry_gdf.columns:
            raise ValueError(
                "Registry is missing 'scales_size' column. "
                "Please update your registry to include scales file size metadata."
            )

        row = self._lookup_tile(year, lon, lat)
        return int(row["scales_size"])

    def get_landmask_file_size(self, lon: float, lat: float) -> int:
        """Get the file size of a landmask tile from the registry.

        Args:
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If landmask not found in registry or file_size column missing
        """
        if self._landmasks_df is None:
            raise ValueError(
                "Landmasks registry is not loaded. "
                "Please ensure landmasks.parquet is available."
            )

        if "file_size" not in self._landmasks_df.columns:
            raise ValueError(
                "Landmasks registry is missing 'file_size' column. "
                "Please update your landmasks registry to include file size metadata."
            )

        row = self._lookup_landmask(lon, lat)
        return int(row["file_size"])

    def calculate_download_requirements(
        self,
        tiles: List[Tuple[int, float, float]],
        output_dir: Path,
        format_type: str,
        check_existing: bool = True,
    ) -> Tuple[int, int, Dict[str, int]]:
        """Calculate download requirements for a set of tiles.

        Args:
            tiles: List of (year, lon, lat) tuples
            output_dir: Output directory where files would be downloaded
            format_type: Either 'npy' or 'tiff'
            check_existing: If True, skip files that already exist (for resume).
                           If False, calculate as if downloading all files (for dry-run estimates).

        Returns:
            Tuple of (total_bytes, total_files, file_sizes_dict)
            - total_bytes: Total download size in bytes
            - total_files: Number of files to download
            - file_sizes_dict: Dictionary mapping file keys to sizes (for NPY format tracking)

        Raises:
            ValueError: If registry is missing required columns or tiles not found
        """
        total_bytes = 0
        total_files = 0
        file_sizes = {}  # For NPY format: cache file sizes by key

        if format_type == "npy":
            # For NPY format: embedding + scales + landmask per tile
            for tile_year, tile_lon, tile_lat in tiles:
                # Use tile_to_embedding_paths for correct directory structure
                embedding_rel, scales_rel = tile_to_embedding_paths(
                    tile_lon, tile_lat, tile_year
                )
                embedding_final = output_dir / EMBEDDINGS_DIR_NAME / embedding_rel
                scales_final = output_dir / EMBEDDINGS_DIR_NAME / scales_rel
                landmask_final = (
                    output_dir
                    / LANDMASKS_DIR_NAME
                    / tile_to_landmask_filename(tile_lon, tile_lat)
                )

                # Create cache keys for tracking file sizes
                embedding_key = f"embedding_{tile_year}_{tile_lon}_{tile_lat}"
                scales_key = f"scales_{tile_year}_{tile_lon}_{tile_lat}"
                landmask_key = f"landmask_{tile_lon}_{tile_lat}"

                # Only count files that need downloading
                if not check_existing or not embedding_final.exists():
                    size = self.get_tile_file_size(tile_year, tile_lon, tile_lat)
                    file_sizes[embedding_key] = size
                    total_bytes += size
                    total_files += 1

                if not check_existing or not scales_final.exists():
                    # Get actual scales file size from registry
                    size = self.get_scales_file_size(tile_year, tile_lon, tile_lat)
                    file_sizes[scales_key] = size
                    total_bytes += size
                    total_files += 1

                if not check_existing or not landmask_final.exists():
                    size = self.get_landmask_file_size(tile_lon, tile_lat)
                    file_sizes[landmask_key] = size
                    total_bytes += size
                    total_files += 1
        else:
            # For TIFF format: one GeoTIFF per tile
            # TIFF files will be larger than NPY due to dequantization (int8 -> float32)
            # and additional metadata. Estimate as 4x the size of quantized embedding.
            for tile_year, tile_lon, tile_lat in tiles:
                embedding_size = self.get_tile_file_size(tile_year, tile_lon, tile_lat)
                landmask_size = self.get_landmask_file_size(tile_lon, tile_lat)
                # Estimate TIFF size: 4x embedding (float32 vs int8) + landmask overhead
                tiff_size = (embedding_size * 4) + landmask_size
                total_bytes += tiff_size
                total_files += 1

        return total_bytes, total_files, file_sizes
