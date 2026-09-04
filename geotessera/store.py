"""
GeoTesseraZarr — read embeddings from a Tessera zarr store.

The store is UTM-native: embeddings live under one ``utm{NN}`` group per UTM
zone, on the grid they were produced on.  Nothing here reprojects pixels, and
the two layers each speak one coordinate system:

``GeoTesseraZarr``
    Takes **longitude and latitude**, routes each query to the ``utm{NN}``
    group that holds it, and hands back that zone's native UTM pixels.  This
    is the layer to use unless you already know your zone.

``.tessera`` accessor
    Takes **eastings and northings in that zone's own CRS**.  Once a zone is
    open there is nothing left to route, so no projection happens at all.

Coordinates crossing between the two are projected — one cheap point
transform — but the embeddings themselves are always returned on their
native grid, never resampled.

Usage::

    from geotessera.store import GeoTesseraZarr

    gt = GeoTesseraZarr()  # default public store
    X = gt.sample_points([(-2.97, 53.44), (-2.96, 53.43)], year=2025)
    mosaic, transform, crs = gt.read_region(bbox, year=2025)  # bbox in lon/lat

    # A fixed-size patch centred on a point, merged across UTM zones
    patch, transform, crs = gt.read_patch(0.0, 52.2, year=2025, size_px=512)

    # On v2 stores, depth=16 reads the matryoshka prefix for 1/8 the bytes
    X16 = gt.sample_points([(-2.97, 53.44)], year=2025, depth=16)

    # Direct zone access, in that zone's UTM
    ds = gt.open_zone(lon=-2.97)
    emb = ds.tessera.sample_at(500_000.0, 5_921_000.0, year=2025)

Working from another CRS?  Project your points to lon/lat once, up front,
rather than per call.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import rasterio.transform
import xarray as xr
import zarr
from obstore.store import HTTPStore
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from zarr.abc.store import Store as ZarrStore
from zarr.storage import ObjectStore

from .registry import zarr_store_url

log = logging.getLogger(__name__)

DEFAULT_STORE = zarr_store_url("v1")

# Shard-aligned chunk sizes so dask tasks match zarr shards
SHARD_CHUNKS = {"time": 1, "band": 128, "y": 4096, "x": 4096}

# obstore retries each request with exponential backoff and jitter, so
# one dropped response from a busy server costs a chunk, not the read.
RETRY_CONFIG = {
    "max_retries": 5,
    "backoff": {
        "init_backoff": timedelta(seconds=1),
        "max_backoff": timedelta(seconds=30),
        "base": 2,
    },
    "retry_timeout": timedelta(minutes=5),
}

CLIENT_OPTIONS = {"timeout": timedelta(seconds=120)}

# Chunk fetches in flight for reads through zarr's own pipeline.  Remote
# reads are latency-bound, so more than zarr's default of 10 helps; kept
# scoped, since dask paths multiply it by their thread count.
POINT_CONCURRENCY = 32


def _store_cache_key(location: str) -> str:
    """Per-store cache subdirectory name for *location*.

    Cache entries are keyed by store-relative paths, so each store must
    cache in its own subdirectory.  A public store URL keys by its
    dataset path (``v1``, ``v2-2B-L_beta1``); any other location keys
    by a slug of the location plus a digest, since the same dataset
    name at two locations can hold different bytes.
    """
    import hashlib
    import re

    from .registry import TESSERA_MIRROR_URL

    location = location.rstrip("/")
    canonical_prefix = f"{TESSERA_MIRROR_URL}/zarr/"
    if location.startswith(canonical_prefix):
        dataset = location[len(canonical_prefix) :]
        if dataset and "/" not in dataset:
            return re.sub(r"[^A-Za-z0-9._-]+", "_", dataset)
    digest = hashlib.sha256(location.encode()).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", location.split("://")[-1]).strip("_")
    return f"{slug[-80:]}-{digest}"


def zarr_store(
    location,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_max_size: Optional[int] = None,
) -> ZarrStore:
    """Open *location* as a ``zarr.abc.store.Store``.

    *location* may be the URL of a Tessera zarr store, a local path, or
    an existing ``Store``, which is returned unchanged.  Stores opened
    from an http(s) URL retry failed requests with exponential backoff,
    because a region read issues hundreds of requests and public data
    servers drop some under load.  ``s3://`` and other URL schemes open
    through fsspec.

    Pass *cache_dir* to persist reads locally through zarr's
    experimental ``CacheStore`` (requires ``zarr>=3.3``)::

        store = zarr_store(DEFAULT_STORE, cache_dir="tessera-cache")

    Each store location caches under its own subdirectory of
    *cache_dir*, so stores never share objects.  Metadata persists
    across runs and chunk data for the session.  *cache_max_size*
    bounds the cache in bytes (default unbounded).  *cache_dir*
    requires a URL or path location; wrap an existing ``Store`` in
    ``CacheStore`` yourself.
    """
    if isinstance(location, ZarrStore):
        if cache_dir is not None:
            raise ValueError(
                "cache_dir requires a URL or path location; wrap an "
                "existing Store in zarr's CacheStore yourself"
            )
        return location
    location = location.rstrip("/")
    if location.startswith(("http://", "https://")):
        http = HTTPStore.from_url(
            location, retry_config=RETRY_CONFIG, client_options=CLIENT_OPTIONS
        )
        store = ObjectStore(http, read_only=True)
    elif "://" in location:
        from zarr.storage import FsspecStore

        store = FsspecStore.from_url(location)
    else:
        from zarr.storage import LocalStore

        store = LocalStore(location)

    if cache_dir is not None:
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import LocalStore

        keyed = Path(cache_dir) / _store_cache_key(location)
        keyed.mkdir(parents=True, exist_ok=True)
        store = CacheStore(
            store, cache_store=LocalStore(keyed), max_size=cache_max_size
        )
    return store


def enable_http_logging(level: int = logging.DEBUG) -> None:
    """Enable fsspec HTTP request logging for debugging.

    Call before opening a store to see every HTTP request::

        from geotessera.store import enable_http_logging
        enable_http_logging()
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fsspec.http").setLevel(level)
    log.setLevel(level)


@lru_cache(maxsize=None)
def _transformer(src_crs: str, dst_crs: str) -> Transformer:
    """A pyproj transformer for this CRS pair, built once and reused.

    Building one costs milliseconds — enough that constructing a fresh
    transformer per point made ``sample_points`` spend most of its time in
    PROJ setup rather than reading data.  Transformers are not thread-safe,
    so a threaded caller needs its own.
    """
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def _project(x: float, y: float, src_crs: str, dst_crs: str):
    """Project ``(x, y)`` between two CRS, a no-op when they match."""
    if src_crs == dst_crs:
        return x, y
    return _transformer(src_crs, dst_crs).transform(x, y)


def _zone_for_lon(lon: float) -> int:
    """UTM zone number (1-60) for a WGS84 longitude."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


# Tiles are 0.1 degrees and UTM zones 6 degrees, so round coordinates land on
# tile edges and multiples of 6 land on zone seams as well. Both leave gaps: a
# tile edge can be one unwritten pixel wide, and a zone's tiles stop at its
# boundary although its grid runs past it.

SEAM_SEARCH_PX = 1  # nearest-valid-pixel radius

# How close to a seam before the neighbouring zone is worth trying. A tile is
# 0.1 degrees and belongs to the zone containing its centre, so a point can
# only be covered by the zone next door if it is within half a tile — 0.05
# degrees — of the boundary. This is that bound doubled, which absorbs the
# sub-pixel spread of a tile's curved UTM footprint and still consults a
# neighbour only where one can actually hold the data.
SEAM_DEGREES = 0.1

VALID, WATER, NODATA, OUTSIDE = "valid", "water", "nodata", "outside"


def _prefetched(load, tops):
    """Yield ``load(t)`` for each t, loading one step ahead on a thread."""
    from concurrent.futures import ThreadPoolExecutor

    tops = list(tops)
    if not tops:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load, tops[0])
        for i in range(len(tops)):
            ready = pending.result()
            if i + 1 < len(tops):
                pending = pool.submit(load, tops[i + 1])
            yield ready


def _progress_iter(items, label: str, total: Optional[int] = None):
    """Yield *items*, logging progress at INFO every few seconds.

    Short runs stay silent.  Pass ``total`` when *items* is lazy and must
    not be materialised.
    """
    if total is None:
        items = list(items)
        total = len(items)
    if total == 0:
        yield from items
        return
    started = time.monotonic()
    next_report = started + 10.0
    reported = False
    for i, item in enumerate(items, 1):
        yield item
        now = time.monotonic()
        if now >= next_report or (reported and i == total):
            reported = True
            next_report = now + 10.0
            elapsed = now - started
            log.info(
                "%s: %d/%d (%.0f%%, %.1f/s)",
                label,
                i,
                total,
                100.0 * i / total,
                i / elapsed if elapsed > 0 else 0.0,
            )


# Causes a bulk read can assign to a point; only some deserve a retry
# through the per-point path.
_OK, _WATER, _HOLE, _OUT = 0, 1, 2, 3


def _nearest_indices(ascending: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the nearest value in a sorted array, for each target."""
    i = np.clip(np.searchsorted(ascending, targets), 1, len(ascending) - 1)
    left, right = ascending[i - 1], ascending[i]
    return np.where(targets - left <= right - targets, i - 1, i)


def _bulk_sample(
    ds: xr.Dataset,
    es: np.ndarray,
    ns: np.ndarray,
    year: int,
    read=None,
    array: str = "embeddings",
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel embeddings for arrays of zone-CRS points, in one read.

    Returns ``(values, cause)``: ``(N, B)`` float32, NaN wherever cause
    is not ``_OK``.  No repair happens here; callers retry those rows
    per point.  ``read(xi, yi)`` fetches ``(embeddings, scales)``; the
    default reads *array* through the Dataset.
    """
    acc = ds.tessera
    es = np.asarray(es, dtype=float)
    ns = np.asarray(ns, dtype=float)
    xname, yname = ("xc", "yc") if "xc" in ds.coords else ("x", "y")
    xs, ys = ds.coords[xname].values, ds.coords[yname].values
    px = acc.pixel_size
    bands = int(ds[array].shape[1])

    values = np.full((len(es), bands), np.nan, np.float32)
    cause = np.full(len(es), _OUT, np.int8)
    inside = (
        (es >= xs[0] - px)
        & (es <= xs[-1] + px)
        & (ns <= ys[0] + px)
        & (ns >= ys[-1] - px)
    )
    if not inside.any():
        return values, cause

    xi = _nearest_indices(xs, es[inside])
    yi = len(ys) - 1 - _nearest_indices(ys[::-1], ns[inside])
    if read is None:

        def read(xi, yi):
            sel = ds.sel(time=year).isel(
                {
                    xname: xr.DataArray(xi, dims="points"),
                    yname: xr.DataArray(yi, dims="points"),
                }
            )
            return sel[array].values.T, sel["scales"].values

    emb, scales = read(xi, yi)
    emb = emb.astype(np.float32) * np.where(
        np.isfinite(scales), scales, np.nan
    )[:, None]
    values[inside] = emb
    cause[inside] = np.where(
        np.isnan(scales), _WATER, np.where(np.isinf(scales), _HOLE, _OK)
    )
    return values, cause


def _resolve_zone(
    zone: Optional[int],
    lon: Optional[float],
    bbox: Optional[Tuple[float, float, float, float]],
) -> int:
    """UTM zone from exactly one of ``zone``, ``lon`` or ``bbox``.

    Accepts a plain ``int`` longitude as readily as a ``float``; the old
    structural match rejected ``lon=-3`` as if no argument had been given.
    """
    given = [
        name
        for name, value in (("zone", zone), ("lon", lon), ("bbox", bbox))
        if value is not None
    ]
    if len(given) != 1:
        raise TypeError(
            "Provide exactly one of zone=, lon=, or bbox="
            + (f" (got {', '.join(given)})" if given else "")
        )
    if zone is not None:
        return int(zone)
    if lon is not None:
        return _zone_for_lon(float(lon))
    return _zone_for_lon((bbox[0] + bbox[2]) / 2)


def _seam_neighbours(lon: float) -> List[int]:
    """Zones to try after the one containing *lon*.  Empty away from a seam."""
    z = _zone_for_lon(lon)
    frac = (lon + 180.0) % 6.0
    out: List[int] = []
    if frac <= SEAM_DEGREES:
        out.append(60 if z == 1 else z - 1)
    if frac >= 6.0 - SEAM_DEGREES:
        out.append(1 if z == 60 else z + 1)
    return out


def _patch_crs(lon: float, lat: float) -> str:
    """A transverse Mercator CRS with its central meridian at *lon*.

    UTM's projection, centred on the patch rather than on a 6-degree zone,
    which keeps distortion small and symmetric whatever the patch spans.
    Returned as WKT with a descriptive name; no EPSG code exists for an
    arbitrary central meridian.
    """
    from pyproj import CRS

    y0 = 0 if lat >= 0 else 10000000
    crs = CRS.from_proj4(
        f"+proj=tmerc +lat_0=0 +lon_0={lon:.8f} +k=0.9996 "
        f"+x_0=500000 +y_0={y0} +datum=WGS84 +units=m +no_defs"
    )
    named = crs.to_json_dict()
    named["name"] = f"Tessera patch Transverse Mercator lon_0={lon:.4f}"
    return CRS.from_json_dict(named).to_wkt()


def _zones_spanned(lons: List[float], centre_lon: float) -> List[int]:
    """The contiguous run of UTM zones covering *lons*, walked the short
    way round the ring so a patch on the antimeridian gets ``[60, 1]``."""
    zc = _zone_for_lon(centre_lon)

    def offset(z: int) -> int:
        d = (z - zc) % 60
        return d - 60 if d > 30 else d

    offs = [offset(_zone_for_lon(lon)) for lon in lons]
    return [(zc - 1 + o) % 60 + 1 for o in range(min(offs), max(offs) + 1)]


def _utm_envelope(
    bbox: Tuple[float, float, float, float], crs: str
) -> Tuple[float, float, float, float]:
    """The UTM extent enclosing a lon/lat bbox, from all four corners.

    Northing extremes sit on different corners as the grid curves away
    from the central meridian; two corners under-cover a wide box.
    """
    corners = [
        _project(lon, lat, "EPSG:4326", crs)
        for lon in (bbox[0], bbox[2])
        for lat in (bbox[1], bbox[3])
    ]
    es = [c[0] for c in corners]
    ns = [c[1] for c in corners]
    return min(es), min(ns), max(es), max(ns)


def open_zone(
    store_url: str = DEFAULT_STORE,
    *,
    zone: Optional[int] = None,
    lon: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    **kwargs,
) -> xr.Dataset:
    """Open a tessera zone as an xarray Dataset.

    Provide exactly one of ``zone``, ``lon``, or ``bbox`` to select the
    UTM zone.  Returns a Dataset with the ``.tessera`` accessor.

    Args:
        store_url: Zarr store URL, local path, or a ``zarr.abc.store.Store``.
        zone: UTM zone number (1-60).
        lon: A longitude — zone is derived automatically.
        bbox: (min_lon, min_lat, max_lon, max_lat) — zone from centre.

    Example::

        from geotessera.store import open_zone
        ds = open_zone(lon=-2.97)
        ds = open_zone(bbox=(-3.0, 53.4, -2.9, 53.5))
        ds = open_zone(zone=30)
    """
    z = _resolve_zone(zone, lon, bbox)

    log.debug("open_zone: utm%02d from %s", z, store_url)
    ds = xr.open_zarr(
        zarr_store(store_url),
        group=f"utm{z:02d}",
        zarr_format=3,
        consolidated=True,
        chunks=SHARD_CHUNKS,
        **kwargs,
    )

    return ds


# ---------------------------------------------------------------------------
# xarray accessor
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("tessera")
class TesseraAccessor:
    """Tessera-aware methods on an xarray Dataset from a zarr zone.

    Uses coordinate-based selection (``sel(method='nearest')``) for all
    spatial lookups — no manual affine math.  Reads ``proj:code`` and
    ``spatial:transform`` from Dataset attrs, years from the time coordinate.
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        attrs = ds.attrs
        self._epsg: int = int(attrs["proj:code"].split(":")[1])
        # Derive years from the time coordinate
        if "time" in ds.coords:
            self._years: list[int] = [int(v) for v in ds.coords["time"].values]
        else:
            self._years = []
        # Read n_bands from geoemb:dimensions if available, else from band dim
        self._n_bands: int = int(
            attrs.get("geoemb:dimensions", ds.sizes.get("band", 128))
        )
        t = attrs["spatial:transform"]
        self._px: float = float(t[0])
        log.debug("TesseraAccessor: EPSG:%d, years=%s", self._epsg, self._years)

    # -- Properties ---------------------------------------------------------

    @property
    def crs(self) -> str:
        """CRS string, e.g. ``'EPSG:32630'``."""
        return f"EPSG:{self._epsg}"

    @property
    def pixel_size(self) -> float:
        """Pixel size in CRS units (metres for UTM)."""
        return self._px

    @property
    def years(self) -> list[int]:
        """Available years, matching the time dimension order."""
        return self._years

    @property
    def n_bands(self) -> int:
        """Number of embedding bands."""
        return self._n_bands

    # -- Dequantisation -----------------------------------------------------

    @staticmethod
    def dequantise(emb_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantise int8 embeddings: ``(B,H,W)`` + ``(H,W)`` → ``(H,W,B)`` float32.

        Non-finite scales (NaN = water, +inf = no data) produce NaN rows.
        """
        valid = np.isfinite(scales)
        safe = np.where(valid, scales, 0.0)
        f32 = emb_int8.astype(np.float32) * safe[np.newaxis, :, :]
        f32[:, ~valid] = np.nan
        return f32.transpose(1, 2, 0)

    # -- Point sampling -----------------------------------------------------

    def sample_at(
        self,
        e: float,
        n: float,
        year: int,
        *,
        search_px: int = SEAM_SEARCH_PX,
    ) -> np.ndarray:
        """Sample a single dequantised embedding.  Returns ``(B,)`` float32.

        Args:
            e: Easting in this zone's CRS.
            n: Northing in this zone's CRS.
            year: Year to sample.
            search_px: Nearest-valid-pixel radius for unwritten pixels;
                0 disables.  See :meth:`probe`.

        Coordinates are this zone's own UTM, matching the grid the data is
        stored on.  For longitude and latitude use
        :meth:`GeoTesseraZarr.sample_at`, which routes a point to its zone.
        """
        vec, _status = self.probe(e, n, year, search_px=search_px)
        return vec if vec is not None else np.full(self.n_bands, np.nan, np.float32)

    def probe(
        self,
        e: float,
        n: float,
        year: int,
        *,
        search_px: int = SEAM_SEARCH_PX,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Sample at UTM ``(e, n)``, reporting why when there is no value.

        Returns ``(embedding, status)``, the status one of ``valid``,
        ``water``, ``nodata`` (never written) or ``outside`` (beyond this
        zone's grid).

        Within *search_px*, an unwritten pixel falls back to the nearest valid
        one; 0 disables that.  Water is answered as ``water`` rather than
        searched past, so the fallback can never report land for a sea
        location.
        """
        xname, yname = ("xc", "yc") if "xc" in self._ds.coords else ("x", "y")
        xs = self._ds.coords[xname].values
        ys = self._ds.coords[yname].values
        xi = int(np.abs(xs - e).argmin())
        yi = int(np.abs(ys - n).argmin())

        # sel(method="nearest") would snap a distant point to an edge pixel.
        if abs(xs[xi] - e) > self._px or abs(ys[yi] - n) > self._px:
            return None, OUTSIDE

        r = max(0, int(search_px))
        x0, x1 = max(0, xi - r), min(len(xs), xi + r + 1)
        y0, y1 = max(0, yi - r), min(len(ys), yi + r + 1)
        win = self._ds.isel({xname: slice(x0, x1), yname: slice(y0, y1)}).sel(time=year)
        scales = np.asarray(win["scales"].values, dtype=np.float64)
        ci, cj = yi - y0, xi - x0

        centre = scales[ci, cj]
        if np.isnan(centre):
            return None, WATER
        if np.isfinite(centre):
            bi, bj = ci, cj
        else:
            rows, cols = np.nonzero(np.isfinite(scales))
            if not len(rows):
                return None, NODATA
            k = int(np.argmin((rows - ci) ** 2 + (cols - cj) ** 2))
            bi, bj = int(rows[k]), int(cols[k])
        emb = np.asarray(win["embeddings"].values)
        return emb[:, bi, bj].astype(np.float32) * float(scales[bi, bj]), VALID

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at points in one vectorised read.

        Args:
            coords: List of ``(easting, northing)`` in this zone's CRS.

        Returns ``(N, B)`` float32, NaN for water and points beyond the
        grid.  Unwritten pixels retry through :meth:`sample_at`.
        ``progress`` is deprecated and ignored: progress is logged
        through the ``geotessera.store`` logger at INFO.
        """
        del progress
        coords = list(coords)
        if not coords:
            return np.empty((0, self.n_bands), np.float32)
        es = np.array([c[0] for c in coords], dtype=float)
        ns = np.array([c[1] for c in coords], dtype=float)
        values, cause = _bulk_sample(self._ds, es, ns, year)
        holes = np.flatnonzero(cause == _HOLE)
        for i in _progress_iter(holes, "Repairing pixels"):
            values[i] = self.sample_at(es[i], ns[i], year)
        return values

    # -- Region reading -----------------------------------------------------

    def _window(self, bbox, year, array):
        """Load a bbox window: ``(emb int8 (B, H, W), scales, transform)``."""
        e_min, e_max = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
        n_min, n_max = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])

        # y is descending (north→south), so slice is (n_max, n_min)
        sub = self._ds.sel(time=year, x=slice(e_min, e_max), y=slice(n_max, n_min))
        h, w = int(sub.sizes["y"]), int(sub.sizes["x"])
        log.info(
            "read_region: %d x %d pixels (%s), %.0fm resolution",
            h,
            w,
            f"{h * w:,}",
            self._px,
        )

        started = time.monotonic()
        scales = sub["scales"].values
        emb_int8 = sub[array].values
        log.info("read_region: loaded in %.1fs", time.monotonic() - started)

        # Build affine from the selected window's coordinate values
        x0 = float(sub["x"].values[0]) - 0.5 * self._px  # pixel centre → corner
        y0 = float(sub["y"].values[0]) + 0.5 * self._px
        transform = rasterio.transform.Affine(self._px, 0, x0, 0, -self._px, y0)
        return emb_int8, scales, transform

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        array: str = "embeddings",
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read and dequantise a bbox region.

        Args:
            bbox: ``(e_min, n_min, e_max, n_max)`` in this zone's CRS.
            array: Embeddings array to read, e.g. a matryoshka depth
                array such as ``"embeddings_d16"``.

        Returns ``(mosaic, transform)`` where mosaic is ``(H, W, B)``
        float32 and transform is a rasterio Affine for the window.  Both are
        in this zone's UTM — nothing is resampled on the way out.
        """
        del progress  # deprecated and ignored; progress is always logged
        emb_int8, scales, transform = self._window(bbox, year, array)
        return self.dequantise(emb_int8, scales), transform

    def read_region_quantized(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        array: str = "embeddings",
        progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, rasterio.transform.Affine]:
        """Read a bbox region without dequantising.

        Returns ``(embeddings, scales, transform)`` with embeddings int8
        ``(H, W, B)`` and scales float32 ``(H, W)``.  The window costs a
        quarter of the bytes of :meth:`read_region`; dequantise blocks of
        rows on demand with
        ``dequantise(emb[rows].transpose(2, 0, 1), scales[rows])``.
        """
        del progress  # deprecated and ignored; progress is always logged
        emb_int8, scales, transform = self._window(bbox, year, array)
        return emb_int8.transpose(1, 2, 0), scales, transform

    def iter_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        array: str = "embeddings",
        strip_rows: int = 512,
        progress: bool = False,
    ) -> Iterator[Tuple[np.ndarray, rasterio.transform.Affine]]:
        """:meth:`read_region` as a stream of row strips.

        Yields ``(block, transform)`` top to bottom, each block a
        ``(strip_rows, W, B)`` float32 slice of the region, downloading
        the next strip while the caller works on the current one.  For a
        dask-native pipeline, apply ``xarray.apply_ufunc`` to the
        Dataset instead.

        Args:
            bbox: ``(e_min, n_min, e_max, n_max)`` in this zone's CRS.
            array: Embeddings array to read.
            strip_rows: Rows per yielded block.
        """
        del progress  # deprecated and ignored; progress is always logged
        e_min, e_max = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
        n_min, n_max = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
        sub = self._ds.sel(time=year, x=slice(e_min, e_max), y=slice(n_max, n_min))
        height = int(sub.sizes["y"])

        def load(top):
            strip = sub.isel(y=slice(top, min(top + strip_rows, height)))
            # One or two dask tasks per strip; raise zarr's per-task concurrency.
            with zarr.config.set({"async.concurrency": POINT_CONCURRENCY}):
                block = self.dequantise(
                    strip[array].values, strip["scales"].values
                )
            x0 = float(strip["x"].values[0]) - 0.5 * self._px
            y0 = float(strip["y"].values[0]) + 0.5 * self._px
            return block, rasterio.transform.Affine(
                self._px, 0, x0, 0, -self._px, y0
            )

        tops = list(range(0, height, strip_rows))
        yield from _progress_iter(
            _prefetched(load, tops), "Reading strips", total=len(tops)
        )


# ---------------------------------------------------------------------------
# GeoTesseraZarr — store-level API with zone routing
# ---------------------------------------------------------------------------


class GeoTesseraZarr:
    """Read embeddings from a Tessera zarr store.

    Routes geographic queries to the correct UTM zone automatically.
    For single-zone work, use :func:`open_zone` directly.

    Args:
        store_url: Zarr store URL, local path, or a ``zarr.abc.store.Store``
            such as a cache-wrapped store from :func:`zarr_store`.
            Defaults to the public TESSERA store at
            ``data.source.coop/tessera/tessera/zarr``.
        cache_dir: Persist reads under this directory, keyed per store
            location (see :func:`zarr_store`). Requires a URL or path
            ``store_url``, not a ``Store`` object.
        cache_max_size: Bound the *cache_dir* cache in bytes (default
            unbounded).

    Example::

        from geotessera.store import GeoTesseraZarr

        gt = GeoTesseraZarr()
        print(gt.years)  # [2017, 2018, ..., 2025]

        # Sample embeddings at points
        X = gt.sample_points([(-2.97, 53.44)], year=2025)

        # Read a region
        mosaic, transform, crs = gt.read_region(
            (-3.0, 53.4, -2.9, 53.5), year=2025,
        )
    """

    def __init__(
        self,
        store_url: Union[str, ZarrStore] = DEFAULT_STORE,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_max_size: Optional[int] = None,
    ):
        if isinstance(store_url, str):
            store_url = store_url.rstrip("/")
        self.url = str(store_url)
        self._store = zarr_store(
            store_url, cache_dir=cache_dir, cache_max_size=cache_max_size
        )
        try:
            root = zarr.open_group(self._store, mode="r")
        except (zarr.errors.GroupNotFoundError, zarr.errors.ArrayNotFoundError, KeyError) as e:
            from .registry import KNOWN_DATASETS
            
            available = ", ".join(sorted({f"v{v}" for v, _, d in KNOWN_DATASETS if d}))
            raise ValueError(
                f"Failed to open zarr store at {self.url!r}. "
                f"The store may not exist or the URL may be incorrect.\n"
                f"Known versions: {available}. "
                f"Use zarr_store_url() to generate correct store URLs, e.g., "
                f"zarr_store_url('v1') or zarr_store_url('v2')."
            ) from e
        
        self._root = root
        root_attrs = dict(root.attrs)
        self.model_version: str = root_attrs.get("geoemb:model", "")
        self.build_version: str = root_attrs.get("geoemb:build_version", "")
        self.n_bands: int = int(root_attrs.get("geoemb:dimensions", 128))
        # Matryoshka depth arrays declared by the store, if any
        self.depths: dict[int, str] = {
            int(d["dimensions"]): str(d["array"])
            for d in root_attrs.get("geoemb:depths", [])
        } or {self.n_bands: "embeddings"}
        # Derive years from the first zone's time coordinate array
        self.years: list[int] = []
        for member_name in sorted(root.keys()):
            if member_name.startswith("utm"):
                try:
                    zone_grp = root[member_name]
                    time_arr = zone_grp["time"][:]
                    self.years = [int(v) for v in time_arr]
                    break
                except Exception:
                    continue
        self._cache: dict[int, xr.Dataset] = {}
        log.info(
            "GeoTesseraZarr: %s, years=%s, model=%s",
            self.url,
            self.years,
            self.model_version,
        )

    def __repr__(self) -> str:
        return f"GeoTesseraZarr({self.url!r}, years={self.years})"

    # -- Zone access --------------------------------------------------------

    def open_zone(
        self,
        *,
        zone: Optional[int] = None,
        lon: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """Open a zone Dataset with the ``.tessera`` accessor.

        Provide exactly one of ``zone``, ``lon``, or ``bbox``.
        Datasets are cached for the lifetime of this instance.
        """
        z = _resolve_zone(zone, lon, bbox)
        if z not in self._cache:
            ds = open_zone(self._store, zone=z)
            self._cache[z] = ds
        return self._cache[z]

    # -- Point sampling (cross-zone) ----------------------------------------

    def sample_at(
        self,
        lon: float,
        lat: float,
        year: int,
        *,
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
    ) -> np.ndarray:
        """Sample a single embedding, routing the point to its UTM zone.

        Args:
            lon: Longitude (WGS84).
            lat: Latitude (WGS84).
            cross_zone: Also try the neighbouring zone near a seam.  A tile
                belongs to the zone holding its centre, so a point on a seam
                is often covered by the zone next door.
            search_px: Nearest-valid-pixel radius; 0 disables.

        Returns ``(B,)`` float32, or a NaN row for open water and for
        locations outside coverage.  Use :meth:`probe` to tell those apart.
        """
        vec, _status = self.probe(
            lon, lat, year, cross_zone=cross_zone, search_px=search_px
        )
        return vec if vec is not None else np.full(self.n_bands, np.nan, np.float32)

    def probe(
        self,
        lon: float,
        lat: float,
        year: int,
        *,
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
    ) -> Tuple[Optional[np.ndarray], str]:
        """:meth:`sample_at`, reporting why when there is no value.

        Returns ``(embedding, status)``; see :meth:`TesseraAccessor.probe`.
        Use it in place of testing ``sample_at`` for NaN, which cannot tell
        open water from a location missing from the store.
        """
        zones = [_zone_for_lon(lon)]
        if cross_zone:
            zones += [z for z in _seam_neighbours(lon) if z not in zones]

        seen = set()
        for z in zones:
            try:
                acc = self.open_zone(zone=z).tessera
            except KeyError as exc:
                # Only "this store has no such zone group". A missing store or
                # one without consolidated metadata raises FileNotFoundError or
                # ValueError and fails every zone alike, so letting those
                # through here would report `outside` for a broken store
                # instead of saying what is wrong.
                log.debug("probe: zone %d not in this store (%s)", z, exc)
                continue
            e, n = _project(lon, lat, "EPSG:4326", acc.crs)
            vec, status = acc.probe(e, n, year, search_px=search_px)
            if status == VALID:
                if z != zones[0]:
                    log.debug("probe: %.6f,%.6f served from zone %d", lon, lat, z)
                return vec, VALID
            seen.add(status)

        # Water is a real answer about the location, so it outranks the
        # rest; outside means no zone's grid covered the point at all.
        for status in (WATER, NODATA):
            if status in seen:
                return None, status
        return None, OUTSIDE

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        progress: bool = True,
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
        depth: Optional[int] = None,
    ) -> np.ndarray:
        """Sample embeddings at points, routing each to its zone.

        Args:
            coords: List of ``(lon, lat)`` tuples in WGS84.
            cross_zone: See :meth:`sample_at`.
            search_px: See :meth:`sample_at`.
            depth: Matryoshka depth to read, e.g. 16 on a v2 store; the
                first *depth* dimensions arrive for a fraction of the
                bytes.  None reads the full embedding.

        Returns ``(N, B)`` float32, one bulk read per UTM zone, NaN rows
        for points without an embedding.  Unwritten pixels and points
        near a zone seam retry through :meth:`sample_at`.
        ``progress`` is deprecated and ignored: progress is logged
        through the ``geotessera.store`` logger at INFO.
        """
        del progress
        array, n_bands = self._embeddings_array(depth)
        coords = list(coords)
        if not coords:
            return np.empty((0, n_bands), np.float32)
        lons = np.array([c[0] for c in coords], dtype=float)
        lats = np.array([c[1] for c in coords], dtype=float)
        values = np.full((len(coords), n_bands), np.nan, np.float32)
        cause = np.full(len(coords), _OUT, np.int8)

        zones = np.array([_zone_for_lon(lon) for lon in lons])
        for z in np.unique(zones):
            idx = np.flatnonzero(zones == z)
            try:
                ds = self.open_zone(zone=int(z))
            except KeyError:
                continue  # not in this store; the seam retry may still answer
            es, ns = _transformer("EPSG:4326", ds.tessera.crs).transform(
                lons[idx], lats[idx]
            )
            values[idx], cause[idx] = _bulk_sample(
                ds, es, ns, year,
                read=self._point_reader(int(z), ds, year, array),
                array=array,
            )

        near_seam = np.array([bool(_seam_neighbours(lon)) for lon in lons])
        retry = ((cause == _HOLE) & (search_px > 0)) | (
            (cause != _OK) & near_seam & cross_zone
        )
        retry_idx = np.flatnonzero(retry)
        for i in _progress_iter(retry_idx, "Repairing points"):
            # Depth arrays are prefixes of the full embedding, so the
            # per-point path reads the full vector and slices it.
            values[i] = self.sample_at(
                lons[i], lats[i], year, cross_zone=cross_zone, search_px=search_px
            )[:n_bands]
        return values

    def _embeddings_array(self, depth: Optional[int]) -> Tuple[str, int]:
        """Array name and band count for *depth*; None selects the full array."""
        if depth is None:
            return "embeddings", self.n_bands
        if depth not in self.depths:
            raise ValueError(
                f"depth {depth} not in this store; available: {sorted(self.depths)}"
            )
        return self.depths[depth], depth

    def _point_reader(self, zone: int, ds: xr.Dataset, year: int, array: str):
        """A pixel-index reader through zarr's own concurrent pipeline."""
        group = self._root[f"utm{zone:02d}"]
        ti = int(np.flatnonzero(ds["time"].values == year)[0])

        def read(xi, yi):
            bands = np.arange(group[array].shape[1])
            t, b, y, x = np.broadcast_arrays(
                np.full((len(xi), 1), ti), bands[None, :], yi[:, None], xi[:, None]
            )
            with zarr.config.set({"async.concurrency": POINT_CONCURRENCY}):
                emb = group[array].vindex[t, b, y, x]
                scales = group["scales"].vindex[np.full(len(xi), ti), yi, xi]
            return emb, scales

        return read

    # -- Region reading (dominant zone) -------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        depth: Optional[int] = None,
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read and dequantise a bbox region.

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
            depth: Matryoshka depth to read, e.g. 16 on a v2 store.
                None reads the full embedding.

        Routes to the zone holding the bbox centre and returns
        ``(mosaic, transform, crs)`` with mosaic ``(H, W, B)`` float32.  The
        mosaic is in that zone's UTM, not in WGS84: the bbox is projected to
        pick the window, and the pixels come back on their native grid
        untouched.

        A bbox crossing a zone boundary is served from the centre zone
        alone; :meth:`read_patch` merges across zones.
        """
        del progress  # deprecated and ignored; progress is always logged
        array, _ = self._embeddings_array(depth)
        z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        edge_zones = _zones_spanned([bbox[0], bbox[2]], (bbox[0] + bbox[2]) / 2)
        if len(edge_zones) > 1:
            log.warning(
                "read_region: bbox spans UTM zones %s; reading zone %d only — "
                "use read_patch() to merge across zones",
                edge_zones,
                z,
            )
        ds = self.open_zone(zone=z)
        zone_crs = ds.tessera.crs

        utm_bbox = _utm_envelope(bbox, zone_crs)
        mosaic, transform = ds.tessera.read_region(utm_bbox, year, array=array)
        return mosaic, transform, zone_crs

    def read_region_quantized(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        depth: Optional[int] = None,
        progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, str]:
        """:meth:`read_region` without dequantising.

        Returns ``(embeddings, scales, transform, crs)`` with embeddings
        int8 ``(H, W, B)`` and scales float32 ``(H, W)``.  The window
        costs a quarter of the bytes of :meth:`read_region`; dequantise
        blocks of rows on demand with
        ``dequantise(emb[rows].transpose(2, 0, 1), scales[rows])``.
        """
        del progress  # deprecated and ignored; progress is always logged
        array, _ = self._embeddings_array(depth)
        z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        ds = self.open_zone(zone=z)
        zone_crs = ds.tessera.crs
        utm_bbox = _utm_envelope(bbox, zone_crs)
        emb, scales, transform = ds.tessera.read_region_quantized(
            utm_bbox, year, array=array
        )
        return emb, scales, transform, zone_crs

    def iter_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        depth: Optional[int] = None,
        strip_rows: int = 512,
        progress: bool = False,
    ) -> Iterator[Tuple[np.ndarray, rasterio.transform.Affine, str]]:
        """:meth:`read_region` as a stream of row strips.

        Yields ``(block, transform, crs)`` top to bottom; see
        :meth:`TesseraAccessor.iter_region`.  Routed like
        :meth:`read_region`, to the zone holding the bbox centre.
        """
        del progress  # deprecated and ignored; progress is always logged
        array, _ = self._embeddings_array(depth)
        z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        ds = self.open_zone(zone=z)
        acc = ds.tessera
        zone_crs = acc.crs
        utm_bbox = _utm_envelope(bbox, zone_crs)

        # A strip is one getitem, so its chunk fetches run concurrently.
        group = self._root[f"utm{z:02d}"]
        xs, ys = ds["x"].values, ds["y"].values
        x0 = int(np.searchsorted(xs, utm_bbox[0], "left"))
        x1 = int(np.searchsorted(xs, utm_bbox[2], "right"))
        y0 = int(np.searchsorted(-ys, -utm_bbox[3], "left"))
        y1 = int(np.searchsorted(-ys, -utm_bbox[1], "right"))
        ti = int(np.flatnonzero(ds["time"].values == year)[0])
        px = acc.pixel_size

        def load(top):
            end = min(top + strip_rows, y1)
            with zarr.config.set({"async.concurrency": POINT_CONCURRENCY}):
                emb = group[array][ti, :, top:end, x0:x1]
                scales = group["scales"][ti, top:end, x0:x1]
            transform = rasterio.transform.Affine(
                px, 0, xs[x0] - 0.5 * px, 0, -px, ys[top] + 0.5 * px
            )
            return acc.dequantise(emb, scales), transform

        tops = list(range(y0, y1, strip_rows))
        for block, transform in _progress_iter(
            _prefetched(load, tops), "Reading strips", total=len(tops)
        ):
            yield block, transform, zone_crs

    # -- Patch reading (cross-zone) ------------------------------------------

    def read_patch(
        self,
        lon: float,
        lat: float,
        year: int,
        size_px: int,
        *,
        depth: Optional[int] = None,
        dst_crs: Optional[str] = None,
        resampling: str = "nearest",
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read a fixed-size square patch centred on a point.

        Returns ``(patch, transform, crs)``.  The patch is exactly
        ``(size_px, size_px, B)`` float32, NaN where the store holds
        nothing; the point falls in pixel ``[size_px // 2, size_px // 2]``.

        A patch within one UTM zone is sliced from that zone's grid
        unresampled, in the zone's CRS.  A patch crossing a zone boundary
        is merged onto a transverse Mercator grid centred on the patch,
        relocating pixels whole with nearest-neighbour resampling; each
        pixel comes from the zone owning its longitude, neighbours filling
        gaps, as :meth:`sample_at` prefers.

        Args:
            lon: Patch centre longitude (WGS84).
            lat: Patch centre latitude (WGS84).
            year: Embedding year.
            size_px: Patch width and height in pixels.
            dst_crs: Output CRS, for pipelines that need every patch in
                one CRS; forces the merge path even within one zone.
            resampling: rasterio resampling name for the merge path.
                Only the ``"nearest"`` default leaves vectors unblended.
            progress: Deprecated and ignored; progress is logged
                through the ``geotessera.store`` logger at INFO.
        """
        del progress
        if size_px <= 0:
            raise ValueError(f"size_px must be positive, got {size_px}")
        array, n_bands = self._embeddings_array(depth)

        centre_ds = self.open_zone(lon=lon)
        acc = centre_ds.tessera
        px = acc.pixel_size
        half = size_px * px / 2.0

        ce, cn = _project(lon, lat, "EPSG:4326", acc.crs)
        corner_lons = [
            _project(ce + dx, cn + dy, acc.crs, "EPSG:4326")[0]
            for dx in (-half, half)
            for dy in (-half, half)
        ]
        zones = _zones_spanned(corner_lons, lon)

        if dst_crs is None and len(zones) == 1:
            return self._read_patch_native(
                centre_ds, ce, cn, year, size_px, array
            )

        target_crs = dst_crs or _patch_crs(lon, lat)
        return self._read_patch_merged(
            zones, target_crs, lon, lat, year, size_px, px,
            array, n_bands, resampling,
        )

    def _read_patch_native(
        self,
        ds: xr.Dataset,
        centre_e: float,
        centre_n: float,
        year: int,
        size_px: int,
        array: str,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Slice a patch straight off one zone's grid, NaN-padded at edges.

        One getitem through zarr's own pipeline; reading through dask
        would materialise whole shard chunks for a small window.
        """
        acc = ds.tessera
        px = acc.pixel_size
        xs, ys = ds["x"].values, ds["y"].values
        ix = int(np.abs(xs - centre_e).argmin())
        iy = int(np.abs(ys - centre_n).argmin())
        x0, y0 = ix - size_px // 2, iy - size_px // 2
        cx0, cx1 = max(0, x0), min(len(xs), x0 + size_px)
        cy0, cy1 = max(0, y0), min(len(ys), y0 + size_px)

        out = np.full((size_px, size_px, int(ds[array].shape[1])), np.nan, np.float32)
        if cx1 > cx0 and cy1 > cy0:
            group = self._root[f"utm{int(acc.crs.split(':')[1]) % 100:02d}"]
            ti = int(np.flatnonzero(ds["time"].values == year)[0])
            with zarr.config.set({"async.concurrency": POINT_CONCURRENCY}):
                emb_int8 = group[array][ti, :, cy0:cy1, cx0:cx1]
                scales = group["scales"][ti, cy0:cy1, cx0:cx1]
            out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = acc.dequantise(
                emb_int8, scales
            )

        transform = rasterio.transform.Affine(
            px, 0, (xs[0] - 0.5 * px) + x0 * px, 0, -px, (ys[0] + 0.5 * px) - y0 * px
        )
        self._log_patch_coverage(out, size_px)
        return out, transform, acc.crs

    def _read_patch_merged(
        self,
        zones: List[int],
        target_crs: str,
        lon: float,
        lat: float,
        year: int,
        size_px: int,
        px: float,
        array: str,
        n_bands: int,
        resampling: str,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Merge each zone's native pixels onto one patch-centred grid."""
        ce, cn = _project(lon, lat, "EPSG:4326", target_crs)
        # The point lands on the centre of pixel [size_px // 2, size_px // 2].
        ox = ce - (size_px // 2 + 0.5) * px
        oy = cn + (size_px // 2 + 0.5) * px
        transform = rasterio.transform.Affine(px, 0, ox, 0, -px, oy)

        # Densified outline: a straight edge curves in a zone's CRS, so
        # corners alone under-cover mid-edge.
        steps = np.linspace(0.0, size_px * px, 33)
        outline = (
            [(ox + s, oy) for s in steps]
            + [(ox + s, oy - size_px * px) for s in steps]
            + [(ox, oy - s) for s in steps]
            + [(ox + size_px * px, oy - s) for s in steps]
        )

        gx, gy = np.meshgrid(
            ox + (np.arange(size_px) + 0.5) * px,
            oy - (np.arange(size_px) + 0.5) * px,
        )
        lons, _ = _transformer(target_crs, "EPSG:4326").transform(gx, gy)
        owner = np.clip(np.floor((lons + 180.0) / 6.0).astype(int) + 1, 1, 60)

        out = np.full((size_px, size_px, n_bands), np.nan, np.float32)
        owned = np.zeros((size_px, size_px), dtype=bool)
        spare = np.full_like(out, np.nan)
        spared = np.zeros((size_px, size_px), dtype=bool)
        for z in zones:
            try:
                acc = self.open_zone(zone=z).tessera
            except KeyError as exc:
                log.debug("read_patch: zone %d not in this store (%s)", z, exc)
                continue
            zone_pts = [_project(e, n, target_crs, acc.crs) for e, n in outline]
            es = [c[0] for c in zone_pts]
            ns = [c[1] for c in zone_pts]
            pad = 2 * px
            try:
                mosaic, src_transform = acc.read_region(
                    (min(es) - pad, min(ns) - pad, max(es) + pad, max(ns) + pad),
                    year,
                    array=array,
                )
            except IndexError:
                continue  # the window misses everything this zone holds
            if 0 in mosaic.shape[:2]:
                continue
            relocated = np.full((n_bands, size_px, size_px), np.nan, np.float32)
            reproject(
                source=mosaic.transpose(2, 0, 1),
                destination=relocated,
                src_transform=src_transform,
                src_crs=acc.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling[resampling],
            )
            relocated = relocated.transpose(1, 2, 0)
            valid = np.isfinite(relocated).any(axis=2)
            mine = valid & (owner == z)
            out[mine] = relocated[mine]
            owned |= mine
            extra = valid & (owner != z) & ~spared
            spare[extra] = relocated[extra]
            spared |= extra

        fill = ~owned & spared
        out[fill] = spare[fill]
        self._log_patch_coverage(out, size_px)
        return out, transform, target_crs

    @staticmethod
    def _log_patch_coverage(patch: np.ndarray, size_px: int) -> None:
        fraction = float(np.isfinite(patch).any(axis=2).mean())
        if fraction < 1.0:
            log.warning(
                "read_patch: %.1f%% of the %dx%d patch has no coverage (NaN)",
                (1.0 - fraction) * 100.0,
                size_px,
                size_px,
            )
