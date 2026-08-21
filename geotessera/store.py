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

    # Direct zone access, in that zone's UTM
    ds = gt.open_zone(lon=-2.97)
    emb = ds.tessera.sample_at(500_000.0, 5_921_000.0, year=2025)

Working from another CRS?  Project your points to lon/lat once, up front,
rather than per call.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import rasterio.transform
import xarray as xr
import zarr
from pyproj import Transformer
from rich.progress import track

from .registry import zarr_store_url

log = logging.getLogger(__name__)

DEFAULT_STORE = zarr_store_url("v1")

# Shard-aligned chunk sizes so dask tasks match zarr shards
SHARD_CHUNKS = {"time": 1, "band": 128, "y": 4096, "x": 4096}


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


def _sample_each(sample_at, coords, progress: bool) -> np.ndarray:
    """Apply a per-point ``sample_at(x, y)`` across *coords*, giving ``(N, B)``."""
    it = coords
    if progress:
        it = track(coords, description="Sampling points...", transient=True)
    return np.array([sample_at(x, y) for x, y in it])


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
        store_url: Zarr store URL or local path.
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
        store_url,
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
        """Sample embeddings at points.  Returns ``(N, B)`` float32.

        Args:
            coords: List of ``(easting, northing)`` in this zone's CRS.
        """
        return _sample_each(lambda e, n: self.sample_at(e, n, year), coords, progress)

    # -- Region reading -----------------------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read and dequantise a bbox region.

        Args:
            bbox: ``(e_min, n_min, e_max, n_max)`` in this zone's CRS.

        Returns ``(mosaic, transform)`` where mosaic is ``(H, W, B)``
        float32 and transform is a rasterio Affine for the window.  Both are
        in this zone's UTM — nothing is resampled on the way out.
        """
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

        if progress:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                scales = sub["scales"].values
                emb_int8 = sub["embeddings"].values
        else:
            scales = sub["scales"].values
            emb_int8 = sub["embeddings"].values

        mosaic = self.dequantise(emb_int8, scales)

        # Build affine from the selected window's coordinate values
        x0 = float(sub["x"].values[0]) - 0.5 * self._px  # pixel centre → corner
        y0 = float(sub["y"].values[0]) + 0.5 * self._px
        transform = rasterio.transform.Affine(self._px, 0, x0, 0, -self._px, y0)
        return mosaic, transform


# ---------------------------------------------------------------------------
# GeoTesseraZarr — store-level API with zone routing
# ---------------------------------------------------------------------------


class GeoTesseraZarr:
    """Read embeddings from a Tessera zarr store.

    Routes geographic queries to the correct UTM zone automatically.
    For single-zone work, use :func:`open_zone` directly.

    Args:
        store_url: Zarr store URL or local path.  Defaults to the public
            TESSERA store at ``data.source.coop/tessera/tessera/zarr``.

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

    def __init__(self, store_url: str = DEFAULT_STORE):
        self.url = store_url.rstrip("/")
        root = zarr.open_group(self.url, mode="r")
        root_attrs = dict(root.attrs)
        self.model_version: str = root_attrs.get("geoemb:model", "")
        self.build_version: str = root_attrs.get("geoemb:build_version", "")
        self.n_bands: int = int(root_attrs.get("geoemb:dimensions", 128))
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
            ds = open_zone(self.url, zone=z)
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
    ) -> np.ndarray:
        """Sample embeddings at points, routing each to its zone.

        Args:
            coords: List of ``(lon, lat)`` tuples in WGS84.
            cross_zone: See :meth:`sample_at`.
            search_px: See :meth:`sample_at`.

        Returns ``(N, B)`` float32.  Points without an embedding get NaN rows.
        """
        return _sample_each(
            lambda lon, lat: self.sample_at(
                lon, lat, year, cross_zone=cross_zone, search_px=search_px
            ),
            coords,
            progress,
        )

    # -- Region reading (dominant zone) -------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read and dequantise a bbox region.

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.

        Routes to the zone holding the bbox centre and returns
        ``(mosaic, transform, crs)`` with mosaic ``(H, W, B)`` float32.  The
        mosaic is in that zone's UTM, not in WGS84: the bbox is projected to
        pick the window, and the pixels come back on their native grid
        untouched.
        """
        z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        ds = self.open_zone(zone=z)
        zone_crs = ds.tessera.crs

        # Project the corners to pick the window; the data itself is never
        # resampled.  A lon/lat box is not axis-aligned in UTM, so take the
        # enclosing easting/northing extent.
        e_nw, n_nw = _project(bbox[0], bbox[3], "EPSG:4326", zone_crs)
        e_se, n_se = _project(bbox[2], bbox[1], "EPSG:4326", zone_crs)
        utm_bbox = (
            min(e_nw, e_se),
            min(n_nw, n_se),
            max(e_nw, e_se),
            max(n_nw, n_se),
        )

        mosaic, transform = ds.tessera.read_region(utm_bbox, year, progress=progress)
        return mosaic, transform, zone_crs
