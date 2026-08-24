"""GeoTessera: access to Tessera satellite embeddings.

There are two ways in, and which one you want depends on how much data you
need at a time.

**Zarr store** (:class:`GeoTesseraZarr`) — read straight from the published
cloud-optimised store.  Nothing is downloaded up front, queries are routed to
the right UTM zone for you, and embeddings come back dequantised.  Use this
for point sampling and for regions you want as arrays::

    >>> from geotessera import GeoTesseraZarr
    >>> gt = GeoTesseraZarr()
    >>> gt.years
    [2017, 2018, ..., 2025]

    >>> # Sample embeddings at points — (N, 128) float32
    >>> X = gt.sample_points([(-2.97, 53.44), (-2.96, 53.43)], year=2025)

    >>> # Read a bounding box — (H, W, 128) float32
    >>> mosaic, transform, crs = gt.read_region((-3.0, 53.4, -2.9, 53.5), year=2025)

    >>> # Read a fixed-size patch centred on a point, whatever zones it spans
    >>> patch, transform, crs = gt.read_patch(0.0, 52.2, year=2025, size_px=512)

    >>> # Tell open water apart from a gap in coverage
    >>> vec, status = gt.probe(-2.97, 53.44, year=2025)  # 'valid' | 'water' | ...

**GeoTIFF export** (:class:`GeoTessera`) — download tiles and write them out
as standards-compliant GeoTIFFs for use in QGIS, GDAL and other GIS tools::

    >>> from geotessera import GeoTessera
    >>> gt = GeoTessera()
    >>> bbox = (-0.2, 51.4, 0.1, 51.6)  # London
    >>> tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    >>> files = gt.export_embedding_geotiffs(tiles, output_dir="tiles/", bands=[0, 1, 2])

Exported GeoTIFFs keep their native UTM projection, carry full metadata tags
and band descriptions, and are tiled and compressed.

For maps and web viewers built on those files, see :mod:`geotessera.visualization`
and :mod:`geotessera.web`::

    >>> from geotessera.visualization import create_rgb_mosaic
    >>> create_rgb_mosaic(files, "mosaic.tif")

The command-line tools are ``geotessera`` (data access and visualisation) and
``geotessera-registry`` (registry and store maintenance).
"""

from .core import GeoTessera, dequantize_embedding
from . import visualization
from . import web
from . import registry


def __getattr__(name):
    if name == "GeoTesseraZarr":
        from .store import GeoTesseraZarr

        return GeoTesseraZarr
    if name == "store":
        import importlib

        return importlib.import_module("geotessera.store")
    raise AttributeError(f"module 'geotessera' has no attribute {name!r}")


try:
    import importlib.metadata

    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development installs
    __version__ = "unknown"

__all__ = [
    "GeoTessera",
    "GeoTesseraZarr",
    "dequantize_embedding",
    "visualization",
    "web",
    "registry",
    "store",
]
