---
name: geotessera
description: Read Tessera satellite embeddings with the geotessera Python library. Use when sampling, mapping, or classifying with Tessera embeddings — reading points, regions, or patches from the zarr store, selecting dataset versions, or building land-cover and detection workflows on the embeddings.
---

# GeoTessera zarr interface

Tessera is a geospatial foundation model. It compresses a year of
Sentinel-1/2 observations into a 128-dimensional embedding for every 10m
pixel of land, for every year since 2017. The `geotessera` library streams
these embeddings from a public zarr v3 store. Nothing is downloaded up front,
and values return dequantised as float32.

Requires Python 3.12 or later: `pip install geotessera`.

## Rules

- Use `GeoTesseraZarr` (the zarr interface), not the tile-download
  `GeoTessera` class, unless the user explicitly needs offline NPY or
  GeoTIFF tile files.
- Never reproject embeddings before analysis. Embeddings return on their
  native UTM grid; classify or cluster on that grid, and reproject only
  the final result (predictions, renders).
- Coordinates are longitude/latitude in x, y order. Bounding boxes are
  `(min_lon, min_lat, max_lon, max_lat)`.
- A NaN vector means no embedding exists: water, or a pixel not yet
  produced. Use `probe` to distinguish, and drop NaN rows before fitting
  a model.
- Dequantised embeddings cost 512 bytes per pixel. For regions much
  beyond 1000x1000 pixels, stream with `iter_region` rather than holding
  a `read_region` mosaic in memory.
- Sample many points with one `sample_points` call; a per-point loop is
  an order of magnitude slower.
- Prefer the latest model version for which tiles exist in the coverage
  for the user's region of interest.

## Core calls

```python
from geotessera import GeoTesseraZarr

gt = GeoTesseraZarr()                    # default v1 store
gt.years                                 # [2017, ..., 2025]

vec, status = gt.probe(lon, lat, year)   # status: valid|water|nodata|outside
X = gt.sample_points(coords, year)       # (N, 128), one bulk read per UTM zone
mosaic, transform, crs = gt.read_region(bbox, year)
patch, transform, crs = gt.read_patch(lon, lat, year, size_px=256)
for block, transform, crs in gt.iter_region(bbox, year, strip_rows=512):
    ...                                  # row strips, next strip prefetched
```

`read_region` mosaics a lon/lat bounding box on the native UTM grid.
`read_patch` returns a fixed-size square centred on a point, merging
across UTM zones when the window spans one. `read_region_quantized`
returns int8 values plus their scales at a quarter of the memory;
dequantise blockwise as `values * scales`.

## Dataset versions and depth

```python
from geotessera.registry import zarr_store_url

gt = GeoTesseraZarr(zarr_store_url("v2"))   # "v1", "v2", or an explicit store URL
```

v2 stores publish matryoshka prefixes of each embedding. Passing
`depth=16` (or `depth=4`) to `sample_points`, `read_region`,
`read_patch`, or `iter_region` reads only the first N dimensions for
proportionally fewer bytes. `gt.depths` lists what a store offers.

## Caching and retries

HTTP retries with exponential backoff are built in; do not add a retry
layer. The constructor accepts any `zarr.abc.store.Store`, so when a
workflow reads the same chunks twice (for example sampling points and
then streaming the region that contains them), wrap the store in zarr's
`CacheStore` (requires `zarr>=3.3`; 3.1 corrupts cached range reads):

```python
from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore
from geotessera.store import DEFAULT_STORE, zarr_store

store = CacheStore(zarr_store(DEFAULT_STORE), cache_store=MemoryStore(),
                   max_size=2 * 1024**3)
gt = GeoTesseraZarr(store)
```

## Zone-level access

`gt.open_zone(lon=0.15)` returns an xarray Dataset for that UTM zone
with a `.tessera` accessor (`sample_at`, `read_region`) that works in
the zone's own eastings and northings and performs no projection.

## Typical classification workflow

1. `sample_points` at labelled coordinates; train a scikit-learn model
   on the (N, 128) matrix after dropping NaN rows.
2. `iter_region` over the target bounding box;
   `model.predict(block.reshape(-1, 128))` per strip.
3. Write predictions to a GeoTIFF with the yielded transform and CRS;
   reproject that raster, not the embeddings, if another CRS is needed.

Worked examples: https://github.com/ucam-eo/geotessera-examples
API reference: https://geotessera.readthedocs.io/en/latest/zarr_quickstart.html
