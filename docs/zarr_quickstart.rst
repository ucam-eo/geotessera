Zarr Quick Start
================

The zarr store streams Tessera embeddings from the public cloud store.
Nothing is downloaded up front, queries are routed to the correct UTM
zone, and values return dequantised as float32 on their native 10m UTM
grid. This is the recommended interface for most work; the
:doc:`quickstart` covers the tile-download interface for offline use.

Requires Python 3.12 or later::

    pip install geotessera

Read one embedding
------------------

Tessera publishes a 128-dimensional embedding for every 10m pixel of
land, for every year since 2017::

    from geotessera import GeoTesseraZarr

    gt = GeoTesseraZarr()
    print(gt.years)                     # [2017, ..., 2025]

    vec, status = gt.probe(0.12, 52.20, year=2024)
    print(status)                       # 'valid'
    print(vec.shape)                    # (128,)

``probe`` reports why when there is no value: ``water`` for open water,
``nodata`` for a pixel never written, ``outside`` for a point beyond the
store. ``sample_at`` returns the vector alone, with NaN for all three.

Sample points
-------------

``sample_points`` reads a list of lon/lat points in one bulk request per
UTM zone::

    coords = [(0.12, 52.20), (-2.97, 53.44)]
    X = gt.sample_points(coords, year=2024)   # (2, 128) float32

Points without an embedding return NaN rows. Points on a UTM zone seam
are served by the neighbouring zone when their own lacks them.

Read a region
-------------

``read_region`` takes a lon/lat bounding box and returns the mosaic on
the zone's native UTM grid, with its transform and CRS::

    bbox = (0.05, 52.15, 0.20, 52.25)
    mosaic, transform, crs = gt.read_region(bbox, year=2024)
    print(mosaic.shape, crs)            # (1152, 1069, 128) EPSG:32631

Nothing is resampled: the bounding box selects the window, and the
pixels come back on the grid they were produced on. Classify or cluster
on this grid, and reproject only the result.

Stream a large region
---------------------

A dequantised region costs four bytes per value, so a large one may not
fit in memory. ``iter_region`` yields the same pixels as row strips,
downloading the next strip while the caller works on the current one::

    for block, transform, crs in gt.iter_region(bbox, year=2024, strip_rows=512):
        predictions = model.predict(block.reshape(-1, 128))

``read_region_quantized`` is the other route to a large window: it
returns the int8 values and their scales without dequantising, a
quarter of the bytes, for dequantisation a block of rows at a time.

Read a patch
------------

``read_patch`` returns a fixed-size square centred on a point, the
shape a training pipeline consumes::

    patch, transform, crs = gt.read_patch(0.12, 52.20, year=2024, size_px=256)
    print(patch.shape)                  # (256, 256, 128)

The point falls in the centre pixel. A patch inside one UTM zone is
sliced from the native grid unresampled; one crossing a zone boundary
is merged onto a transverse Mercator grid centred on the patch, and its
CRS is returned as named WKT since no EPSG code exists for it.

Matryoshka depths
-----------------

The v2 model orders its dimensions by importance, and v2 stores carry
prefix arrays alongside the full embeddings. ``depth=`` reads them::

    from geotessera.registry import zarr_store_url

    gt2 = GeoTesseraZarr(zarr_store_url("v2"))
    X16 = gt2.sample_points(coords, year=2024, depth=16)   # (2, 16)

Sixteen dimensions arrive for an eighth of the bytes of 128, and equal
the first sixteen of the full embedding exactly. A store without the
requested depth raises and lists the depths it has.

Embedding releases
------------------

Releases sit side by side in the store, so trialling a model is a
one-line change::

    gt = GeoTesseraZarr(zarr_store_url("v1"))    # the default
    gt = GeoTesseraZarr(zarr_store_url("v2"))    # the v2 beta

Do not mix embeddings from different releases in one analysis; the
feature spaces are independently learned. See :ref:`dataset-versions`.

Caching and store wrapping
--------------------------

Reads retry failed requests with exponential backoff, so a dropped
response from a busy server costs one chunk rather than the read.
Pass ``cache_dir`` to persist reads locally.  Each store caches under
its own subdirectory (``tessera-cache/v1/``,
``tessera-cache/v2-2B-L_beta1/``), so dataset versions never mix::

    gt = GeoTesseraZarr(cache_dir="tessera-cache")

    # or with a size bound on the cache:
    gt = GeoTesseraZarr(
        cache_dir="tessera-cache", cache_max_size=2 * 1024**3
    )

To layer other behaviour over the transport, wrap the store from
:func:`~geotessera.store.zarr_store` and pass it in.

Under the hood
--------------

The store layout, quantisation, and seam handling are described in
:doc:`architecture`. The
`geotessera-examples <https://github.com/ucam-eo/geotessera-examples>`_
repository carries runnable pipelines built on this interface, including
a five-step teaching tour that ends by reading the store with plain
``xarray`` and ``zarr``.
