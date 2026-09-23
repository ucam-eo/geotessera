GeoTessera Documentation
========================

GeoTessera provides access to open geospatial embeddings from the `Tessera foundation model <https://github.com/ucam-eo/tessera>`_
(`paper <https://arxiv.org/abs/2506.20380>`_). Tessera processes Sentinel-1 and
Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m
resolution, compressing a full year of temporal-spectral features into dense
representations optimized for downstream geospatial analysis tasks.

.. important::

   **Multiple Tessera versions are published.** The default is **1.1** /
   ``dclimate``, a complete global run streamed from Icechunk. Its NPY
   tiles are not published; v1.1 NPY tiles come from the separate
   ``cambridge`` run. **TESSERA v2** betas (``2B-L~beta1``, ``2B-L~beta2``)
   are also published. The legacy 1.0 line is frozen. **Never mix
   embeddings from different versions or variants in the same downstream
   task**: the 128-channel feature spaces are independently learned and not
   interchangeable. Pick one ``(dataset_version, dataset_variant)`` pair per
   project — list the choices with ``geotessera info`` — and see
   :ref:`dataset-versions` for full details and the CLI/Python flags.

Overview
--------

GeoTessera offers two ways in:

1. **Stream from the zarr store** (:class:`~geotessera.store.GeoTesseraZarr`):
   read points, regions, and patches straight from the public cloud store,
   dequantised on their native UTM grid, with nothing downloaded up front.
   This is the recommended interface for most work — see the
   :doc:`zarr_quickstart`.
2. **Download tiles** (:class:`~geotessera.GeoTessera`): fetch 0.1-degree
   tiles to disk and export them as numpy arrays or georeferenced GeoTIFF
   files, for offline reuse and GIS integration — see the
   :doc:`quickstart`.

Key Features
------------

* Read points, regions, and fixed-size patches through the Zarr API.
* Export selected embedding bands as GeoTIFFs on their native UTM grids.
* Download individual NPY or GeoTIFF tiles for offline use.
* Create PCA visualizations or web maps directly from selected embedding bands.
* Reuse completed web maps and resume individual tile downloads.
* Select a dataset version, variant, year, and published embedding depth.

Installation
------------

Requires Python 3.12 or later. Install GeoTessera using pip::

    pip install geotessera

For development installation::

    git clone https://github.com/ucam-eo/geotessera
    cd geotessera
    pip install -e .

Quick Start
-----------

Stream one embedding from the zarr store::

    from geotessera import GeoTesseraZarr

    gt = GeoTesseraZarr()
    vec, status = gt.probe(0.12, 52.20, year=2024)   # (128,) float32, 'valid'

The :doc:`zarr_quickstart` continues from here to regions, streams,
patches, GeoTIFF exports, and matryoshka depths.

Export a region or create a web map from Zarr::

    geotessera download --bbox '-3.0,53.4,-2.9,53.5' --year 2024 --output region/
    geotessera webmap --bbox '-3.0,53.4,-2.9,53.5' --year 2024 --output map/ --serve

See :doc:`cli_reference` for options and restart behavior. The following
examples use individual tiles with ``--source tiles``, which are
deprecated and will be removed. v1.1 tiles exist only for the
``cambridge`` variant; commands without ``--dataset-variant cambridge``
use it with a warning.

Check data availability first::

    # Map the default v1.1 dclimate dataset from its tile registry
    geotessera coverage --output coverage_map.png

    # Map tile coverage (creates PNG map, JSON data, and interactive HTML globe)
    geotessera coverage --dataset-variant cambridge --output coverage_map.png
    # Creates: coverage_map.png, coverage.json, globe.html

    # View tile coverage for a specific year
    geotessera coverage --dataset-variant cambridge --year 2024

    # Check coverage for a single country with precise boundary outline
    geotessera coverage --country "United Kingdom"
    geotessera coverage --country uk  # Also accepts country codes

Download embeddings in your preferred format::

    # Download as GeoTIFF (default, georeferenced, ready for GIS)
    geotessera download --source tiles --bbox "-0.2,51.4,0.1,51.6" --year 2024 --output ./london_tiffs --bands 1,2,3

    # Download as quantized numpy arrays (for analysis, includes scales and landmask TIFFs)
    geotessera download --source tiles --bbox "-0.2,51.4,0.1,51.6" --format npy --year 2024 --output ./london_arrays
    # NPY format includes: quantized .npy, _scales.npy, and landmask .tiff files

    # Download tiles within a country's bounding box.
    geotessera download --source tiles --country "United Kingdom" --year 2024 --output ./uk_tiles

    # Download tiles from a region file (supports GeoJSON, Shapefile, or URLs)
    geotessera download --source tiles --region-file example/CB.geojson --year 2024 --output ./cambridge
    geotessera download --source tiles --region-file https://example.com/region.geojson --year 2024 --output ./remote_region


Python API usage::

    from geotessera import GeoTessera

    # Initialize client
    gt = GeoTessera()

    # Method 1: Fetch a single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
    print(f"CRS: {crs}")  # UTM projection

    # Method 2: Fetch all tiles in a bounding box
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}")

    # Method 3: Sample embeddings at specific point locations
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]  # (lon, lat) tuples
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled embeddings shape: {embeddings.shape}")  # (3, 128)

    # Export as GeoTIFF files with preserved UTM projections
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./output",
        bands=[0, 1, 2]  # Export first 3 bands only
    )

Create PCA visualizations and web maps::

    # Create PCA mosaic from GeoTIFFs
    geotessera visualize ./london_tiffs pca_mosaic.tif

    # Create web tiles and serve interactively
    geotessera webmap pca_mosaic.tif --serve

Architecture Overview
---------------------

Coordinate System and Tile Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings use a **0.1-degree grid system**:

* **Tile size**: Each tile covers 0.1° × 0.1° (approximately 11km × 11km at the equator)
* **Tile naming**: Tiles are named by their **center coordinates** (e.g., ``grid_0.15_52.05``)
* **Tile bounds**: A tile at center (lon, lat) covers [lon ± 0.05°, lat ± 0.05°]
* **Resolution**: 10m per pixel (variable pixels per tile depending on latitude)

File Structure and Downloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you request embeddings, GeoTessera downloads files over plain HTTPS from
the public Source Cooperative repository
(``https://data.source.coop/tessera/tessera``, fronted by Cloudflare) into
your chosen output directory, where they persist for re-use:

**Embedding Files** (via ``fetch_embedding``):

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):

   * Shape: ``(height, width, 128)``
   * Data type: int8 (quantized for storage efficiency)
   * Contains the compressed embedding values

2. **Scale files** (``grid_X.XX_Y.YY_scales.npy``):

   * Shape: ``(height, width)`` or ``(height, width, 128)``
   * Data type: float32
   * Contains scale factors for dequantization

3. **Dequantization**: ``final_embedding = quantized_embedding * scales``

4. **Persistent Storage**: Files are downloaded into your output directory and skipped on rerun, so interrupted downloads resume cleanly

**Landmask Files** (with CRS and masks for GeoTIFF export):

* **Landmask tiles** (``grid_X.XX_Y.YY.tiff``):

  * Provide UTM projection information
  * Define precise geospatial transforms
  * Contain land/water masks
  * Cached alongside the embedding tiles for re-use

The geotessera CLI can also export these into GeoTIFF format with each band
dequantised into 128-bands and with the GeoTIFF CRS metadata intact.

Data Flow
~~~~~~~~~

::

    User Request (lat/lon bbox, dataset_version, dataset_variant)
        ↓
    Per-version Manifest Lookup (filter manifest.parquet by year/lon/lat/variant)
        ↓
    HTTPS Downloads from Source Cooperative (with integrity checks on the wire)
        ├── embedding.npy (int8 quantized) → output_dir
        └── embedding_scales.npy (float32 scale factors) → output_dir
        ↓
    Dequantization at use time: float = quantized.astype('f4') * scales
        ↓
    Output Format
        ├── NumPy arrays + tessera_metadata.json sidecar → Direct analysis
        └── GeoTIFF (with TESSERA_DATASET_VERSION/VARIANT tags) → GIS integration

**Storage Note**: Manifest + landmask Parquets (tens to a couple of hundred
MB per dataset) are cached under ``~/.cache/geotessera/`` in per-dataset
subdirectories. Embedding tiles land in the user-specified ``--output``
directory (resumable across runs via existence checks).

Manifest System
~~~~~~~~~~~~~~~

GeoTessera uses a Parquet-based per-version manifest for efficient data access:

* **One manifest per dataset** — a ``(version, variant)`` pair with its own
  ``npy/`` directory:
  ``data.source.coop/tessera/tessera/npy/{v1,v1.1-cam,v2-2B-L~beta1}/manifest.parquet``.
  Each carries the file-scan inventory schema (``year, lon, lat, grid_size,
  scales_size, grid_path, ...``) plus explicit ``version`` and ``variant``
  columns.
* **Fast queries**: pandas/GeoPandas DataFrames with spatial R-tree on lon/lat
* **Block-based queries**: Internal 5×5° geographic blocks keep region lookups O(blocks)
* **Conditional fetches**: cached manifests are revalidated with an
  ``If-Modified-Since`` conditional GET keyed on the cached file's mtime —
  refetches only happen when the server copy has actually changed;
  otherwise the server returns 304 with no body.
* **Integrity checking**: Every download is verified against the response
  ``Content-Length``, and against a streamed MD5 whenever the server's
  ``ETag`` is a content MD5 (single-part uploads).

The manifest can be loaded from multiple sources:

1. **Default remote** (recommended, downloads and caches automatically per version)
2. **Local file** (via the ``registry_path`` Python parameter)
3. **Local directory** (via ``--registry-dir`` on the CLI or the ``registry_dir`` Python parameter; looks for ``manifest.parquet``)
4. **Custom URL** (via the ``registry_url`` Python parameter)

Maintainers can regenerate the manifests at any time by scanning the Source
Cooperative repository with ``geotessera-registry s3scan`` — see
:doc:`maintenance`.

Understanding Tessera Embeddings
--------------------------------

Each embedding tile:

* Covers a 0.1° × 0.1° area (approximately 11km × 11km at equator)
* Contains 128 channels of learned features per pixel
* Represents patterns from a full year of satellite observations
* Is stored in quantized format for efficient transmission and storage

The 128 channels capture various environmental features learned by the
Tessera foundation model, including vegetation patterns, water bodies,
urban structures, and seasonal changes.

.. _dataset-versions:

Dataset Versions and Variants
-----------------------------

GeoTessera ships embeddings under two orthogonal axes:

* **dataset version** — the trained Tessera model (``1.0`` or ``1.1``).
  Different versions have *different 128-channel feature spaces*: a feature
  vector from one version is **not comparable** to a vector from another.
* **dataset variant** — for a given version, a separate inference run.
  Embeddings from different variants do not interoperate either, even
  within one version. Each version has a default variant, selected when
  ``--dataset-variant`` is omitted.

Each variant is published in one or more formats: NPY tiles under
``npy/``, a Zarr store under ``zarr/`` on
``data.source.coop/tessera/tessera``, or an Icechunk repository. List them
with ``geotessera info``:

+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+
| ``version`` | ``variant``          | NPY (``npy/``)       | Zarr (``zarr/``)   | Icechunk                                           | Years     |
+=============+======================+======================+====================+====================================================+===========+
| ``1.0``     | ``vultr`` (default)  | ``v1/``              | ``v1/``            | —                                                  | 2017–2025 |
+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+
| ``1.1``     | ``dclimate``         | —                    | —                  | ``s3://tessera-embeddings/v1.1/dclimate.icechunk`` | 2017–2025 |
|             | (default)            |                      |                    |                                                    |           |
+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+
| ``1.1``     | ``cambridge``        | ``v1.1-cam/``        | ``v1.1/``          | —                                                  | 2015–2025 |
+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+
| ``2.0``     | ``2B-L~beta1``       | ``v2-2B-L~beta1/``   | ``v2-2B-L~beta1/`` | —                                                  | 2017–2025 |
|             | (default)            |                      |                    |                                                    |           |
+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+
| ``2.0``     | ``2B-L~beta2``       | ``v2-2B-L~beta2/``   | ``v2-2B-L~beta2/`` | —                                                  | 2017–2025 |
+-------------+----------------------+----------------------+--------------------+----------------------------------------------------+-----------+

``1.0`` / ``vultr`` is frozen. ``1.1`` / ``dclimate`` is the complete
global v1.1 run. ``1.1`` / ``cambridge`` holds the Cambridge test
embeddings. The ``2.0`` variants are experimental betas.

The default version is ``v1.1``. Streamed reads use ``dclimate``. NPY tiles
of v1.1 exist only for ``cambridge``, so ``GeoTessera`` and ``download
--format npy`` fall back to it with a warning; select it explicitly with
``--dataset-variant cambridge``.

.. note::

   NPY tiles are deprecated and will be removed. Use Zarr or Icechunk.

Which one should I use?
~~~~~~~~~~~~~~~~~~~~~~~

**Prefer ``1.1`` / ``dclimate`` for new projects.** Use ``1.1`` /
``cambridge`` only when a workflow requires NPY tiles, and then use it
throughout. Use ``1.0`` / ``vultr`` only to reproduce prior work.

.. warning::

   **Do not mix embeddings from different ``(version, variant)`` pairs in
   the same analysis.** Each (version, variant) is a distinct learned
   representation:

   * Cosine similarity, classification heads, clustering, PCA, or any
     downstream model trained on one set produces meaningless results if
     fed vectors from another.
   * Even tiles at the same lat/lon for the same year carry *different
     numeric values* across versions/variants. The grid geometry matches;
     the channel semantics do not.

   Each client reads a single ``(version, variant)``. Downloads record it
   (see `What gets recorded`_), and GeoTessera refuses to download into a
   directory holding another dataset or to merge GeoTIFFs of different
   datasets.

Specifying version + variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CLI** — every data-fetching command (``download``, ``webmap``,
``coverage``, ``info``) accepts both flags::

    geotessera download --source tiles \
        --dataset-version v1.1 \
        --dataset-variant cambridge \
        --region-file area.geojson \
        --year 2024 \
        --output ./tiles

``--dataset-version`` accepts either form: ``v1`` and ``1.0`` are aliases,
as are ``v1.1``/``1.1`` and ``v2``/``2.0``. The internal normalised form
(used in manifests and the metadata sidecar) is ``1.0`` / ``1.1`` / ``2.0``;
the ``npy/`` tree directory is derived from the (version, variant) pair
(``v1``, ``v1.1-cam``, ``v2-2B-L~beta1``).

``--dataset-variant`` defaults to the version's default variant
(``vultr`` for 1.0, ``dclimate`` for 1.1, ``2B-L~beta1`` for 2.0). If
that variant is not published in the requested format, the first variant
that is is used with a warning: v1.1 NPY tiles come from ``cambridge``.
List every dataset and its formats with ``geotessera info``.

**Python API**::

    from geotessera import GeoTessera, GeoTesseraZarr
    from geotessera.registry import zarr_store_url

    # Recommended: stream the default v1.1 dclimate dataset
    gt = GeoTesseraZarr()
    print(gt.dataset.name)                      # 1.1-dclimate

    # Stream another dataset
    gt = GeoTesseraZarr(zarr_store_url('v1.1', 'cambridge'))

    # NPY tiles (deprecated); v1.1 tiles exist only for cambridge
    gt = GeoTessera(dataset_version='v1.1', dataset_variant='cambridge')
    print(gt.dataset_version, gt.dataset_variant)

What gets recorded
~~~~~~~~~~~~~~~~~~

Every download writes a ``tessera_metadata.json`` file in its output
directory naming the dataset version and variant, and every exported
GeoTIFF carries ``TESSERA_DATASET_VERSION``,
``TESSERA_DATASET_VERSION_PATH`` and ``TESSERA_DATASET_VARIANT`` tags.
Streamed GeoTIFFs also record the store in ``TESSERA_SOURCE``.

GeoTessera reads these records to keep datasets apart:

* A download into a directory whose ``tessera_metadata.json`` names
  another dataset fails.
* Merging GeoTIFFs whose tags name different datasets fails, including
  RGB mosaics and web maps.

A directory without ``tessera_metadata.json``, a GeoTIFF without dataset
tags, and a ``--store-url`` store that is not a published dataset are not
checked.

Coverage compositing
~~~~~~~~~~~~~~~~~~~~

For situations where you want to *visualise* multiple versions/variants
together (without combining them analytically), ``geotessera coverage
--by-source`` renders each ``(version, variant)`` group in its own colour
on the same map and produces an interactive ``globe.html`` with per-dataset
layer toggles. See the CLI reference for the full flag set.


Data Organization
-----------------

**Remote Server Structure** (Source Cooperative)::

    https://data.source.coop/tessera/tessera/
    ├── npy/                                         # NPY embeddings + scales
    │   │                                            # (one dir per (version, variant) dataset)
    │   ├── v1/                                      # 1.0 — all variants share this dir
    │   │   ├── manifest.parquet                     # Per-dataset tile manifest
    │   │   └── 2024/grid_0.15_52.05/grid_0.15_52.05{,_scales}.npy
    │   ├── v1.1-cam/                                # 1.1 / cambridge
    │   │   ├── manifest.parquet
    │   │   └── 2024/grid_0.15_52.05/grid_0.15_52.05{,_scales}.npy
    │   └── v2-2B-L~beta1/                           # 2.0 / 2B-L~beta1 (beta)
    │       └── 2024/grid_0.15_52.05/grid_0.15_52.05{,_scales}.npy
    ├── landmasks/                                   # Landmask TIFFs (per version)
    │   ├── v1/
    │   │   ├── landmasks.parquet                    # Landmask manifest
    │   │   └── grid_0.15_52.05.tiff
    │   ├── v1.1/
    │   │   ├── landmasks.parquet
    │   │   └── grid_0.15_52.05.tiff
    │   └── v2/
    │       ├── landmasks.parquet
    │       └── grid_0.15_52.05.tiff
    └── zarr/                                        # Cloud-native zarr stores
        ├── v1/                                      # 1.0 / vultr: 60 UTM zone groups + RGB pyramid
        ├── v1.1/                                    # 1.1 / cambridge
        ├── v2-2B-L~beta1/                           # v2 beta, with embeddings_d4/d16
        └── v2-2B-L~beta2/

The ``1.1`` / ``dclimate`` dataset is an Icechunk repository at
``s3://tessera-embeddings/v1.1/dclimate.icechunk``, with its tile registry
under ``s3://tessera-embeddings/v1.1/dclimate.registry/``.

Each ``manifest.parquet`` is scoped to one dataset — the npy/ directory
name encodes the ``(version, variant)`` pair. The client downloads only the
manifest for its dataset directory and filters by ``dataset_variant`` on
load. Landmasks are a property of the 0.1° grid, so they stay keyed by
plain version (``landmasks/v1.1/`` serves every 1.1 variant).

**Local Mirror Structure** (when downloading via ``geotessera download --source tiles``)::

    output_dir/
    ├── tessera_metadata.json                        # version/variant provenance
    ├── global_0.1_degree_representation/            # Always this bare name,
    │   └── 2024/grid_0.15_52.05/                    # regardless of variant.
    │       ├── grid_0.15_52.05.npy
    │       └── grid_0.15_52.05_scales.npy
    └── global_0.1_degree_tiff_all/
        └── grid_0.15_52.05.tiff

**Local Cache Structure** (mirrors the remote trees: manifests are cached
per dataset directory, landmask registries per version)::

    ~/.cache/geotessera/                             # Default cache location
    ├── v1/
    │   ├── manifest.parquet                         # 1.0 embeddings manifest
    │   └── landmasks.parquet                        # v1 landmask registry
    ├── v1.1-cam/
    │   └── manifest.parquet                         # 1.1/cambridge manifest
    ├── v1.1/
    │   └── landmasks.parquet                        # v1.1 landmask registry
    ├── v2-2B-L~beta1/
    │   └── manifest.parquet                         # 2.0 beta manifest
    └── v2/
        └── landmasks.parquet                        # v2 landmask registry

Cached manifests are revalidated with conditional ``If-Modified-Since``
requests keyed on the cached file's modification time: the client refetches
only when the server copy has actually changed, and the server returns
``304 Not Modified`` (zero body bytes) otherwise.

Embeddings are organized by:

* **Year**: 2017–2025 for ``1.0/vultr``, ``1.1/dclimate`` and
  ``2.0/2B-L~beta1``; 2015–2025 for ``1.1/cambridge``
* **Location**: Global 0.1-degree grid system (same grid across all versions)
* **Format**: NumPy arrays with shape (height, width, 128) after dequantisation

Cache Configuration
-------------------

Control where the Parquet registry is cached::

    from geotessera import GeoTessera

    # Use custom cache directory for registry
    gt = GeoTessera(cache_dir="/path/to/cache")

    # Use default cache location (recommended)
    gt = GeoTessera()

Or via CLI::

    # Specify custom cache directory
    geotessera download --source tiles --cache-dir /path/to/cache ...

    # Use default cache location
    geotessera download --source tiles ...

Default cache locations (when not specified):

* **Linux/macOS**: ``~/.cache/geotessera/``
* **Windows**: ``%LOCALAPPDATA%/geotessera/``

Documentation Sections
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   zarr_quickstart
   quickstart
   architecture
   tutorials
   cli_reference
   maintenance

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   GitHub Repository <https://github.com/ucam-eo/geotessera>
   Tessera Model <https://github.com/ucam-eo/tessera>
   Issue Tracker <https://github.com/ucam-eo/geotessera/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
