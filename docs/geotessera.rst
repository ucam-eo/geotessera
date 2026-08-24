geotessera package
==================

The GeoTessera package provides streamlined access to Tessera geospatial embeddings through a clean Python API and comprehensive CLI.

Package Overview
----------------

.. automodule:: geotessera
   :members:
   :show-inheritance:
   :undoc-members:

API Reference
-------------

.. _geotessera-core:

:mod:`geotessera.core` -- Core Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main interface for accessing Tessera embeddings. Contains the primary :class:`~geotessera.GeoTessera` class with methods for fetching embeddings with CRS information and exporting to various formats while preserving native UTM projections.

**Key Features:**

* :meth:`~geotessera.GeoTessera.fetch_embedding` - Fetch a single embedding tile with CRS and transform
* :meth:`~geotessera.GeoTessera.fetch_embeddings` - Fetch multiple tiles in a bounding box with projection info
* :meth:`~geotessera.GeoTessera.export_embedding_geotiff` - Export single embedding as GeoTIFF with native UTM
* :meth:`~geotessera.GeoTessera.export_embedding_geotiffs` - Export multiple embeddings as GeoTIFF files

.. automodule:: geotessera.core
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-registry:

:mod:`geotessera.registry` -- Registry Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registry management for efficient data discovery and access. Handles the block-based registry system, lazy loading, and metadata management.

**Key Features:**

* :class:`~geotessera.registry.Registry` - Main registry class for data discovery
* :func:`~geotessera.registry.tile_to_bounds` - Get geographic bounds of a tile
* :func:`~geotessera.registry.tile_from_world` - Convert geographic to tile coordinates
* :func:`~geotessera.registry.block_from_world` - Get block coordinates for a tile

.. automodule:: geotessera.registry
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-visualization:

:mod:`geotessera.visualization` -- Visualization Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualization utilities for creating maps, web tiles, and interactive visualizations from GeoTIFF files.

**Key Features:**

* :func:`~geotessera.visualization.visualize_global_coverage` - Create global coverage maps
* :func:`~geotessera.visualization.create_rgb_mosaic` - Combine multiple GeoTIFFs into a mosaic
* :func:`~geotessera.web.geotiff_to_web_tiles` - Generate web tiles for interactive maps
* :func:`~geotessera.web.create_coverage_summary_map` - Create coverage summary visualizations

.. automodule:: geotessera.visualization
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-store:

:mod:`geotessera.store` -- Zarr Store Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cloud-native access to Tessera embeddings via Zarr v3 format, with automatic UTM zone routing, point sampling, and region reading. Implements the ``geoemb:`` convention for geospatial embedding stores.

**Key Features:**

* :class:`~geotessera.store.GeoTesseraZarr` - Cloud-native Zarr store for streaming access without downloads
* :meth:`~geotessera.store.GeoTesseraZarr.sample_points` - Sample embeddings at specific coordinates
* :meth:`~geotessera.store.GeoTesseraZarr.read_region` - Read rectangular regions as mosaics
* :meth:`~geotessera.store.GeoTesseraZarr.read_patch` - Read a fixed-size patch centred on a point, merged across UTM zones
* :meth:`~geotessera.store.GeoTesseraZarr.open_zone` - Open UTM zone as xarray Dataset
* :meth:`~geotessera.store.GeoTesseraZarr.probe` - Sample a point, reporting why when there is no value

.. automodule:: geotessera.store
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-tiles:

:mod:`geotessera.tiles` -- Tile Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format-agnostic tile abstraction supporting both GeoTIFF and NPY format tiles with automatic format detection.

**Key Features:**

* :class:`~geotessera.tiles.Tile` - Format-agnostic embedding tile wrapper
* :func:`~geotessera.tiles.discover_tiles` - Auto-detect and discover tiles in a directory
* :func:`~geotessera.tiles.discover_formats` - Discover all available tile formats

.. automodule:: geotessera.tiles
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-cli:

:mod:`geotessera.cli` -- Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive command-line interface providing download, visualization, and serving capabilities.

.. automodule:: geotessera.cli
   :members:
   :show-inheritance:
   :undoc-members:

Examples
--------

Basic Usage
~~~~~~~~~~~

Initialize and fetch embeddings::

    from geotessera import GeoTessera

    # Initialize client
    gt = GeoTessera()

    # Method 1: Fetch single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"CRS: {crs}")  # Native UTM projection

    # Method 2: Fetch region with projection info
    bbox = (-0.2, 51.4, 0.1, 51.6)

    # Step 1: Get list of tiles in region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Fetch the tiles (returns generator for memory efficiency)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}, CRS: {crs}")

    # Method 3: Sample at specific points
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled {len(points)} points: {embeddings.shape}")

Export to GeoTIFF
~~~~~~~~~~~~~~~~~

Export embeddings for GIS use with preserved projections::

    # Step 1: Get list of tiles to export
    bbox = (-0.2, 51.4, 0.1, 51.6)
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Export all bands with native UTM projections
    files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./output",
    )

    # Export specific bands
    rgb_files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./rgb_output",
        bands=[0, 1, 2]  # Each tile preserves its native UTM projection
    )

    # Export single tile
    single_file = gt.export_embedding_geotiff(
        lon=0.15, lat=52.05,
        output_path="./single_tile.tif",
        year=2024,
        bands=[10, 20, 30]
    )

Create Visualizations
~~~~~~~~~~~~~~~~~~~~~

Generate visualizations::

    from geotessera.visualization import (
        create_rgb_mosaic,
        visualize_global_coverage
    )
    
    # Create RGB mosaic
    create_rgb_mosaic(
        geotiff_paths=rgb_files,
        output_path="mosaic.tif",
        bands=(0, 1, 2)
    )
    
    # Create coverage map
    visualize_global_coverage(
        tessera_client=gt,
        output_path="coverage.png",
        year=2024
    )