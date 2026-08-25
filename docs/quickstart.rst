Quick Start Guide
=================

This guide covers the tile-download interface: fetching embeddings to
disk and exporting them for GIS use. To stream embeddings without
downloading files, start with the :doc:`zarr_quickstart` instead.

.. tip::

   **Pick your Tessera version first.** Several model versions are
   published on the Source Cooperative repository: the frozen ``1.0``
   default, the newer ``1.1``, and experimental ``2.0`` betas. List
   every available dataset (and each version's default variant)
   with ``geotessera info``.
   Add ``--dataset-version v1.1 --dataset-variant cambridge`` to every
   command in this guide if that's the line you want. Never mix versions or
   variants within the same downstream task — they are independently learned
   feature spaces. See the main :ref:`dataset-versions` section for details.

Installation
------------

Requires Python 3.12 or later. Install GeoTessera using pip::

    pip install geotessera

Verify the installation::

    geotessera --help

Step 1: Check Data Availability
--------------------------------

Before downloading embeddings, we recommend check what data is available for your region of interest.

Generate coverage visualizations (PNG map, JSON data, and interactive HTML globe)::

    geotessera coverage --output global_coverage.png
    # Creates three files:
    # 1. global_coverage.png - Static world map with tiles
    # 2. coverage.json - JSON data with global coverage information
    # 3. globe.html - Interactive 3D globe visualization

This creates a world map showing all available embedding tiles. By default, it uses multi-year color coding:

- **Green**: All available years present for this tile
- **Blue**: Only the latest year available for this tile
- **Orange**: Partial years coverage (some combination of years)

When you specify a country or region file, its boundary is outlined on
the map.

For a specific region (recommended)::

    geotessera coverage --region-file study_area.geojson
    # Next step: geotessera download --region-file study_area.geojson --output tiles/
    
    # You can also use remote URLs directly:
    geotessera coverage --region-file https://example.com/region.geojson

    # Or check coverage for a specific country (with precise boundary outline):
    geotessera coverage --country "United Kingdom"
    # Next step: geotessera download --country "United Kingdom" --output tiles/

For a specific year::

    geotessera coverage --year 2024 --output coverage_2024.png

You can customize the visualization::

    geotessera coverage \
        --region-file area.geojson \
        --tile-alpha 0.3

Step 2: Download Embeddings
----------------------------

GeoTessera supports two output formats:

- **tiff**: Georeferenced GeoTIFF files (default, best for GIS) - fully dequantized and ready to use
- **npy**: Quantized numpy arrays with scales and landmask TIFFs (for advanced analysis and storage efficiency)

For cloud-native zarr access without downloading files, see
:ref:`zarr-access` below.

Download as GeoTIFF (Recommended for GIS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download embeddings for London as GeoTIFF files::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_tiles
    # Next step: geotessera visualize ./london_tiles pca_mosaic.tif

This downloads all 128 bands with LZW compression.

Download specific bands only::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --bands "0,1,2" \
        --year 2024 \
        --output ./london_rgb
    # Next step: geotessera visualize ./london_rgb pca_mosaic.tif

Download by country name::

    geotessera download \
        --country "United Kingdom" \
        --year 2024 \
        --output ./uk_tiles
    # Next step: geotessera visualize ./uk_tiles pca_mosaic.tif

    # Or use short country codes
    geotessera download \
        --country "GB" \
        --year 2024 \
        --output ./uk_tiles
    # Next step: geotessera visualize ./uk_tiles pca_mosaic.tif

Download using a region file::

    # Create a GeoJSON file defining your region
    cat > cambridge.json << EOF
    {
      "type": "Polygon",
      "coordinates": [[
        [-0.2, 51.9], [0.3, 51.9], [0.3, 52.3], [-0.2, 52.3], [-0.2, 51.9]
      ]]
    }
    EOF
    
    geotessera download \
        --region-file cambridge.json \
        --year 2024 \
        --output ./cambridge_tiles
    # Next step: geotessera visualize ./cambridge_tiles pca_mosaic.tif

Download as NumPy Arrays (For Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download quantized numpy arrays with scales and landmask TIFFs::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --format npy \
        --year 2024 \
        --output ./london_arrays

This creates the registry directory structure:

- ``global_0.1_degree_representation/{year}/grid_{lon}_{lat}/grid_{lon}_{lat}.npy`` - Quantized embeddings (int8)
- ``global_0.1_degree_representation/{year}/grid_{lon}_{lat}/grid_{lon}_{lat}_scales.npy`` - Scale factors (float32)
- ``global_0.1_degree_tiff_all/grid_{lon}_{lat}.tiff`` - Landmask TIFF with CRS and transform

To dequantize: ``dequantized = quantized.astype(np.float32) * scales``

Step 3: Work with the Data
---------------------------

Python API Examples
~~~~~~~~~~~~~~~~~~~

Initialize the client::

    from geotessera import GeoTessera
    import numpy as np
    
    gt = GeoTessera()

Fetch a single embedding tile with CRS information::

    # Fetch embedding for Cambridge, UK (note: lon, lat order)
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
    print(f"Data type: {embedding.dtype}")  # float32
    print(f"CRS: {crs}")  # UTM projection from landmask
    print(f"Transform: {transform}")  # Geospatial transform
    print(f"Value range: [{embedding.min():.2f}, {embedding.max():.2f}]")

Fetch multiple tiles in a bounding box::

    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)

    # Step 1: Get list of tiles in the region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Fetch the tiles (returns a generator for memory efficiency)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding_array, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding_array.shape}")
        print(f"  CRS: {crs}")

        # Compute basic statistics
        mean_values = np.mean(embedding_array, axis=(0, 1))  # Mean per channel
        print(f"  Mean of first 5 channels: {mean_values[:5]}")

Sample embeddings at specific points::

    # Define points of interest (lon, lat tuples)
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]

    # Sample embeddings at these points (automatically downloads tiles if needed)
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled embeddings shape: {embeddings.shape}")  # (3, 128)

    # With metadata
    embeddings, metadata = gt.sample_embeddings_at_points(points, year=2024, include_metadata=True)
    for i, meta in enumerate(metadata):
        print(f"Point {i}: tile ({meta['tile_lon']}, {meta['tile_lat']}), pixel ({meta['pixel_row']}, {meta['pixel_col']})")

Export embeddings to GeoTIFF::

    # Step 1: Get list of tiles in the region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Export as GeoTIFF files
    files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./output",
        bands=[10, 30, 50],  # Custom band selection
        compress="lzw"
    )
    print(f"Created {len(files)} GeoTIFF files")

Working with Downloaded NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and analyze downloaded numpy arrays (NPY format)::

    import numpy as np
    from pathlib import Path
    from geotessera import dequantize_embedding

    # NPY format uses the registry directory structure
    base_dir = Path("london_arrays")
    year = 2024

    # Example: Load a specific tile
    lon, lat = 0.15, 52.05
    grid_name = f"grid_{lon:.2f}_{lat:.2f}"

    # Load quantized embedding and scales
    embedding_path = base_dir / "global_0.1_degree_representation" / str(year) / grid_name / f"{grid_name}.npy"
    scales_path = base_dir / "global_0.1_degree_representation" / str(year) / grid_name / f"{grid_name}_scales.npy"
    landmask_path = base_dir / "global_0.1_degree_tiff_all" / f"{grid_name}.tiff"

    quantized = np.load(embedding_path)  # int8
    scales = np.load(scales_path)  # float32

    # Dequantize using the helper function
    embedding = dequantize_embedding(quantized, scales)

    print(f"Tile ({lat}, {lon}):")
    print(f"  Shape: {embedding.shape}")
    print(f"  Data type: {embedding.dtype}")  # float32
    print(f"  Mean per band (first 5): {np.mean(embedding, axis=(0,1))[:5]}")

    # Extract center pixel features
    center_pixel = embedding[embedding.shape[0]//2, embedding.shape[1]//2, :]
    print(f"  Center pixel features (first 5): {center_pixel[:5]}")

    # Load landmask for CRS information (optional)
    import rasterio
    with rasterio.open(landmask_path) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Transform: {src.transform}")

Step 4: Create PCA Visualizations
----------------------------------

Create a PCA Mosaic
~~~~~~~~~~~~~~~~~~~

From GeoTIFF files, create a PCA visualization::

    geotessera visualize ./london_tiles pca_mosaic.tif
    # Next step: geotessera webmap pca_mosaic.tif --serve

This combines all embedding data across tiles, applies PCA transformation, and creates a unified RGB mosaic from the first 3 principal components. This eliminates tiling artifacts and provides consistent visualization across the region.

Customize the PCA visualization::

    # Use histogram equalization for maximum contrast
    geotessera visualize ./london_tiles pca_balanced.tif --balance histogram

    # Use adaptive scaling based on variance
    geotessera visualize ./london_tiles pca_adaptive.tif --balance adaptive

    # Custom percentile range for outlier-robust scaling
    geotessera visualize ./london_tiles pca_custom.tif --percentile-low 5 --percentile-high 95

    # Compute more components for research (still uses first 3 for RGB)
    geotessera visualize ./london_tiles pca_research.tif --n-components 10

Create Interactive Web Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate web tiles and viewer from your PCA mosaic::

    geotessera webmap pca_mosaic.tif --serve

This automatically:
1. Reprojects the mosaic for web viewing if needed
2. Generates web tiles at multiple zoom levels
3. Creates an HTML viewer
4. Starts a local web server and opens in your browser

Customize web tile generation::

    # Custom zoom levels and output directory
    geotessera webmap pca_mosaic.tif --min-zoom 6 --max-zoom 18 --output webmap/

    # Add region boundary overlay
    geotessera webmap pca_mosaic.tif --region-file study_area.geojson --serve

    # Force regeneration of existing tiles
    geotessera webmap pca_mosaic.tif --force --serve

Coverage Maps
~~~~~~~~~~~~~

Create coverage maps using the coverage command to visualize data availability::

    # Generate coverage map for your downloaded tiles
    geotessera coverage --output my_coverage.png
    
    # Or generate coverage for a specific region
    geotessera coverage --region-file area.geojson --output area_coverage.png

Step 5: Advanced Workflows
---------------------------

Python Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Complete analysis workflow::

    from geotessera import GeoTessera
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize client
    gt = GeoTessera()
    
    # Define region of interest
    bbox = (-0.15, 52.15, 0.0, 52.25)  # Cambridge area
    
    # Fetch embeddings
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    embeddings = gt.fetch_embeddings(tiles_to_fetch)
    
    # Analyze each tile
    results = []
    for year, tile_lon, tile_lat, embedding, crs, transform in embeddings:
        # Compute statistics
        mean_per_band = np.mean(embedding, axis=(0, 1))
        std_per_band = np.std(embedding, axis=(0, 1))

        results.append({
            'lat': tile_lat,
            'lon': tile_lon,
            'mean_band_50': mean_per_band[50],
            'std_band_50': std_per_band[50],
            'total_variance': np.var(embedding),
            'crs': str(crs)
        })
    
    # Print results
    for result in results:
        print(f"Tile ({result['lat']:.2f}, {result['lon']:.2f}): "
              f"Band 50 mean={result['mean_band_50']:.3f}, "
              f"variance={result['total_variance']:.3f}")
    
    # Export interesting tiles as GeoTIFF
    threshold = np.median([r['mean_band_50'] for r in results])
    selected_tiles = [r for r in results if r['mean_band_50'] > threshold]
    
    print(f"Exporting {len(selected_tiles)} tiles above threshold")

    # Prepare tiles for export
    tiles_to_export = [(2024, tile['lon'], tile['lat']) for tile in selected_tiles]

    files = gt.export_embedding_geotiffs(
        tiles_to_fetch=tiles_to_export,
        output_dir="./selected_tiles",
        bands=[40, 50, 60]  # Bands around band 50
    )
    
    # Create PCA visualization from selected tiles
    # CLI: geotessera visualize ./selected_tiles pca_selected.tif
    # CLI: geotessera webmap pca_selected.tif --serve

Mixed Format Workflow
~~~~~~~~~~~~~~~~~~~~~

Use both numpy and GeoTIFF formats in the same workflow::

    from geotessera import GeoTessera
    
    gt = GeoTessera()
    bbox = (-0.1, 51.5, 0.0, 51.55)
    
    # Step 1: Analyze with numpy arrays
    print("Analyzing embeddings...")
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    tiles = gt.fetch_embeddings(tiles_to_fetch)
    
    # Custom analysis to select interesting tiles
    selected_tiles = []
    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        # Example: select tiles with high variance in band 64
        band_64_var = np.var(embedding[:, :, 64])
        if band_64_var > 0.5:  # Threshold
            selected_tiles.append((year, tile_lon, tile_lat))

    print(f"Selected {len(selected_tiles)} interesting tiles")

    # Step 2: Export selected tiles as GeoTIFF
    tiles_to_export = selected_tiles

    all_files = gt.export_embedding_geotiffs(
        tiles_to_fetch=tiles_to_export,
        output_dir="./interesting_tiles",
        bands=[60, 64, 68]  # Bands around interesting band 64
    )
    
    # Step 3: Create PCA visualization from selected tiles
    print("Creating PCA visualization...")
    # Use CLI for PCA visualization:
    # geotessera visualize ./interesting_tiles pca_interesting.tif
    # geotessera webmap pca_interesting.tif --serve
    
    print("Use CLI commands to create PCA visualization and web viewer")

.. _zarr-access:

Cloud-Native Zarr Access
~~~~~~~~~~~~~~~~~~~~~~~~~

For analysis without downloading files, stream from the zarr store::

    from geotessera import GeoTesseraZarr

    gt = GeoTesseraZarr()
    X = gt.sample_points([(-2.97, 53.44), (0.15, 52.05)], year=2025)   # (2, 128)
    mosaic, transform, crs = gt.read_region((-3.0, 53.4, -2.9, 53.5), year=2025)

The :doc:`zarr_quickstart` covers this interface in full: points,
regions, streamed strips, fixed-size patches, matryoshka depths, and
caching.

Next Steps
----------

- Read the :doc:`architecture` section to understand how GeoTessera works internally
- Check the :doc:`tutorials` for more detailed examples
- Browse the :doc:`cli_reference` for all available command options
- Explore the :doc:`modules` for complete API documentation

Common Issues
-------------

**No tiles found in region**:
   Check the coverage map first using ``geotessera coverage`` with your region or bounding box. The region might not have available data.

**Slow downloads**:
   Files are cached after first download. Subsequent access will be much faster.

**Memory issues with large regions**:
   GeoTIFF export uses lazy streaming (one tile at a time). If you still
   hit memory limits, download smaller regions or use ``--bands`` to
   select fewer channels.

**Projection issues in GIS software**:
   GeoTIFF files use UTM projections. Most GIS software will handle this automatically.

**PCA visualization issues**:
   Ensure you have enough tiles for meaningful PCA. Single tiles may not produce good results.
