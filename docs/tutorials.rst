Tutorials
=========

These tutorials use the tile-download interface. For zarr-based
pipelines — classification, large-region inference, and false-colour
visualisation built on ``GeoTesseraZarr`` — see the
`geotessera-examples <https://github.com/ucam-eo/geotessera-examples>`_
repository and the :doc:`zarr_quickstart`.

Tutorial 1: Basic Data Analysis
-------------------------------

This tutorial covers the fundamentals of downloading and analyzing Tessera embeddings using Python.

Setup
~~~~~

First, install GeoTessera and import required libraries::

    pip install geotessera matplotlib jupyter

    # In Python/Jupyter
    import numpy as np
    import matplotlib.pyplot as plt
    from geotessera import GeoTessera
    import json

Initialize the Client
~~~~~~~~~~~~~~~~~~~~~

Create a GeoTessera client::

    # Initialize with default settings
    gt = GeoTessera()
    
    # Check available years
    years = gt.registry.get_available_years()
    print(f"Available years: {years}")

Explore Data Availability
~~~~~~~~~~~~~~~~~~~~~~~~~

Before downloading, check what data is available using the CLI::

    # Generate a coverage map
    # CLI command (preferred):
    # geotessera coverage --year 2024 --output global_coverage.png
    
    # For specific regions with boundary visualization:
    # geotessera coverage --country "United Kingdom" --year 2024  # Shows precise UK boundaries
    # geotessera coverage --region-file study_area.geojson --year 2024  # Shows custom region outline
    
    # Or using Python API:
    from geotessera.visualization import visualize_global_coverage
    
    visualize_global_coverage(
        tessera_client=gt,
        output_path="global_coverage.png",
        year=2024,
        width_pixels=2000,
        tile_color="blue",
        tile_alpha=0.4
    )
    
    print("Coverage map saved to global_coverage.png")
    print("Next step: Download data for your region")
    print("For better regional focus, use: geotessera coverage --country 'YourCountry'")

Download and Analyze a Single Tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with a single tile for Cambridge, UK::

    # Download embedding for Cambridge (note: lon, lat order)
    lon, lat = 0.15, 52.05
    year = 2024
    
    embedding, crs, transform = gt.fetch_embedding(lon=lon, lat=lat, year=year)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Data type: {embedding.dtype}")
    print(f"CRS: {crs}")
    print(f"Transform: {transform}")
    print(f"Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"Memory usage: {embedding.nbytes / 1024**2:.1f} MB")

Basic Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute basic statistics for the embedding::

    # Per-channel statistics
    mean_per_channel = np.mean(embedding, axis=(0, 1))
    std_per_channel = np.std(embedding, axis=(0, 1))
    
    print("First 10 channels:")
    for i in range(10):
        print(f"  Channel {i:2d}: mean={mean_per_channel[i]:6.3f}, "
              f"std={std_per_channel[i]:6.3f}")
    
    # Spatial statistics
    center_pixel = embedding[embedding.shape[0]//2, embedding.shape[1]//2, :]
    corner_pixel = embedding[0, 0, :]
    
    print(f"\nCenter pixel (first 5 channels): {center_pixel[:5]}")
    print(f"Corner pixel (first 5 channels): {corner_pixel[:5]}")

Visualize Individual Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create visualizations of different channels::

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Visualize channels 0, 10, 20, 30, 60, 100
    channels_to_plot = [0, 10, 20, 30, 60, 100]
    
    for i, channel in enumerate(channels_to_plot):
        ax = axes[i]
        im = ax.imshow(embedding[:, :, channel], cmap='viridis')
        ax.set_title(f'Channel {channel}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('cambridge_channels.png', dpi=150, bbox_inches='tight')
    plt.show()

Multi-Tile Regional Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download and analyze multiple tiles for a region::

    # Define bounding box for Cambridge area
    bbox = (0.0, 52.0, 0.3, 52.2)  # (min_lon, min_lat, max_lon, max_lat)

    # Step 1: Get list of available tiles in the region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    print(f"Found {len(tiles_to_fetch)} tiles in the region")

    # Step 2: Fetch all tiles with projection info (returns generator)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    # Analyze each tile
    tile_stats = []
    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        stats = {
            'lat': tile_lat,
            'lon': tile_lon,
            'mean_all_channels': np.mean(embedding),
            'std_all_channels': np.std(embedding),
            'channel_50_mean': np.mean(embedding[:, :, 50]),
            'channel_50_std': np.std(embedding[:, :, 50]),
            'crs': str(crs)
        }
        tile_stats.append(stats)

        print(f"Tile ({tile_lon:.2f}, {tile_lat:.2f}): "
              f"overall_mean={stats['mean_all_channels']:.3f}, "
              f"ch50_mean={stats['channel_50_mean']:.3f}")

Save Analysis Results
~~~~~~~~~~~~~~~~~~~~~

Save the analysis results for later use::

    # Save tile statistics
    with open('cambridge_analysis.json', 'w') as f:
        json.dump(tile_stats, f, indent=2)
    
    # Save raw embeddings for further analysis
    # Note: Need to re-fetch tiles if already consumed the generator
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        filename = f'cambridge_tile_{tile_lat:.2f}_{tile_lon:.2f}.npy'
        np.save(filename, embedding)
        print(f"Saved {filename}")

Tutorial 2: Point Sampling Workflow
------------------------------------

This tutorial shows how to efficiently sample embeddings at specific point locations.

Sample Embeddings at Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sample_embeddings_at_points()`` method provides an efficient way to extract embedding values at arbitrary locations::

    from geotessera import GeoTessera
    import numpy as np

    gt = GeoTessera()

    # Define points of interest (lon, lat tuples)
    points = [
        (0.15, 52.05),   # Cambridge
        (0.25, 52.15),   # Nearby location
        (-0.05, 51.55),  # London
        (-0.12, 51.50),  # Westminster
    ]

    # Sample embeddings at these points (auto-downloads tiles if needed)
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled embeddings shape: {embeddings.shape}")  # (4, 128)

    # Analyze the sampled embeddings
    for i, point in enumerate(points):
        print(f"Point {i} ({point[0]}, {point[1]}):")
        print(f"  Mean: {np.mean(embeddings[i]):.3f}")
        print(f"  Std: {np.std(embeddings[i]):.3f}")
        print(f"  First 5 channels: {embeddings[i][:5]}")

Get Metadata About Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include metadata to know which tile and pixel each sample came from::

    # Sample with metadata
    embeddings, metadata = gt.sample_embeddings_at_points(
        points, year=2024, include_metadata=True
    )

    for i, meta in enumerate(metadata):
        print(f"Point {i}:")
        print(f"  From tile: ({meta['tile_lon']}, {meta['tile_lat']})")
        print(f"  Pixel location: row={meta['pixel_row']}, col={meta['pixel_col']}")
        print(f"  CRS: {meta['crs']}")

Offline Mode for Point Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For guaranteed offline operation with no network requests::

    from geotessera import GeoTessera

    # Initialize with embeddings directory containing pre-downloaded tiles
    gt = GeoTessera(embeddings_dir="./my_tiles")

    # Option 1: Pre-download tiles first
    points = [(0.15, 52.05), (0.25, 52.15)]
    gt.download_tiles_for_points(points, year=2024)

    # Option 2: Sample in offline mode (will fail if tiles are missing)
    embeddings = gt.sample_embeddings_at_points(
        points, year=2024, auto_download=False
    )

Sample from GeoJSON or GeoDataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also pass GeoJSON or GeoPandas GeoDataFrame directly::

    import geopandas as gpd

    # From GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0.15, 52.05]}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0.25, 52.15]}},
        ]
    }
    embeddings = gt.sample_embeddings_at_points(geojson, year=2024)

    # From GeoPandas GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'name': ['Cambridge', 'Nearby']},
        geometry=gpd.points_from_xy([0.15, 0.25], [52.05, 52.15]),
        crs='EPSG:4326'
    )
    embeddings = gt.sample_embeddings_at_points(gdf, year=2024)

Tutorial 3: GIS Integration Workflow
------------------------------------

This tutorial shows how to work with GeoTIFF exports for GIS software integration.

Check Coverage and Export Embeddings as GeoTIFF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First check coverage, then export as georeferenced GeoTIFF files::

    # Step 1: Check coverage (CLI recommended)
    # geotessera coverage --bbox "-0.2,51.4,0.1,51.6" --year 2024
    
    from geotessera import GeoTessera
    
    gt = GeoTessera()
    
    # Define region (London area)
    bbox = (-0.2, 51.4, 0.1, 51.6)
    year = 2024
    
    # Step 2: Download via CLI (preferred) or Python API
    # CLI: geotessera download --bbox "-0.2,51.4,0.1,51.6" --year 2024 --output ./london_full
    # CLI: geotessera download --bbox "-0.2,51.4,0.1,51.6" --year 2024 --bands "30,60,90" --output ./london_rgb
    
    # Or using Python API:
    # Export all bands
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
    all_files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./london_full",
        compress="lzw"
    )
    
    print(f"Exported {len(all_files)} GeoTIFF files")
    print("Next step: geotessera visualize ./london_full pca_mosaic.tif")
    
    # Export RGB subset for visualization
    rgb_files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./london_rgb",
        bands=[30, 60, 90],  # Custom RGB bands
        compress="lzw"
    )
    
    print(f"Exported {len(rgb_files)} RGB GeoTIFF files")
    print("Next step: geotessera visualize ./london_rgb pca_rgb_mosaic.tif")

Inspect GeoTIFF Metadata
~~~~~~~~~~~~~~~~~~~~~~~~

Check the georeferencing information::

    import rasterio
    
    # Inspect the first file
    sample_file = all_files[0]
    
    with rasterio.open(sample_file) as src:
        print(f"File: {sample_file}")
        print(f"Shape: {src.shape}")
        print(f"Bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Bounds: {src.bounds}")
        
        # Read a sample of the data
        sample_data = src.read(1)  # Read first band
        print(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")

Create PCA Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

Create a PCA visualization from the exported tiles::

    # Using CLI:
    # geotessera visualize ./london_rgb pca_rgb_mosaic.tif
    # geotessera visualize ./london_full pca_full_mosaic.tif --n-components 5
    
    # This creates a PCA-based RGB mosaic that:
    # 1. Combines all embedding data across tiles
    # 2. Applies PCA transformation for dimensionality reduction
    # 3. Maps first 3 principal components to RGB channels
    # 4. Eliminates tiling artifacts through consistent PCA across region
    
    print("PCA mosaic created")
    print("Next step: geotessera webmap pca_rgb_mosaic.tif --serve")

Generate Web Tiles and Viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create interactive web tiles from the PCA mosaic::

    # Using CLI:
    # geotessera webmap pca_rgb_mosaic.tif --serve
    
    # This command automatically:
    # 1. Reprojects mosaic for web viewing if needed
    # 2. Generates web tiles at multiple zoom levels
    # 3. Creates HTML viewer with Leaflet map
    # 4. Starts web server and opens in browser
    
    # For custom options:
    # geotessera webmap pca_rgb_mosaic.tif --min-zoom 6 --max-zoom 18 --output webmap/ --serve
    
    print("Web tiles and viewer created")
    print("Interactive map should open in your browser")

QGIS Integration
~~~~~~~~~~~~~~~~

Tips for using the GeoTIFF files in QGIS:

1. **Loading files**: Drag and drop GeoTIFF files directly into QGIS
2. **Projection**: Files use UTM projection - QGIS will handle reprojection automatically
3. **Styling**: Use single-band pseudocolor for individual channels
4. **RGB composites**: Use the RGB mosaic files for natural color visualization
5. **Analysis**: Use QGIS raster calculator for band math operations

Example QGIS workflow::

    # In QGIS Python console
    from qgis.core import QgsRasterLayer
    
    # Load a GeoTIFF
    layer = QgsRasterLayer('/path/to/london_full/grid_51.45_-0.05.tif', 'Tessera Embedding')
    QgsProject.instance().addMapLayer(layer)
    
    # Set single-band pseudocolor for channel 50
    from qgis.core import QgsColorRampShader, QgsSingleBandPseudoColorRenderer
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 50)  # Channel 50
    shader = QgsColorRampShader()
    # Configure color ramp...
    layer.setRenderer(renderer)

Tutorial 4: Large-Scale Analysis
--------------------------------

This tutorial covers working with large regions and multiple years of data.

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with large regions, use CLI for efficient processing::

    # Use CLI for large regions
    # Step 1: Check coverage first
    # geotessera coverage --bbox "-3.0,50.0,2.0,53.0" --year 2024

    # Step 2: Download in smaller chunks or use selective bands
    # geotessera download --bbox "-3.0,50.0,2.0,53.0" --year 2024 --bands "0,10,20,30,40" --output ./southern_england

    # Step 3: Create PCA visualization (handles large datasets efficiently)
    # geotessera visualize ./southern_england pca_southern_england.tif --n-components 5

    # For Python analysis of large regions:
    from geotessera import GeoTessera
    import numpy as np

    gt = GeoTessera()

    # Large region (entire southern England)
    bbox = (-3.0, 50.0, 2.0, 53.0)
    year = 2024

    def process_large_region_efficiently(bbox, year, analysis_func):
        """Process a large region without loading all tiles into memory."""

        # Step 1: Get list of available tiles (metadata only, no data loaded)
        tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
        total_tiles = len(tiles_to_fetch)

        print(f"Processing {total_tiles} tiles...")
        print("Consider using CLI: geotessera download + geotessera visualize for large regions")

        # Step 2: Fetch tiles as generator (one at a time, memory efficient)
        tiles = gt.fetch_embeddings(tiles_to_fetch)

        results = []
        for i, (year, tile_lon, tile_lat, embedding, crs, transform) in enumerate(tiles):
            # Process one tile at a time
            result = analysis_func(embedding, tile_lat, tile_lon)
            results.append(result)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_tiles} tiles")

            # Free memory
            del embedding

        return results
    
    def vegetation_analysis(embedding, lat, lon):
        """Example analysis function for vegetation detection."""
        # Hypothetical vegetation channels (example)
        veg_channels = [20, 25, 30, 35, 40]
        
        # Compute vegetation index
        veg_data = embedding[:, :, veg_channels]
        veg_index = np.mean(veg_data, axis=2)
        
        return {
            'lat': lat,
            'lon': lon,
            'mean_vegetation': float(np.mean(veg_index)),
            'max_vegetation': float(np.max(veg_index)),
            'vegetation_pixels': int(np.sum(veg_index > 0.5))
        }
    
    # Run the analysis
    results = process_large_region_efficiently(bbox, year, vegetation_analysis)
    
    # Save results
    with open('vegetation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

Batch Export for Multiple Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export multiple regions efficiently using CLI commands::

    # Use CLI for batch processing
    # Create a shell script for batch downloads:
    
    #!/bin/bash
    
    # Check coverage for all regions first (with boundary visualization)
    echo "Checking coverage for all regions..."
    geotessera coverage --country "United Kingdom" --year 2024 --output uk_coverage.png  # Shows full UK boundaries
    geotessera coverage --bbox "-0.3,51.3,0.2,51.7" --year 2024 --output london_coverage.png
    geotessera coverage --bbox "-0.2,52.0,0.3,52.3" --year 2024 --output cambridge_coverage.png
    geotessera coverage --bbox "-1.4,51.6,-1.1,51.9" --year 2024 --output oxford_coverage.png
    
    # Download regions
    echo "Downloading regions..."
    geotessera download --bbox "-0.3,51.3,0.2,51.7" --year 2024 --bands "10,20,30,40,50" --output ./batch_exports/london
    geotessera download --bbox "-0.2,52.0,0.3,52.3" --year 2024 --output ./batch_exports/cambridge
    geotessera download --bbox "-1.4,51.6,-1.1,51.9" --year 2024 --bands "0,1,2" --output ./batch_exports/oxford
    
    # Create PCA visualizations
    echo "Creating PCA visualizations..."
    geotessera visualize ./batch_exports/london pca_london.tif
    geotessera visualize ./batch_exports/cambridge pca_cambridge.tif
    geotessera visualize ./batch_exports/oxford pca_oxford.tif
    
    # Create web viewers
    echo "Creating web viewers..."
    geotessera webmap pca_london.tif --output london_web/ --serve &
    geotessera webmap pca_cambridge.tif --output cambridge_web/
    geotessera webmap pca_oxford.tif --output oxford_web/
    
    # For Python-based batch processing:
    def batch_export_regions(regions_config, base_output_dir):
        """Export multiple regions as GeoTIFF files."""
        import os
        from pathlib import Path
        
        gt = GeoTessera()
        
        for region_name, config in regions_config.items():
            print(f"Processing region: {region_name}")
            print(f"Recommend using CLI: geotessera download --bbox '{','.join(map(str, config['bbox']))}' --year {config['year']} --output ./batch_exports/{region_name}")
            
            output_dir = Path(base_output_dir) / region_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=config['bbox'], year=config['year'])
                files = gt.export_embedding_geotiffs(
                    tiles_to_fetch,
                    output_dir=str(output_dir),
                    bands=config.get('bands', None),
                    compress="lzw"
                )
                
                print(f"  Exported {len(files)} files to {output_dir}")
                print(f"  Next step: geotessera visualize {output_dir} pca_{region_name}.tif")
                
                # Create metadata file
                metadata = {
                    'region': region_name,
                    'bbox': config['bbox'],
                    'year': config['year'],
                    'files': files,
                    'band_count': len(config.get('bands', list(range(128)))),
                    'next_steps': [
                        f"geotessera visualize {output_dir} pca_{region_name}.tif",
                        f"geotessera webmap pca_{region_name}.tif --serve"
                    ]
                }
                
                with open(output_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                print(f"  Error processing {region_name}: {e}")
    
    # Define regions to process
    regions = {
        'london': {
            'bbox': (-0.3, 51.3, 0.2, 51.7),
            'year': 2024,
            'bands': [10, 20, 30, 40, 50]  # Subset of bands
        },
        'cambridge': {
            'bbox': (-0.2, 52.0, 0.3, 52.3),
            'year': 2024,
            'bands': None  # All bands
        },
        'oxford': {
            'bbox': (-1.4, 51.6, -1.1, 51.9),
            'year': 2024,
            'bands': [0, 1, 2]  # RGB only
        }
    }
    
    # Run batch export
    batch_export_regions(regions, "./batch_exports")

Multi-Year Comparison
~~~~~~~~~~~~~~~~~~~~~

Compare embeddings across different years::

    def compare_years(lat, lon, years):
        """Compare a single location across multiple years."""
        gt = GeoTessera()
        
        yearly_data = {}
        for year in years:
            try:
                embedding, crs, transform = gt.fetch_embedding(lon=lon, lat=lat, year=year)
                
                # Compute summary statistics
                yearly_data[year] = {
                    'mean_per_channel': np.mean(embedding, axis=(0, 1)).tolist(),
                    'std_per_channel': np.std(embedding, axis=(0, 1)).tolist(),
                    'overall_mean': float(np.mean(embedding)),
                    'overall_std': float(np.std(embedding))
                }
                
                print(f"Year {year}: mean={yearly_data[year]['overall_mean']:.3f}")
                
            except Exception as e:
                print(f"Year {year}: Data not available ({e})")
                yearly_data[year] = None
        
        return yearly_data
    
    # Compare Cambridge across years
    cambridge_comparison = compare_years(
        lat=52.05, lon=0.15, 
        years=[2020, 2021, 2022, 2023, 2024]
    )
    
    # Save comparison
    with open('cambridge_temporal_comparison.json', 'w') as f:
        json.dump(cambridge_comparison, f, indent=2)
    
    # Plot temporal trends
    valid_years = [year for year, data in cambridge_comparison.items() if data is not None]
    overall_means = [cambridge_comparison[year]['overall_mean'] for year in valid_years]
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_years, overall_means, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Year')
    plt.ylabel('Mean Embedding Value')
    plt.title('Temporal Trend - Cambridge (52.05°N, 0.15°E)')
    plt.grid(True, alpha=0.3)
    plt.savefig('cambridge_temporal_trend.png', dpi=150, bbox_inches='tight')
    plt.show()

Tutorial 5: Coverage Analysis with Boundary Visualization
---------------------------------------------------------

Understanding data coverage with precise geographic boundaries.

Visualizing Country Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coverage command shows precise country boundaries when using ``--country``::

    # Countries with simple boundaries
    geotessera coverage --country "Germany" --year 2024
    geotessera coverage --country "France" --year 2024
    
    # Countries with complex coastlines and islands
    geotessera coverage --country "Greece" --year 2024     # Shows all Greek islands
    geotessera coverage --country "United Kingdom" --year 2024  # Shows England, Scotland, Wales, N. Ireland
    geotessera coverage --country "Indonesia" --year 2024  # Shows thousands of islands
    
    # Using country codes
    geotessera coverage --country "UK" --year 2024
    geotessera coverage --country "US" --year 2024

Comparing Boundary vs Bounding Box
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Bounding box approach (rectangular, may include unwanted areas)
    geotessera coverage --bbox "20,35,30,42" --output greece_bbox.png
    
    # Precise boundary approach (follows actual country borders)
    geotessera coverage --country "Greece" --output greece_precise.png
    
The country approach shows only tiles that actually intersect with Greek territory,
excluding tiles over water or neighboring countries that fall within the bounding box.

Tutorial 6: Custom Analysis Workflows
-------------------------------------

Advanced analysis techniques and custom workflows.

Principal Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce dimensionality of the 128-channel embeddings::

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    def perform_pca_analysis(embeddings_list, n_components=10):
        """Perform PCA on a collection of embedding tiles."""
        
        # Reshape all embeddings to 2D (pixels x channels)
        all_pixels = []
        tile_info = []
        
        for year, tile_lon, tile_lat, embedding, crs, transform in embeddings_list:
            # Reshape from (H, W, 128) to (H*W, 128)
            pixels = embedding.reshape(-1, embedding.shape[-1])
            all_pixels.append(pixels)
            
            # Track which pixels belong to which tile
            n_pixels = pixels.shape[0]
            tile_info.extend([(tile_lat, tile_lon)] * n_pixels)
        
        # Combine all pixels
        X = np.vstack(all_pixels)
        print(f"Total pixels for PCA: {X.shape[0]:,}")
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Print explained variance
        print(f"\nExplained variance by component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca, pca, scaler, tile_info
    
    # Example usage
    gt = GeoTessera()
    bbox = (-0.1, 51.9, 0.1, 52.1)  # Small region around Cambridge
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    tiles = list(gt.fetch_embeddings(tiles_to_fetch))
    
    X_pca, pca, scaler, tile_info = perform_pca_analysis(tiles, n_components=5)
    
    # Visualize first two principal components
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA: First Two Components')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

If you would like to now try more advanced classification, go to the
`Tessera interactive notebook <https://github.com/ucam-eo/tessera-interactive-notebook>`_
for a Jupyter-based label classifier application.

For cloud-native zarr access to the TESSERA store, see
:class:`~geotessera.store.GeoTesseraZarr`.
