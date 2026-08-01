Architecture Guide
==================

This guide explains the internal architecture of GeoTessera and how the various components work together to provide efficient access to Tessera embeddings.

Overview
--------

GeoTessera is designed around a simple but powerful architecture that optimizes for:

- **Efficient data access**: Only download what you need
- **Projection preservation**: Maintain native UTM projections for accuracy
- **Scalability**: Handle large datasets with lazy loading
- **Flexibility**: Support both analysis and GIS workflows
- **Reliability**: Ensure data integrity with checksums

Core Architecture
-----------------

The library follows a layered architecture:

.. code-block::

    User Interface Layer
    ├── CLI Commands (geotessera download, visualize, etc.)
    └── Python API (GeoTessera class)
            ↓
    Core Processing Layer
    ├── GeoTessera class (main interface)
    ├── Registry (Parquet-based data discovery)
    └── Visualization (rendering and web maps)
            ↓
    Data Access Layer
    ├── HTTPS downloads (urllib3 pool, no cloud SDK)
    ├── Zarr v3 store (cloud-native streaming)
    ├── Rasterio (GeoTIFF I/O)
    └── GeoPandas (geospatial operations)
            ↓
    Storage Layer
    ├── Source Cooperative repository (https://data.source.coop/tessera/tessera)
    ├── Zarr store (https://data.source.coop/tessera/tessera/zarr/v1)
    └── Local cache (~/.cache/geotessera/{v1,v1.1-cam,...}/manifest.parquet)

Coordinate System and Grid
--------------------------

Understanding the Tessera Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings are organized on a **0.1-degree grid system**:

**Grid Properties**:

- **Grid spacing**: 0.1° latitude × 0.1° longitude
- **Tile naming**: Named by center coordinates (e.g., ``grid_0.15_52.05``)
- **Coverage**: Each tile spans from (center - 0.05°) to (center + 0.05°)
- **Resolution**: Approximately 11km × 11km at the equator

**Coordinate Calculations**::

    # For a tile at center coordinates (lon, lat)
    west = lon - 0.05
    east = lon + 0.05  
    south = lat - 0.05
    north = lat + 0.05

**Grid Alignment**:

Tile centers are aligned to 0.1-degree boundaries::

    # Valid tile centers (examples)
    valid_centers = [
        (0.05, 52.05),   # Northwest Europe
        (0.15, 52.05),   # Adjacent tile
        (-0.05, 51.95),  # Southwest tile
    ]
    
    # Invalid centers (not on grid)
    invalid_centers = [
        (0.07, 52.03),   # Off-grid
        (0.1, 52.1),     # Off by 0.05°
    ]

Resolution and Pixel Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of pixels per tile varies with latitude due to the Earth's curvature:

.. code-block:: python

    import math
    
    def pixels_per_tile(latitude, resolution_meters=10):
        """Calculate approximate pixels per tile at given latitude."""
        # Earth circumference at equator (meters)
        earth_circumference = 40075000
        
        # Degrees per meter at equator
        degrees_per_meter = 360 / earth_circumference
        
        # Adjust for latitude (longitude only)
        lon_degrees_per_meter = degrees_per_meter / math.cos(math.radians(latitude))
        lat_degrees_per_meter = degrees_per_meter
        
        # Tile size in meters
        tile_width_meters = 0.1 / lon_degrees_per_meter
        tile_height_meters = 0.1 / lat_degrees_per_meter
        
        # Pixels in tile
        pixels_width = int(tile_width_meters / resolution_meters)
        pixels_height = int(tile_height_meters / resolution_meters)
        
        return pixels_width, pixels_height
    
    # Examples
    eq_pixels = pixels_per_tile(0)      # ~(1111, 1111) at equator
    uk_pixels = pixels_per_tile(52)     # ~(1823, 1111) in UK
    arctic_pixels = pixels_per_tile(80) # ~(6389, 1111) near poles

Data Format and Storage
-----------------------

Quantization System
~~~~~~~~~~~~~~~~~~~

Tessera embeddings are stored using a quantization system for efficiency:

**Storage Format**:

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):
   
   - Data type: ``int8`` (values -128 to 127)
   - Shape: ``(height, width, 128)``
   - Storage efficient: ~1MB per tile vs ~64MB unquantized

2. **Scale factors** (``grid_X.XX_Y.YY_scales.npy``):
   
   - Data type: ``float32``
   - Shape: ``(height, width)`` or ``(height, width, 128)``
   - Contains dequantization multipliers

**Dequantization Process**::

    import numpy as np
    
    # Load quantized data and scales
    quantized = np.load("grid_0.15_52.05.npy")         # int8
    scales = np.load("grid_0.15_52.05_scales.npy")     # float32
    
    # Dequantize
    if scales.ndim == 2:
        # Broadcast 2D scales to 3D
        scales = scales[..., np.newaxis]
    
    embedding = quantized.astype(np.float32) * scales
    
    # Result: (height, width, 128) float32 array

This process is handled automatically by ``GeoTessera.fetch_embedding()``, which now returns the dequantized embedding along with CRS and transform information from the corresponding landmask tile.

Metadata and Projections
~~~~~~~~~~~~~~~~~~~~~~~~

**Landmask Files** (``grid_X.XX_Y.YY.tiff``):

- Provide native UTM projection information for each tile
- Define precise geospatial transforms (no reprojection needed)
- Preserve original coordinate system for maximum accuracy
- Used for georeferencing when exporting to GeoTIFF
- Contain binary land/water masks

**Projection Selection**:

Each tile uses an appropriate UTM zone based on its location::

    def get_utm_zone(longitude):
        """Get UTM zone number for a longitude."""
        return int((longitude + 180) / 6) + 1
    
    def get_utm_epsg(longitude, latitude):
        """Get EPSG code for UTM projection."""
        zone = get_utm_zone(longitude)
        
        if latitude >= 0:
            # Northern hemisphere
            return f"EPSG:{32600 + zone}"
        else:
            # Southern hemisphere  
            return f"EPSG:{32700 + zone}"
    
    # Example: London at (0.15, 52.05)
    epsg = get_utm_epsg(0.15, 52.05)  # "EPSG:32631" (UTM Zone 31N)

Registry System
---------------

Parquet-Based Registry
~~~~~~~~~~~~~~~~~~~~~~

The registry uses one **Parquet manifest per dataset version** for efficient
data discovery and querying. The manifest is filtered by ``(version,
variant)`` at load time, then queried by lat/lon/year:

**Manifest Structure**
(``data.source.coop/tessera/tessera/npy/{v1,v1.1-cam,v2-2B-L~beta1}/manifest.parquet``):

.. code-block::

    manifest.parquet (one per dataset version)
    ├── Columns:
    │   ├── version          # Normalised dataset version ('1.0', '1.1')
    │   ├── variant          # Dataset variant ('vultr', 'cambridge', ...)
    │   ├── lon, lat         # Tile center coordinates
    │   ├── year             # Data year
    │   ├── grid_size        # Embedding NPY byte size
    │   ├── scales_size      # Scales NPY byte size
    │   ├── grid_path        # Full s3:// URI of the embedding
    │   ├── scales_path      # Full s3:// URI of the scales
    │   ├── grid_mtime       # Object mtime on S3
    │   └── scales_mtime
    └── Rows: One per (version, variant, year, lon, lat) tile

Integrity is **not** carried in the manifest — every download is verified
against the response ``Content-Length``, and against an MD5 computed over
the streamed body whenever the server's ``ETag`` is a content MD5
(single-part uploads).

The per-version manifests are regenerated by maintainers with
``geotessera-registry s3scan``, which spiders the Source Cooperative
repository directly — see :doc:`maintenance`.

**Querying the Manifest**::

    import pandas as pd

    df = pd.read_parquet("manifest.parquet")  # the v1.1 file
    # Filter to (version, variant) you want — the manifest can carry both
    df = df[(df['version'] == '1.1') & (df['variant'] == 'cambridge')]

    # Query tiles in a region
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = df[
        (df['lon'] >= bbox[0]) & (df['lon'] <= bbox[2]) &
        (df['lat'] >= bbox[1]) & (df['lat'] <= bbox[3]) &
        (df['year'] == 2024)
    ]

**Manifest Loading Process**:

1. **Download per-version manifest** (only the one matching ``dataset_version``)
2. **Filter** to the requested ``(version, variant)``
3. **Materialise Point geometry** from lon/lat (R-tree spatial index)
4. **Set MultiIndex** on ``(year, lon_i, lat_i)`` for O(1) lookups
5. **Cache** the parquet at ``~/.cache/geotessera/{dataset_dir}/manifest.parquet``
   (e.g. ``v1/``, ``v1.1-cam/``);
   its mtime (set from the server's ``Last-Modified``) drives
   ``If-Modified-Since`` conditional refreshes

Manifest Sources
~~~~~~~~~~~~~~~~

The manifest can be loaded from multiple sources:

**1. Default Remote** (recommended)::

    # Downloads and caches the v1 manifest automatically.
    from geotessera import GeoTessera
    gt = GeoTessera()

    # Cached at: ~/.cache/geotessera/v1/manifest.parquet

    # Pick a different (version, variant) — see :ref:`dataset-versions`
    gt = GeoTessera(dataset_version="v1.1", dataset_variant="cambridge")

**2. Local File**::

    gt = GeoTessera(registry_path="/path/to/manifest.parquet")

**3. Local Directory**::

    # Looks for manifest.parquet in the directory (also accepts the legacy
    # registry.parquet name for backward compat).
    gt = GeoTessera(registry_dir="/path/to/manifest-dir")

**4. Custom URL**::

    gt = GeoTessera(registry_url="https://example.com/manifest.parquet")

**5. CLI Option**::

    geotessera download --cache-dir /custom/cache ...

Data Access Layer
-----------------

HTTPS Downloads
~~~~~~~~~~~~~~~

Manifests, embedding tiles, and landmasks all stream over plain HTTPS from
the public Source Cooperative repository. There is no cloud SDK; every
request shares one ``urllib3`` connection pool.

**Features**:

- **Per-output-dir mirroring**: Tiles land in the ``--output`` directory and
  persist there for re-use across runs
- **Connection pooling**: The thousands of per-tile GETs in a region download
  reuse connections rather than paying a handshake each
- **Retries**: Rate limiting, server errors, and dropped connections are
  retried with exponential backoff, honouring ``Retry-After``
- **Integrity checking**: A truncated response raises rather than ending the
  stream silently, and the body is checked against the server's ``ETag``
  wherever that is a content MD5
- **Conditional caching**: Manifests are refetched with an
  ``If-Modified-Since`` GET, so an unchanged one costs a 304 and no body
- **Progress callbacks**: Real-time download feedback with speed and size info
- **Resumable**: Existing files in the output dir are skipped on rerun

**Cache Structure**::

    ~/.cache/geotessera/
    ├── v1/
    │   ├── manifest.parquet           # Per-version tile manifest
    │   └── landmasks.parquet
    └── v1.1/
        ├── manifest.parquet
        └── landmasks.parquet

    # Embedding/landmark tile data lives in the user's --output dir,
    # not in this cache. The cache holds only per-version manifests.

**Download Process**::

    import numpy as np
    from geotessera import dequantize_embedding

    def fetch_embedding(lon, lat, year):
        # 1. Fetch the quantized embedding and scales tiles. ``fetch`` returns
        #    a path under embeddings_dir, downloading over HTTPS from the
        #    Source Cooperative repository if not present.
        embedding_file = registry.fetch(year=year, lon=lon, lat=lat, is_scales=False)
        scales_file = registry.fetch(year=year, lon=lon, lat=lat, is_scales=True)

        # 2. Load and dequantize
        quantized = np.load(embedding_file)
        scales = np.load(scales_file)
        embedding = dequantize_embedding(quantized, scales)

        # 3. Get CRS from the landmask tile
        crs, transform = get_utm_projection_from_landmask(lon, lat)

        return embedding, crs, transform

Persistent Tile Storage
~~~~~~~~~~~~~~~~~~~~~~~~

**Why persist tiles?**

- Tiles land in the user-supplied ``--output`` (``embeddings_dir``) and are
  re-used across runs rather than re-downloaded
- Existing files are skipped on rerun, making interrupted downloads resumable
- Only the small per-version manifests live in ``~/.cache/geotessera``;
  the bulk embedding data stays under the output directory the user controls

**Cache Configuration**::

    from geotessera import GeoTessera

    # Control where registry is cached
    gt = GeoTessera(cache_dir="/custom/cache")

    # Default cache locations:
    # - Linux/macOS: ~/.cache/geotessera/
    # - Windows: %LOCALAPPDATA%/geotessera/

GeoTIFF Export Process
~~~~~~~~~~~~~~~~~~~~~~

When exporting to GeoTIFF, additional processing occurs:

**Export Workflow**:

1. **Fetch embedding data** (quantized + scales)
2. **Fetch landmask tile** for projection information  
3. **Extract native UTM projection** and transform from landmask
4. **Apply dequantization** to embedding data
5. **Preserve original coordinate system** (no reprojection)
6. **Select bands** (if specified)
7. **Write GeoTIFF** with native UTM CRS and accurate transform
8. **Apply compression** (LZW, DEFLATE, etc.)

**Projection Inheritance**::

    import rasterio
    
    def export_geotiff(embedding, landmask_path, output_path, bands=None):
        # Read projection from landmask
        with rasterio.open(landmask_path) as landmask:
            crs = landmask.crs
            transform = landmask.transform
            
        # Select bands
        if bands:
            embedding = embedding[:, :, bands]
            
        # Write GeoTIFF
        with rasterio.open(output_path, 'w',
                          driver='GTiff',
                          height=embedding.shape[0],
                          width=embedding.shape[1], 
                          count=embedding.shape[2],
                          dtype=embedding.dtype,
                          crs=crs,
                          transform=transform,
                          compress='lzw') as dst:
            
            for i in range(embedding.shape[2]):
                dst.write(embedding[:, :, i], i + 1)

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

**Large Region Handling**:

When processing large regions, GeoTessera uses several strategies:

- **Tile-by-tile processing**: Process one tile at a time to limit memory usage
- **Band selection**: Only load required bands to reduce memory footprint  
- **Generator patterns**: Use generators for large tile collections
- **Progress callbacks**: Provide feedback for long operations

**Example Memory-Efficient Processing**::

    def process_large_region(bbox, year, bands=None):
        """Process a large region without loading all tiles into memory."""
        gt = GeoTessera()

        # Step 1: Get tile list (metadata only, no data loaded)
        tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)

        # Step 2: Process tiles one at a time using generator
        for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
            # Apply band selection early to reduce memory
            if bands:
                embedding = embedding[:, :, bands]

            # Process this tile
            result = process_single_tile(embedding)

            # Save or accumulate results
            save_tile_result(result, tile_lat, tile_lon)

            # Free memory
            del embedding

Network Optimization
~~~~~~~~~~~~~~~~~~~~

**Sequential Processing**:

The fetch_embeddings() generator processes tiles sequentially, which is optimal for most use cases::

    # Sequential processing (recommended for most cases)
    gt = GeoTessera()
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Returns generator - tiles are fetched one at a time
    for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
        process_tile(embedding)  # Memory efficient

**Point Sampling**:

For sampling at specific locations, use the optimized point sampling method::

    # Efficient point sampling with automatic tile download
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]
    embeddings = gt.sample_embeddings_at_points(points, year=2024)

    # With metadata about which tile each point came from
    embeddings, metadata = gt.sample_embeddings_at_points(
        points, year=2024, include_metadata=True
    )

**Cache Efficiency**:

- **Pre-warming**: Download commonly used tiles in advance
- **Batch processing**: Group requests by geographic region
- **Size limits**: Respect server rate limits

Zarr Store (Cloud-Native Access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GeoTesseraZarr`` class provides cloud-native access to embeddings
without downloading files. It implements the ``geoemb:`` convention for
geospatial embedding data stored in Zarr v3 format.

**Architecture**:

Data is organized by UTM zone, with each zone stored as a separate Zarr
group. The store automatically routes geographic queries to the correct zone::

    zarr store
    ├── Root attributes (geoemb:model, geoemb:build_version)
    ├── utm30/           # UTM Zone 30, on its own native UTM grid
    │   ├── embeddings   # (time, band, y, x) int8
    │   ├── scales       # (time, y, x) float32 — per-pixel dequantisation
    │   ├── time[:]      # Year coordinate array
    │   ├── x[:], y[:]   # UTM easting / northing coordinates
    │   └── band[:]
    ├── utm31/           # UTM Zone 31
    │   └── ...
    └── ...

**Coordinate systems**:

The store is UTM-native and nothing on the read path resamples pixels. Each
layer speaks exactly one coordinate system:

- ``GeoTesseraZarr`` takes **longitude and latitude** and routes each query
  to the ``utm{NN}`` group holding it
- The ``.tessera`` accessor takes **eastings and northings in that zone's own
  CRS**, since by then there is nothing left to route

Crossing between the two costs one point transform. Embeddings always come
back on their native UTM grid — ``read_region()`` returns the zone's CRS
alongside the mosaic, not the CRS of the bbox you asked with. Working from a
national grid such as EPSG:27700? Project your points to lon/lat once, up
front, rather than per call.

**Access patterns**:

- **Point sampling**: ``sample_points()`` / ``sample_at()`` for extracting
  embeddings at lon/lat coordinates across zones
- **Region reading**: ``read_region()`` takes a lon/lat bbox and returns the
  mosaic on the zone's native UTM grid, with its transform and CRS
- **Zone access**: ``open_zone()`` returns an xarray Dataset with a
  ``.tessera`` accessor for direct manipulation
- **Diagnostics**: ``probe()`` returns ``(embedding, status)``, the status
  one of ``valid``, ``water``, ``nodata`` or ``outside``. ``sample_at()``
  returns NaN for all but the first, so use ``probe()`` to tell open water
  from a location the store does not cover

Datasets are cached per zone for the lifetime of the ``GeoTesseraZarr``
instance.



This guide explains the internal architecture of GeoTessera and how the various components work together to provide efficient access to Tessera embeddings.

Overview
--------

GeoTessera is designed around a simple but powerful architecture that optimizes for:

- **Efficient data access**: Only download what you need
- **Projection preservation**: Maintain native UTM projections for accuracy
- **Scalability**: Handle large datasets with lazy loading
- **Flexibility**: Support both analysis and GIS workflows
- **Reliability**: Ensure data integrity with checksums

Core Architecture
-----------------

The library follows a layered architecture:

.. code-block::

    User Interface Layer
    ├── CLI Commands (geotessera download, visualize, etc.)
    └── Python API (GeoTessera class)
            ↓
    Core Processing Layer
    ├── GeoTessera class (main interface)
    ├── Registry (Parquet-based data discovery)
    └── Visualization (rendering and web maps)
            ↓
    Data Access Layer
    ├── HTTPS downloads (urllib3 pool, no cloud SDK)
    ├── Zarr v3 store (cloud-native streaming)
    ├── Rasterio (GeoTIFF I/O)
    └── GeoPandas (geospatial operations)
            ↓
    Storage Layer
    ├── Source Cooperative repository (https://data.source.coop/tessera/tessera)
    ├── Zarr store (https://data.source.coop/tessera/tessera/zarr/v1)
    └── Local cache (~/.cache/geotessera/{v1,v1.1-cam,...}/manifest.parquet)

Coordinate System and Grid
--------------------------

Understanding the Tessera Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings are organized on a **0.1-degree grid system**:

**Grid Properties**:

- **Grid spacing**: 0.1° latitude × 0.1° longitude
- **Tile naming**: Named by center coordinates (e.g., ``grid_0.15_52.05``)
- **Coverage**: Each tile spans from (center - 0.05°) to (center + 0.05°)
- **Resolution**: Approximately 11km × 11km at the equator

**Coordinate Calculations**::

    # For a tile at center coordinates (lon, lat)
    west = lon - 0.05
    east = lon + 0.05  
    south = lat - 0.05
    north = lat + 0.05

**Grid Alignment**:

Tile centers are aligned to 0.1-degree boundaries::

    # Valid tile centers (examples)
    valid_centers = [
        (0.05, 52.05),   # Northwest Europe
        (0.15, 52.05),   # Adjacent tile
        (-0.05, 51.95),  # Southwest tile
    ]
    
    # Invalid centers (not on grid)
    invalid_centers = [
        (0.07, 52.03),   # Off-grid
        (0.1, 52.1),     # Off by 0.05°
    ]

Resolution and Pixel Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of pixels per tile varies with latitude due to the Earth's curvature:

.. code-block:: python

    import math
    
    def pixels_per_tile(latitude, resolution_meters=10):
        """Calculate approximate pixels per tile at given latitude."""
        # Earth circumference at equator (meters)
        earth_circumference = 40075000
        
        # Degrees per meter at equator
        degrees_per_meter = 360 / earth_circumference
        
        # Adjust for latitude (longitude only)
        lon_degrees_per_meter = degrees_per_meter / math.cos(math.radians(latitude))
        lat_degrees_per_meter = degrees_per_meter
        
        # Tile size in meters
        tile_width_meters = 0.1 / lon_degrees_per_meter
        tile_height_meters = 0.1 / lat_degrees_per_meter
        
        # Pixels in tile
        pixels_width = int(tile_width_meters / resolution_meters)
        pixels_height = int(tile_height_meters / resolution_meters)
        
        return pixels_width, pixels_height
    
    # Examples
    eq_pixels = pixels_per_tile(0)      # ~(1111, 1111) at equator
    uk_pixels = pixels_per_tile(52)     # ~(1823, 1111) in UK
    arctic_pixels = pixels_per_tile(80) # ~(6389, 1111) near poles

Data Format and Storage
-----------------------

Quantization System
~~~~~~~~~~~~~~~~~~~

Tessera embeddings are stored using a quantization system for efficiency:

**Storage Format**:

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):
   
   - Data type: ``int8`` (values -128 to 127)
   - Shape: ``(height, width, 128)``
   - Storage efficient: ~1MB per tile vs ~64MB unquantized

2. **Scale factors** (``grid_X.XX_Y.YY_scales.npy``):
   
   - Data type: ``float32``
   - Shape: ``(height, width)`` or ``(height, width, 128)``
   - Contains dequantization multipliers

**Dequantization Process**::

    import numpy as np
    
    # Load quantized data and scales
    quantized = np.load("grid_0.15_52.05.npy")         # int8
    scales = np.load("grid_0.15_52.05_scales.npy")     # float32
    
    # Dequantize
    if scales.ndim == 2:
        # Broadcast 2D scales to 3D
        scales = scales[..., np.newaxis]
    
    embedding = quantized.astype(np.float32) * scales
    
    # Result: (height, width, 128) float32 array

This process is handled automatically by ``GeoTessera.fetch_embedding()``, which now returns the dequantized embedding along with CRS and transform information from the corresponding landmask tile.

Metadata and Projections
~~~~~~~~~~~~~~~~~~~~~~~~

**Landmask Files** (``grid_X.XX_Y.YY.tiff``):

- Provide native UTM projection information for each tile
- Define precise geospatial transforms (no reprojection needed)
- Preserve original coordinate system for maximum accuracy
- Used for georeferencing when exporting to GeoTIFF
- Contain binary land/water masks

**Projection Selection**:

Each tile uses an appropriate UTM zone based on its location::

    def get_utm_zone(longitude):
        """Get UTM zone number for a longitude."""
        return int((longitude + 180) / 6) + 1
    
    def get_utm_epsg(longitude, latitude):
        """Get EPSG code for UTM projection."""
        zone = get_utm_zone(longitude)
        
        if latitude >= 0:
            # Northern hemisphere
            return f"EPSG:{32600 + zone}"
        else:
            # Southern hemisphere  
            return f"EPSG:{32700 + zone}"
    
    # Example: London at (0.15, 52.05)
    epsg = get_utm_epsg(0.15, 52.05)  # "EPSG:32631" (UTM Zone 31N)

Registry System
---------------

Parquet-Based Registry
~~~~~~~~~~~~~~~~~~~~~~

The registry uses one **Parquet manifest per dataset version** for efficient
data discovery and querying. The manifest is filtered by ``(version,
variant)`` at load time, then queried by lat/lon/year:

**Manifest Structure**
(``data.source.coop/tessera/tessera/npy/{v1,v1.1-cam,v2-2B-L~beta1}/manifest.parquet``):

.. code-block::

    manifest.parquet (one per dataset version)
    ├── Columns:
    │   ├── version          # Normalised dataset version ('1.0', '1.1')
    │   ├── variant          # Dataset variant ('vultr', 'cambridge', ...)
    │   ├── lon, lat         # Tile center coordinates
    │   ├── year             # Data year
    │   ├── grid_size        # Embedding NPY byte size
    │   ├── scales_size      # Scales NPY byte size
    │   ├── grid_path        # Full s3:// URI of the embedding
    │   ├── scales_path      # Full s3:// URI of the scales
    │   ├── grid_mtime       # Object mtime on S3
    │   └── scales_mtime
    └── Rows: One per (version, variant, year, lon, lat) tile

Integrity is **not** carried in the manifest — every download is verified
against the response ``Content-Length``, and against an MD5 computed over
the streamed body whenever the server's ``ETag`` is a content MD5
(single-part uploads).

The per-version manifests are regenerated by maintainers with
``geotessera-registry s3scan``, which spiders the Source Cooperative
repository directly — see :doc:`maintenance`.

**Querying the Manifest**::

    import pandas as pd

    df = pd.read_parquet("manifest.parquet")  # the v1.1 file
    # Filter to (version, variant) you want — the manifest can carry both
    df = df[(df['version'] == '1.1') & (df['variant'] == 'cambridge')]

    # Query tiles in a region
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = df[
        (df['lon'] >= bbox[0]) & (df['lon'] <= bbox[2]) &
        (df['lat'] >= bbox[1]) & (df['lat'] <= bbox[3]) &
        (df['year'] == 2024)
    ]

**Manifest Loading Process**:

1. **Download per-version manifest** (only the one matching ``dataset_version``)
2. **Filter** to the requested ``(version, variant)``
3. **Materialise Point geometry** from lon/lat (R-tree spatial index)
4. **Set MultiIndex** on ``(year, lon_i, lat_i)`` for O(1) lookups
5. **Cache** the parquet at ``~/.cache/geotessera/{dataset_dir}/manifest.parquet``
   (e.g. ``v1/``, ``v1.1-cam/``);
   its mtime (set from the server's ``Last-Modified``) drives
   ``If-Modified-Since`` conditional refreshes

Manifest Sources
~~~~~~~~~~~~~~~~

The manifest can be loaded from multiple sources:

**1. Default Remote** (recommended)::

    # Downloads and caches the v1 manifest automatically.
    from geotessera import GeoTessera
    gt = GeoTessera()

    # Cached at: ~/.cache/geotessera/v1/manifest.parquet

    # Pick a different (version, variant) — see :ref:`dataset-versions`
    gt = GeoTessera(dataset_version="v1.1", dataset_variant="cambridge")

**2. Local File**::

    gt = GeoTessera(registry_path="/path/to/manifest.parquet")

**3. Local Directory**::

    # Looks for manifest.parquet in the directory (also accepts the legacy
    # registry.parquet name for backward compat).
    gt = GeoTessera(registry_dir="/path/to/manifest-dir")

**4. Custom URL**::

    gt = GeoTessera(registry_url="https://example.com/manifest.parquet")

**5. CLI Option**::

    geotessera download --cache-dir /custom/cache ...

Data Access Layer
-----------------

HTTPS Downloads
~~~~~~~~~~~~~~~

Manifests, embedding tiles, and landmasks all stream over plain HTTPS from
the public Source Cooperative repository. There is no cloud SDK; every
request shares one ``urllib3`` connection pool.

**Features**:

- **Per-output-dir mirroring**: Tiles land in the ``--output`` directory and
  persist there for re-use across runs
- **Connection pooling**: The thousands of per-tile GETs in a region download
  reuse connections rather than paying a handshake each
- **Retries**: Rate limiting, server errors, and dropped connections are
  retried with exponential backoff, honouring ``Retry-After``
- **Integrity checking**: A truncated response raises rather than ending the
  stream silently, and the body is checked against the server's ``ETag``
  wherever that is a content MD5
- **Conditional caching**: Manifests are refetched with an
  ``If-Modified-Since`` GET, so an unchanged one costs a 304 and no body
- **Progress callbacks**: Real-time download feedback with speed and size info
- **Resumable**: Existing files in the output dir are skipped on rerun

**Cache Structure**::

    ~/.cache/geotessera/
    ├── v1/
    │   ├── manifest.parquet           # Per-version tile manifest
    │   └── landmasks.parquet
    └── v1.1/
        ├── manifest.parquet
        └── landmasks.parquet

    # Embedding/landmark tile data lives in the user's --output dir,
    # not in this cache. The cache holds only per-version manifests.

**Download Process**::

    import numpy as np
    from geotessera import dequantize_embedding

    def fetch_embedding(lon, lat, year):
        # 1. Fetch the quantized embedding and scales tiles. ``fetch`` returns
        #    a path under embeddings_dir, downloading over HTTPS from the
        #    Source Cooperative repository if not present.
        embedding_file = registry.fetch(year=year, lon=lon, lat=lat, is_scales=False)
        scales_file = registry.fetch(year=year, lon=lon, lat=lat, is_scales=True)

        # 2. Load and dequantize
        quantized = np.load(embedding_file)
        scales = np.load(scales_file)
        embedding = dequantize_embedding(quantized, scales)

        # 3. Get CRS from the landmask tile
        crs, transform = get_utm_projection_from_landmask(lon, lat)

        return embedding, crs, transform

Persistent Tile Storage
~~~~~~~~~~~~~~~~~~~~~~~~

**Why persist tiles?**

- Tiles land in the user-supplied ``--output`` (``embeddings_dir``) and are
  re-used across runs rather than re-downloaded
- Existing files are skipped on rerun, making interrupted downloads resumable
- Only the small per-version manifests live in ``~/.cache/geotessera``;
  the bulk embedding data stays under the output directory the user controls

**Cache Configuration**::

    from geotessera import GeoTessera

    # Control where registry is cached
    gt = GeoTessera(cache_dir="/custom/cache")

    # Default cache locations:
    # - Linux/macOS: ~/.cache/geotessera/
    # - Windows: %LOCALAPPDATA%/geotessera/

GeoTIFF Export Process
~~~~~~~~~~~~~~~~~~~~~~

When exporting to GeoTIFF, additional processing occurs:

**Export Workflow**:

1. **Fetch embedding data** (quantized + scales)
2. **Fetch landmask tile** for projection information  
3. **Extract native UTM projection** and transform from landmask
4. **Apply dequantization** to embedding data
5. **Preserve original coordinate system** (no reprojection)
6. **Select bands** (if specified)
7. **Write GeoTIFF** with native UTM CRS and accurate transform
8. **Apply compression** (LZW, DEFLATE, etc.)

**Projection Inheritance**::

    import rasterio
    
    def export_geotiff(embedding, landmask_path, output_path, bands=None):
        # Read projection from landmask
        with rasterio.open(landmask_path) as landmask:
            crs = landmask.crs
            transform = landmask.transform
            
        # Select bands
        if bands:
            embedding = embedding[:, :, bands]
            
        # Write GeoTIFF
        with rasterio.open(output_path, 'w',
                          driver='GTiff',
                          height=embedding.shape[0],
                          width=embedding.shape[1], 
                          count=embedding.shape[2],
                          dtype=embedding.dtype,
                          crs=crs,
                          transform=transform,
                          compress='lzw') as dst:
            
            for i in range(embedding.shape[2]):
                dst.write(embedding[:, :, i], i + 1)

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

**Large Region Handling**:

When processing large regions, GeoTessera uses several strategies:

- **Tile-by-tile processing**: Process one tile at a time to limit memory usage
- **Band selection**: Only load required bands to reduce memory footprint  
- **Generator patterns**: Use generators for large tile collections
- **Progress callbacks**: Provide feedback for long operations

**Example Memory-Efficient Processing**::

    def process_large_region(bbox, year, bands=None):
        """Process a large region without loading all tiles into memory."""
        gt = GeoTessera()

        # Step 1: Get tile list (metadata only, no data loaded)
        tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)

        # Step 2: Process tiles one at a time using generator
        for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
            # Apply band selection early to reduce memory
            if bands:
                embedding = embedding[:, :, bands]

            # Process this tile
            result = process_single_tile(embedding)

            # Save or accumulate results
            save_tile_result(result, tile_lat, tile_lon)

            # Free memory
            del embedding

Network Optimization
~~~~~~~~~~~~~~~~~~~~

**Sequential Processing**:

The fetch_embeddings() generator processes tiles sequentially, which is optimal for most use cases::

    # Sequential processing (recommended for most cases)
    gt = GeoTessera()
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Returns generator - tiles are fetched one at a time
    for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
        process_tile(embedding)  # Memory efficient

**Point Sampling**:

For sampling at specific locations, use the optimized point sampling method::

    # Efficient point sampling with automatic tile download
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]
    embeddings = gt.sample_embeddings_at_points(points, year=2024)

    # With metadata about which tile each point came from
    embeddings, metadata = gt.sample_embeddings_at_points(
        points, year=2024, include_metadata=True
    )

**Cache Efficiency**:

- **Pre-warming**: Download commonly used tiles in advance
- **Batch processing**: Group requests by geographic region
- **Size limits**: Respect server rate limits

Zarr Store (Cloud-Native Access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GeoTesseraZarr`` class provides cloud-native access to embeddings
without downloading files. It implements the ``geoemb:`` convention for
geospatial embedding data stored in Zarr v3 format.

**Architecture**:

Data is organized by UTM zone, with each zone stored as a separate Zarr
group. The store automatically routes geographic queries to the correct zone::

    zarr store
    ├── Root attributes (geoemb:model, geoemb:build_version)
    ├── utm30/           # UTM Zone 30
    │   ├── time[:]      # Year coordinate array
    │   ├── embedding    # (time, y, x, band) float32
    │   └── ...
    ├── utm31/           # UTM Zone 31
    │   └── ...
    └── ...

**Access patterns**:

- **Point sampling**: ``sample_points()`` / ``sample_at()`` for extracting
  embeddings at specific coordinates across zones
- **Region reading**: ``read_region()`` for loading rectangular areas as
  mosaics with CRS and transform metadata
- **Zone access**: ``open_zone()`` returns an xarray Dataset with a
  ``.tessera`` accessor for direct manipulation
- **Diagnostics**: ``probe()`` returns ``(embedding, status)``, the status
  one of ``valid``, ``water``, ``nodata`` or ``outside``. ``sample_at()``
  returns NaN for all but the first, so use ``probe()`` to tell open water
  from a location the store does not cover

Datasets are cached per zone for the lifetime of the ``GeoTesseraZarr``
instance.

**Seam handling**:

Tiles are 0.1 degrees and UTM zones 6 degrees, so round coordinates fall on
tile edges and multiples of 6 fall on zone seams. Data is present at both,
but a tile edge may be one unwritten pixel wide, and a point on a seam is
often held by the zone next door. Point reads apply two fallbacks, each on
by default and disableable on its own:

- ``cross_zone`` also tries the neighbouring zone within 1 degree of a seam
- ``search_px`` accepts the nearest valid pixel within that radius

Neither reports land for sea. Water returns immediately as ``water``; only
an unwritten pixel starts a search.

Building a Zarr Store
~~~~~~~~~~~~~~~~~~~~~

Stores are built by the maintainer-facing ``geotessera-registry`` CLI in
three steps: ``zarr-init`` lays out the (metadata-only) hierarchy from the
landmask registry, ``zarr-fill`` writes tile data into it, and
``zarr-consolidate`` refreshes the root metadata that HTTP readers depend on.

**Locations, not paths**: the tile source and the output store are each
either a local directory or an fsspec URL. A store on one S3 node can be
filled from tiles on another with no local mirror::

    geotessera-registry zarr-fill \
        s3://source-bucket/tessera \
        s3://dest-bucket/tessera.zarr \
        --year 2024 --zones 30 \
        --source-endpoint-url https://data.source.coop --source-anon \
        --store-endpoint-url https://s3.example.org

Credentials come from the environment (``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``, ``AWS_PROFILE``, instance roles) rather than
flags, so they never appear in a process listing. ``--source-*`` and
``--store-*`` flags configure the two endpoints independently, so a named
AWS CLI profile is selected with ``--store-profile``.

Where the destination bucket belongs to another account, ``--store-acl``
stamps a canned ACL on every object written — the equivalent of the AWS
CLI's ``--acl``::

    --store-profile sc-writer --store-acl bucket-owner-full-control

It applies to the store's Zarr chunks and metadata as well as the sidecar
parquet and lock objects, and is filtered out of read requests.

.. note::

   **Source Cooperative reads and writes use different endpoints.**
   ``https://data.source.coop`` is the read-only gateway: anonymous reads
   work, but any write (and even a listing with write credentials) returns
   ``AccessDenied``. Writes go directly to the backing AWS bucket with *no*
   ``--endpoint-url`` at all::

       # read from the gateway, write to the backing bucket
       --source-endpoint-url https://data.source.coop --source-anon
       --output s3://us-west-2.opendata.source.coop/<account>/<repo>/zarr/v1
       --store-profile <writer> --store-region us-west-2
       --store-acl bucket-owner-full-control

   Passing ``--store-endpoint-url https://data.source.coop`` is the common
   mistake and produces a 403 that looks like a credentials problem.

``s3://`` locations need the optional ``s3`` extra, which pulls in ``s3fs``
and ``botocore``::

    pip install 'geotessera[s3]'

The core install stays free of both; ``https://`` sources (including the
public Source Cooperative front) work without the extra, since fsspec reads
those over the ``aiohttp`` already required.

**Streaming reads**: a ``.npy`` tile is a short header followed by a flat
C-ordered buffer, so the rows a shard needs are one contiguous byte range.
Remote tiles are read with a single ranged GET per shard overlap rather
than downloaded whole, which means no scratch disk and no cache eviction
policy to tune.

Adding a Year to an Existing Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time axis is fixed at ``zarr-init``, but it can be grown afterwards.
Because time is chunked one year per chunk, appending is a **metadata-only**
edit — existing chunks keep their keys and are never rewritten, however large
the store::

    # 1. grow every zone's time axis (no fills in flight)
    geotessera-registry zarr-extend s3://dest-bucket/tessera.zarr --years 2026

    # 2. fill the new year, one process per zone as usual
    geotessera-registry zarr-fill s3://source-bucket/tessera \
        s3://dest-bucket/tessera.zarr --year 2026 --zones 30

The new slice reads back exactly like a freshly initialised year —
embeddings at 0, scales at ``+inf`` ("land, no data yet") — so ``zarr-fill``
treats it no differently from the original ones, and the per-zone ingestion
registry keys on ``(zone, year)`` so earlier years are untouched.

Two constraints:

* **Append only.** Adding a year *earlier* than the current maximum would
  renumber every existing chunk's time index, i.e. rewrite the store. It is
  refused rather than done silently.
* **Single writer.** Unlike a fill, this rewrites array metadata for every
  zone, so it refuses to run while any fill lock is held, and it *does*
  re-consolidate afterwards (readers cannot see the new year until it has).

.. _zarr-parallel-sweep:

Parallel Per-Zone Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~

A UTM zone's pixels live entirely within its own ``utm{zone}`` group, and
shards never straddle zones. That makes ``--zones N`` the natural unit of
parallelism: one process per zone, all writing to the same store.

Everything a fill mutates is keyed by ``(zone, year)``, and none of it lives
inside the store — build bookkeeping goes to a sibling location so the
published hierarchy contains only Zarr:

.. code-block:: text

    tessera.zarr/
        zarr.json                              # shared — consolidation only
        utm30/, utm31/, ...                    # one zone per process

    tessera.zarr.build/                        # --state-url to relocate
        _registry/utm30_2024.parquet           # per-zone ingestion tracking
        _registry.parquet                      # merged view, written by consolidate
        _locks/utm30_2024.json                 # advisory fill lock

* **Ingestion tracking** is one object per zone/year, so no two jobs
  read-modify-write the same file. It records which tiles have already been
  written, which is what makes a fill resumable and lets a later run pick up
  tiles the manifest has gained since. It is build state, not published
  data — a reader of the store never needs it — so it lives in the state
  sibling. Stores built before this split kept a ``_registry.parquet``
  inside the hierarchy; that is still read, so they resume correctly.
* **An advisory lock** is taken for the duration of a zone/year fill. It
  catches the same zone being launched twice — the case that would silently
  corrupt data, because a shard write replaces the whole shard. Object
  stores offer no atomic create, so the lock is advisory; ``--force-lock``
  takes over one left behind by a dead run.
* **Consolidation is skipped** by default when ``--zones`` is given, since
  the root ``zarr.json`` is the one object all jobs share.

Resuming After a Crash
~~~~~~~~~~~~~~~~~~~~~~

The ingestion registry is written when a (zone, year) finishes, so a run
that dies partway — an OOM kill leaves no traceback — loses that year's
bookkeeping even though the shards it wrote are safely in the store.

The shard objects are the ground truth, and they survive anything::

    geotessera-registry zarr-fill <source> <store> --zones 30 \
        --skip-existing-shards

This lists the shard objects already present for each (zone, year), skips
them, and records their tiles so subsequent runs need no flag. It assumes
the tile inventory has not grown since those shards were written: a tile
added to the manifest afterwards falls inside an existing shard and would
be skipped rather than merged in. Where the credentials cannot list the
store, it falls back to probing only the shards the run would touch.

To see what is outstanding before committing to a sweep::

    geotessera-registry zarr-scan <source> <store> --output index.parquet

Every shard is classified ``written``, ``missing``, or ``empty`` — the last
meaning no manifest tiles fall in it, so it is ocean or outside coverage and
will never be filled. Keeping those separate means the percentages are over
land, not over each zone's bounding box. The command prints per-zone/year
and per-year summaries and writes the full per-shard index as parquet.

Note that worker memory, not cores, bounds the fill: each holds a full
``(128, 4096, 4096)`` int8 shard buffer plus its scales, about 2.1 GiB, so
``--workers 16`` needs 33 GiB resident. ``zarr-fill`` warns when the
requested count will not fit.

A sweep therefore looks like::

    # fan out — one process per zone, in parallel
    parallel -j8 geotessera-registry zarr-fill \
        s3://source-bucket/tessera s3://dest-bucket/tessera.zarr \
        --year 2024 --zones {} ::: $(seq 1 60)

    # single-writer finish: merge the per-zone registries, refresh the root
    geotessera-registry zarr-consolidate s3://dest-bucket/tessera.zarr

Each zone job exits non-zero if any of its shards failed, and tiles are only
recorded as written once every shard covering them succeeded — so re-running
the same command retries exactly the unfinished work.

The one thing that is *not* safe is splitting a single zone across
processes: whole-shard writes mean two jobs with different tile subsets
would erase each other's pixels. Within one job this is handled by
rebuilding each touched shard from all of its tiles, including ones an
earlier run already wrote.

Future Extensions
~~~~~~~~~~~~~~~~~

The architecture supports future enhancements:

- **Temporal queries**: Multi-year analysis
- **Cloud optimization**: Direct cloud storage access
- **ML integration**: TensorFlow/PyTorch data loaders
- **Real-time updates**: Live data ingestion
- **Distributed processing**: Dask/Ray integration
