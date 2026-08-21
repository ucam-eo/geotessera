## Unreleased

### Breaking Changes

- All NPY downloads now come from the Source Cooperative, fronted by CloudFlare.
  Embeddings, landmasks and manifests are served from the public
  `https://data.source.coop/tessera/tessera` repository over HTTPS,
  replacing the retired `tessera-embeddings` AWS S3 bucket.

- `sphinx` and `cram` are no longer runtime dependencies.
  Sphinx moved to a `docs` extra (`pip install geotessera[docs]`)
  and cram to the `dev` dependency group (@avsm)

- The Zarr read API drops `crs=`. The store is UTM-native, so 
  `GeoTesseraZarr` takes lon/lat and routes to the `utm{NN}` group holding
  the point, and the `.tessera` accessor takes eastings and northings in
  that zone's own CRS. Project to lon/lat once up front, rather than per call.

- `botocore` and `awscrt` are no longer required dependencies as `urllib3`
  is a new direct dependency (@avsm)

### New Features

- The hosted Zarr store is available at
  `https://data.source.coop/tessera/tessera/zarr/v1`. `GeoTesseraZarr()`
  streams from it with no downloads and no configuration (@avsm)

- Seam-aware point reads for `sample_at` and `sample_points`.
  Tiles are 0.1 degrees and UTM zones 6 degrees, so round coordinates fall on
  tile edges and multiples of 6 fall on zone seams.  Our Zarr wrapper now
  tries the neighbouring UTM zone within 0.1 degrees of a seam, and
  `search_px` accepts the nearest valid pixel within 1 pixel.

-  A new `probe()` on the store and on the `.tessera` accessor returns
  `(embedding, status)`, the status one of `valid`, `water`, `nodata` or
  `outside`. Use it to tell open water from a location the store does not
  cover, which `sample_at` reports alike as NaN. It also rejects a point
  beyond the grid instead of snapping it to the nearest edge pixel.
  Reported by Srinivasan Keshav (@avsm)

- Per-version default variant allows omitting the dataset-variant.
  `--dataset-variant` now selects the version's default variant (`vultr`
  for v1, `cambridge` for v1.1, `2B-L~beta1` for v2) so
  `GeoTessera(dataset_version="v1.1")` works. 
  The known datasets are listed in the new "Known Datasets" table printed by
  `geotessera info` (@avsm)

- `geotessera-registry zarr-consolidate` is a new subcommand that
  re-consolidates a store's root metadata after in-place changes.
  Mostly only for repairs and not regular use.

- `geotessera-registry s3scan` scans Source Cooperative to list
  embeddings. This allows manifests and landmask registries to be
  regenerated directly from the Source Cooperative repository.  (@avsm)

- `open_zone(lon=...)` and `GeoTesseraZarr.open_zone(lon=...)` accept a
  whole-number longitude. `lon=-3` previously raised `TypeError`.  (@avsm)

### Performance

- Point sampling no longer rebuilds a pyproj transformer for every point.

### Documentation

- The `geotessera` package docstring now covers `GeoTesseraZarr` alongside the
  GeoTIFF export path; it previously described only the latter (@avsm)

### Internal

- The globe viewer HTML embedded in `cli.py` moved to
  `geotessera/templates/globe.html`, shipped as package data (@avsm)

## v0.9.0 (2026-06-09)

This release introduces support for multiple model versions (Tessera v1.0 and
1.1) along with dataset variants so that multiple model runs can be selected
and compared. All data downloads now go through the an AWS Open Data S3 bucket
using anonymous requests, with end-to-end checksum verification. This replaces
the previous direct HTTP mechanism and should hopefully be faster and more
reliable.

### New Features

- **Dataset variants** (`--dataset-variant`, `dataset_variant=`): The
  `GeoTessera` class and the `info`, `download`, and `coverage` CLI commands
  now accept a `--dataset-variant` option (default `vultr`) alongside
  `--dataset-version`. `--dataset-version` now also accepts the `v1.1` series
  in addition to `v1`.
  The resolved version/variant is recorded in a `tessera_metadata.json`
  provenance sidecar next to downloaded tiles (#250 @avsm)
- **`coverage --by-source`**: New flag that renders each `(version, variant)`
  source in a distinct colour on the coverage map and globe viewer. When set
  without an explicit `--dataset-version`/`--dataset-variant`, it downloads
  every known version's manifest and renders all sources together.  The
  `globe.html` viewer gains toggleable per-source layers and multi-dataset
  tile tooltips (#250 @avsm)
- **Coordinate lists accept iterables**: Functions taking coordinate lists now
  accept any iterable or generator (e.g. `zip(lons, lats)`), not just
  materialised lists (#259 @mdales)
- **`geotessera-registry s3scan`**: New subcommand that spiders the public S3
  bucket for embedding tiles across versions and variants and writes per-version
  `manifest.parquet` and `landmasks.parquet` files in an S3-mirroring layout
  (#250 @avsm)
- **`geotessera-registry zarr-stretch`**: New subcommand that computes a global
  cross-zone RGB stretch and stores it on a Zarr root for consistent colour
  across UTM zones. Supports `--mode bands` and `--mode pca` (learning three
  colour axes from the 128 embedding bands), with percentile, sampling, and
  worker controls (@avsm)
- **Chroma and gamma controls on `zarr-global-preview`**: The Zarr GeoTIFF
  preview renderer gains `--gamma` (per-channel power-law adjustment) and
  `--saturation` (luma/chroma decomposition with chroma scaling) for richer
  colour output, consuming the global stretch produced by `zarr-stretch` (@avsm)

### Breaking Changes

- **Downloads now use AWS S3 only**: All embedding, manifest, and landmask
  downloads switched from direct HTTP to anonymous (unsigned) S3 requests via
  The default base URL is now `https://s3.us-west-2.amazonaws.com/tessera-embeddings`.
  Custom non-S3 mirror URLs are no longer supported (#276 #278 @avsm)
- **New dependencies**: `botocore>=1.43.14` and `awscrt>=0.33.0` are now
  required (@avsm)
- **Per-version registry file renamed**: The downloaded registry file is now
  named `manifest.parquet` (per dataset version) rather than `registry.parquet`.
  The legacy `registry.parquet` name is still auto-detected for local
  `--registry-dir` overrides (#250 @avsm)

### Bug Fixes

- **End-to-end download integrity**: Downloads are now verified against the S3
  CRC64NVMe checksum. (#261 @mdales, #276 #278 @avsm)

## v0.8.0 (2026-04-05)

This release adds cloud-native Zarr access, GeoTIFF download improvements,
and several registry and CLI fixes.

The Zarr mode is an alternative to the npy, which will continue to be
supported. The embeddings are currently being transcoded to the new
format, and a future release will add registry support for the Zarr
as well for easy queries. For now, this release is mainly providing
the geotessera-registry support.

### New Features

- **Zarr v3 store** (`geotessera.store.GeoTesseraZarr`): Cloud-native access
  to Tessera embeddings via Zarr, with automatic UTM zone routing, point
  sampling, and region reading. Implements the `geoemb:` convention for
  geospatial embedding stores (@avsm)
- **GeoTIFF resume capability**: GeoTIFF downloads now skip existing files
  and resume interrupted downloads, matching the existing NPY resume behaviour
  (#222 @maawoo)
- **`scan --only` flag**: Selectively generate only the embeddings or landmasks
  parquet database during registry scans (@avsm)
- **Truncated NPY detection**: `geotessera-registry check` now detects
  truncated `.npy` files and reports them (@avsm)
- **`refresh` parameter**: `download_tile` and `export_embedding_geotiff`
  now expose a `refresh` parameter (default `False`) to force re-download
  of tiles even when local files exist (@avsm #238, reported in #237)

### Breaking Changes

- **Old Zarr format removed from `download` command**: `--format zarr` is no
  longer accepted; use `geotessera-registry zarr-init`/`zarr-fill` to
  build zarr stores and `GeoTesseraZarr` to read them
- **`visualize` command no longer accepts zarr input**: Only GeoTIFF and
  NPY format directories are supported

### Bug Fixes

- **Fixed memory leak in GeoTIFF export**: `export_embedding_geotiffs` now
  uses a lazy generator instead of materialising all tile data into memory,
  fixing out-of-memory errors for large regions (#137 #222 @maawoo)
- **Fixed bbox calculation for projected geometry files**: `--region-file`
  now reprojects to WGS84 before computing the bounding box, so files in
  UTM or other projected CRS produce correct results (#226 @maawoo)
- **Handle CRS-less geometry files**: Geometry files without CRS metadata
  (common with GeoJSON) now assume WGS84 instead of crashing (@avsm)
- **Fixed GeoTIFF export progress callback**: Resolved conflicting progress
  values between fetch and export phases that caused erratic progress bar
  behaviour (@avsm)
- **Atomic parquet writes with correct permissions**: Registry parquet files
  are now written atomically with 644 permissions (@avsm)
- **Skip ocean-only TIFFs**: Landmask parquet generation now skips
  ocean-only TIFFs that contain no land pixels (@avsm)

### Other

- Requires Python >= 3.12 (previously >= 3.11)
- New dependencies: `fsspec`, `aiohttp`, `geozarr-toolkit`, `contextily`

## v0.7.5 (2026-02-15)

This release reduces startup time for the library, improved coordinate clamping
and reduces the size of coverage data for the globe viewer.

- Auto-snap coordinates to valid tile centers in `fetch_embedding` and `download_tile`, so callers no longer need to compute exact 0.05-offset grid centers themselves (#166 #164 @avsm, reported by @tonyboston-au)
- Replaced tile/landmask dictionary caches with direct pandas MultiIndex lookups on `(year, lon_i, lat_i)` and `(lon_i, lat_i)`, simplifying the registry internals (#176 @avsm, reported by @sk818 in #175)
- Coverage JSON output split into per-year files (`coverage_YYYY.json`) to reduce payload size for the globe viewer (@avsm)
- Globe viewer now detects land vs ocean from the coverage texture pixels instead of storing `no_coverage`/`landmasks` lists in JSON (@avsm)

## v0.7.4 (2026-01-27)

This release adds convenience options for querying single tiles.

- New `--tile` option added to `download` and `coverage` commands for single-tile queries by any point within the tile (@avsm)
- Enhanced `--bbox` option to support both single-tile and bounding box formats (@avsm)

Licensing and docs clarifications as well:
- License clarification to fix mismatch between README and LICENSE and clarify MIT license (reported @adamjstewart in torchgeo/torchgeo#3243, fix by @avsm)
- Removed support request section due to resource limitations (@sk818)

## v0.7.3 (2025-12-17)

This release contains registry tooling improvements.

- Retired Pooch text manifest generation in favour of Parquet manifests (@avsm)
- Added tolerance for incomplete embedding directories during registry scans (@avsm)
- Improved warning grouping and diagnostics output (@avsm)
- Missing embeddings now written to a file for easier debugging (@avsm)

## v0.7.2 (2025-12-02)

This release adds Windows platform support, more robust tolerance to
interrupted scripts leaving temporary files around, and documentation fixes for
coordinate printing and tile discovery.

### Windows Support

Added Windows testing infrastructure in CI and applied code fixes (@avsm):
- New conda-based CI workflow for Windows runners
- PowerShell test suite (`tests/cli.ps1`) for Windows compatibility
- Cross-platform path handling improvements throughout the codebase

### Bug Fixes

- Fixed lon/lat printing order into a standardized coordinate order to lon/lat
  throughout CLI output. (Reported @GieziJo fix by @avsm).
 
- Fixed tile discovery false negatives arising from temporary files by removing
  pattern pre-filtering in `discover_tiles()` (Report from @sadiqj, fix @avsm)

- Fixed Windows file handling by closing temporary files before overwriting.
  (Fix from @dra27)

### Documentation

- **Fixed quickstart documentation**: Corrected `export_embedding_geotiffs` examples
  - Updated for year/lon/lat parameter order changes from v0.7.1
  - Fixed function signatures and usage examples (docs/quickstart.rst)

- **Updated README**:
  - Fixed coverage map image links
  - Corrected Windows path format examples

## v0.7.1 (2025-11-19)

This release adds Zarr format support for efficient cloud-native data
access and includes improvements to registry management tools.

### Zarr Format Support

- **New `--format zarr` option** for `download` command: Download embeddings as Zarr archives for efficient chunked access
  - Cloud-native format that's optimised for both local and cloud storage with built-in compression
  - xarray integration for analysis workflows
  - Metadata preservation includes CRS, scales, and georeferencing information
  - Usage: `geotessera download --bbox '...' --format zarr --output embeddings.zarr`

### Registry Improvements

- **New `scan` command** for `geotessera-registry`: Utility to scan directories of embeddings and build registry metadata
  - Efficiently indexes large collections of embedding files and validates file integrity and extracts metadata. Only for registry maintainers.

### Bug Fixes

- Fixed antimeridian handling in country point-in-polygon tests for accurate tile-country mapping, in the global coverage maps.

## v0.7.0 (2025-11-11)

This release moves to a Parquet-based registry for more efficient handling of
the growing embeddings metadata for TESSERA. It no longer maintains a central
cache, instead preferring the user to specify an embeddings directory within
which the remote registry tiles are mirrored (as npy files) and additional
mosaics and GeoTIFFs are generated. This helps make efficient use of disk space
due to the large size of the embeddings.

There are also new APIs for efficiently sampling embeddings for point data, and
to generate mosaics for classifiers over ROIs.

Note that there are significant interface changes throughout this release
compared to 0.6; please read the migration notes below. The library will
continue to evolve as we add more usecases, so please create issues on
<https://github.com/ucam-eo/geotessera> with your wishlists!

- **GeoParquet registry support**: Transitioned from text-based manifests to
  Parquet files (`registry.parquet`, `landmasks.parquet') for all tile metadata
- **Remove caching layer for tiles**: All embedding and landmask tiles are
  now directly downloaded to temporary files and only the Parquet registry is
  cached, since users were finding that embeddings storage was being duplicated
  in the old tile cache. This leads to a significant reduction in disk space.
- **Enhanced hash verification**: SHA256 verification now covers all downloaded files:
  - Embedding files (`.npy`) verified using `hash` column from registry
  - Scales files are also verified using the `scales_hash` column from the registry
  - Landmask files (`.tiff`) verified using `hash` column from landmasks registry
  - Can be disabled via `verify_hashes=False` parameter, `--skip-hash` CLI flag, or the `GEOTESSERA_SKIP_HASH=1` environment variable
  - Hash verification is **enabled by default** for data integrity
- **Lazy iterators** for reducing memory usage for large ROIs.

Note that the default registry hosting is now at <https://dl2.geotessera.org/v1/>
instead of the older server, as we had to upgrade our hosting to support the large
number of embeddings being generated for global coverage. We plan on bringing more
diverse hosting options online before the end of 2025.

### CLI Changes

- **New global options**:
  - `--registry-path` - Specify registry.parquet file
  - `--registry-url` - Specify registry URL
  - `--cache-dir` - Control registry cache location (replaces `TESSERA_DATA_DIR`)
  - Removed `--auto-update` and `--manifests-repo-url`

- **Enhanced `info` command**: Shows tiles per year and total landmask counts using fast pandas operations
- **Enhanced `coverage` command**: Generate a 3D globegl globe with coverage textures for HTML viewing.
- **New `--dry-run` option for `download` command**: Calculate total download size without downloading
  - Shows file count, total size, number of tiles, year, and format
  - Accounts for existing files (resume capability) - only counts files that would be downloaded
  - For NPY format: calculates exact sizes from registry for embeddings, scales, and landmasks
  - For TIFF format: provides size estimates (4x quantized size due to float32 conversion)
  - Useful for planning downloads and estimating bandwidth/storage requirements
  - Usage: `geotessera download --bbox '...' --dry-run`

- **New `--skip-hash` option for `download` command**: Skip SHA256 hash verification
  - Disables hash verification for embedding, scales, and landmask files
  - Can also be controlled via `GEOTESSERA_SKIP_HASH=1` environment variable
  - Hash verification is **enabled by default** for security
  - Usage: `geotessera download --bbox '...' --skip-hash`

### Registry CLI Changes

- **New `export-manifests` command**: Convert Parquet registry files to Pooch-format text manifests for backwards compatibility
  - Reads `registry.parquet` and `landmasks.parquet` files
  - Generates block-based text registry files in `registry/embeddings/` and `registry/landmasks/` subdirectories
  - Creates separate entries for `.npy` and `_scales.npy` files with their respective hashes
  - Useful for maintaining the tessera-manifests repository
  - Usage: `geotessera-registry export-manifests /path/to/v1 --output-dir ~/src/git/ucam-eo/tessera-manifests`

### Infrastructure Improvements

- **CRAM test suite**: Added comprehensive CLI tests using CRAM (Command-line Regression Acceptance Testing)
- **Dumb terminal support**: Added `TERM=dumb` support for non-interactive environments and CI pipelines
- **Logging system**: Migrated from print statements to Python's standard `logging` module for better integration

### Breaking Changes

- **NPY Download Format**: `geotessera download --format npy` now saves **quantized** embeddings with scales instead of dequantized embeddings
  - **New structure**: Files saved in `embeddings/{year}/grid_{lon}_{lat}.npy` (quantized) and `_scales.npy` (float32 scales)
  - **Landmasks included**: Saved in `landmasks/landmask_{lon}_{lat}.tif` structure
  - **No JSON metadata**: Removed JSON metadata files (use registry for metadata)
  - **Resume capability**: Can interrupt and restart downloads without re-downloading existing files
  - If you have existing NPY downloads, re-download with new version. Downloaded directories can now be reused with `GeoTessera(embeddings_dir=...)`

- **Registry API Changes**: Internal registry methods now return tuple for better resource management
  - `Registry.fetch()` now returns `(file_path, needs_cleanup)` tuple instead of just path
  - `Registry.fetch_landmask()` now returns `(file_path, needs_cleanup)` tuple instead of just path
  - These are internal changes - most users won't be affected

- **Registry Format Requirements**: Updated schema for Parquet registry files
  - `registry.parquet` now requires both `file_size` and `scales_hash` columns
  - `landmasks.parquet` requires `file_size` column
  - `file_size` used for accurate download progress reporting with total size
  - `scales_hash` stores SHA256 hash for scales files separately from embedding hash
  - Registry validation will fail if required columns are missing
  - Regenerate registries with latest `geotessera-registry scan` to include new columns

- **Environment variables**: `TESSERA_REGISTRY_DIR` and `TESSERA_DATA_DIR` deprecated in favor of CLI parameters
- **Registry format**: Completely new backend that migrates from text manifests to GeoParquet.
- **Cache behavior**: Only the registry is now cached, and not tile data to allow clients to manage their own disk usage.

### New API Features

- **`Tiles` class**: New abstraction for working with Tessera tiles
  - Provides unified interface for tile manipulation as either GeoTIFF or dequantized NumPy arrays
  - Simplifies conversion between formats
  - Accessible via `from geotessera.tiles import Tiles`

- **`GeoTessera(embeddings_dir=...)`**: New constructor parameter for local tile reuse
  - Points to directory containing pre-downloaded tiles
  - Expected structure: `embeddings/{year}/grid_{lon}_{lat}.npy` and `_scales.npy`, `landmasks/landmask_{lon}_{lat}.tif`
  - Automatically uses local files when available, downloads only if missing

- **`sample_embeddings_at_points(points, year, embeddings_dir=None, refresh=False)`**: Efficient point sampling
  - Extract embedding values at arbitrary lon/lat coordinates
  - Supports multiple input formats: list of tuples, GeoJSON FeatureCollection, GeoPandas GeoDataFrame
  - Automatically groups points by tile for efficient batch processing
  - Optional metadata return (tile info, pixel coords, CRS)
  - Can override instance `embeddings_dir` per call
  - Example: `embeddings = gt.sample_embeddings_at_points([(lon, lat), ...], year=2024)`

- **`fetch_embedding(..., refresh=False)`**: New parameter to force re-download
  - When `refresh=True`, re-downloads even if local tiles exist in `embeddings_dir`
  - Useful for updating tiles or verifying data integrity

- **New Registry size query methods**: Public API for querying file sizes from registry
  - `registry.get_tile_file_size(year, lon, lat)` - Get size of an embedding tile in bytes
  - `registry.get_landmask_file_size(lon, lat)` - Get size of a landmask tile in bytes
  - `registry.calculate_download_requirements(tiles, output_dir, format_type)` - Calculate total download size for a list of tiles
  - These methods replace direct registry DataFrame access and provide proper error handling
  - Used internally by CLI `--dry-run` option and available for programmatic use
  - Example: `size = gt.registry.get_tile_file_size(2024, 0.15, 52.05)`

- **`embeddings_count(bbox, year)`**: Get count of tiles in a bounding box
  - Returns total number of embedding tiles within a geographic region
  - Useful for planning downloads and estimating processing requirements
  - Example: `count = gt.embeddings_count((min_lon, min_lat, max_lon, max_lat), 2024)`

- **`export_coverage_map(output_file)`**: Export coverage data to JSON
  - Generates global coverage map showing which tiles have embeddings for which years
  - Returns dictionary with tile coverage information
  - Optionally saves to JSON file for use in visualizations

- **`generate_coverage_texture(coverage_data, output_file)`**: Generate coverage texture for globe visualization
  - Creates 3600x1800 pixel equirectangular projection texture
  - Each pixel represents a 0.1-degree tile, colored by coverage status
  - Used with `coverage` command for 3D globe visualizations, but also for your own visualisations

- **`dequantize_embedding(quantized_embedding, scales)`**: Public utility function for dequantization
  - Converts quantized embeddings to float32 by multiplying with scale factors
  - Useful when working directly with downloaded quantized NPY files, but use the Tiles class for normal usage.
  - Example: `embedding = dequantize_embedding(quantized, scales)`

### Migration Notes

From v0.6.0 to v0.7.0:
- Update initialization code to use new `cache_dir` parameter instead of environment variables
- Remove any custom `TESSERA_DATA_DIR` or `TESSERA_REGISTRY_DIR` environment variable usage
- Expect reduced disk usage as tiles are no longer cached but potentially more downloads.
- **If using NPY downloads**: Re-download tiles with new format to get quantized structure
- **To reuse downloaded tiles**: Use `GeoTessera(embeddings_dir="path/to/tiles")` when initializing
- **For point sampling**: Replace manual tile iteration with `sample_embeddings_at_points()`

## v0.6.0 (2025-09-15)

- registry: Add support for a Parquet registry as an alternative source
  to lookup tile information (#16).
- docs: Fix old documentation examples (#22, #18 report by @cjissmart)

## v0.5.2 (2025-09-02)

- cli: Add date/time/repo/hash information to the coverage maps
- cli(registry): Add commit command to help automation of manifests

## v0.5.1

- Added support for providing URLs as a `--region-file` parameter
- Added version information to CLI help text and command titles
- Added git manifest hash to version information for better traceability
- Reorganized CLI command order to be more logical and intuitive
- Removed deprecated `tilemap` command (replaced by improved `coverage` functionality)
- Improved the `geotessera-registry` hashing to be incremental

## v0.5.0

This release represents a significant architectural overhaul of GeoTessera as we
build more usecases. The library now focuses on delivering tiles with the CRS
system preserved 

## geotessera CLI commands

- `visualize` Command
  - **PCA visualization**: Create PCA visualizations from multiband GeoTIFF files
  - **Usage**: `geotessera visualize INPUT_PATH OUTPUT_FILE [OPTIONS]`
  - **New options**: CRS reprojection, PCA component selection, RGB balancing methods
  - **Support for**: Single tiles, directories of tiles, and complex mosaicking

- New `webmap` Command
  - **Complete web mapping pipeline**: `geotessera webmap RGB_MOSAIC [OPTIONS]`
  - **Features**: Generate web tiles, create HTML viewer, optional web server
  - **Customizable zoom levels**: Configurable min/max zoom for tile generation
  - **Boundary support**: Overlay GeoJSON/Shapefile boundaries on maps

- New `tilemap` Command
  - **Coverage visualization**: `geotessera tilemap INPUT_PATH [OPTIONS]`
  - **Generate HTML maps**: Show spatial coverage of GeoTIFF collections
  - **Customizable styling**: Title and display options

- Enhanced `download` Command
  - **Country support**: `--country` parameter for downloads by country boundary
  - **Multiple formats**: Enhanced support for both TIFF and NumPy formats
  - **Better metadata**: JSON metadata files with detailed tile information
  - **Improved progress reporting**: Rich progress bars with ETA and speed

- Enhanced `serve` Command
  - **Multi-format support**: Serve various visualization types
  - **Auto-open browser**: Automatic browser launching option
  - **Flexible file serving**: Support for HTML, image, and tile directory serving

- New `coverage` Command Options
  - **Enhanced styling**: Customizable tile colors, transparency, and sizing
  - **Output control**: Configurable DPI and figure dimensions
-   **Regional focus**: Filter coverage display by region files

### Breaking API Changes

- **Core library:**
  - `fetch_embedding()` returns `(embedding, crs, transform)` instead of just `embedding`
  - `fetch_embeddings()` returns list of `(lat, lon, embedding, crs, transform)` tuples instead of `(lat, lon, embedding)`
  - This provides direct access to the coordinate reference system from landmask tiles
  - Useful for applications that need projection information without exporting to GeoTIFF

- **Module restructuring**: Several modules have been reorganized for better functionality
  - **Removed**: `export.py`, `io.py`, `parallel.py`, `spatial.py`, `registry_utils.py` (these will return in future editions)
  - **Added**: `country.py`, `progress.py`, `visualization.py`, `web.py`
  - **Enhanced**: `core.py`, `cli.py`, `registry.py` with significant new functionality

- **New core methods**: Enhanced GeoTIFF processing capabilities
  - `merge_geotiffs_to_mosaic()` - Intelligent merging of multiple GeoTIFF files with CRS handling
  - `apply_pca_to_embeddings()` - Apply Principal Component Analysis to embedding data
  - `export_pca_geotiffs()` - Export PCA-transformed embeddings as georeferenced GeoTIFFs
  - Proper coordinate reference system preservation and transformation

- **New `visualization.py` module**:
  - `create_pca_mosaic()` - Generate PCA-based RGB visualizations from multiband GeoTIFFs
  - `visualize_global_coverage()` - Create global coverage maps with customizable styling
  - `create_rgb_mosaic()` - Advanced RGB composite creation with multiple balance methods
  - Support for histogram, percentile, and adaptive RGB balancing techniques

- **New `web.py` module**: Web mapping pipeline
  - `geotiff_to_web_tiles()` - Generate web map tiles from GeoTIFFs using GDAL
  - `create_simple_web_viewer()` - Generate complete HTML web map viewers
  - Support for Leaflet-based interactive maps with customizable zoom levels
  - Automatic boundary overlay support from GeoJSON/Shapefile regions

- **New `country.py` module**: Geographic boundary support using Natural Earth data
  - `CountryLookup` class for resolving country names, codes, and boundaries
  - Support for multiple country identifiers (names, ISO codes, etc.)
  - Automatic download and caching of Natural Earth 50m countries dataset
  - Integration with CLI `--country` parameter for easy regional downloads

- **New `progress.py` module**: Rich-based progress tracking system
  - Progress bars with detailed status information
  - Callback-based progress reporting for programmatic use
  - Integration throughout CLI commands for better user experience


### Performance and Efficiency Improvements

- Registry System Optimization
  - **Lazy loading**: Registry blocks loaded only when needed
  - **Memory efficiency**: Significant reduction in startup memory usage
  - **Caching improvements**: Better local caching and update mechanisms

- Processing Optimizations
  - **Coordinate system handling**: Preserved local projections until final export
  - **GDAL integration**: Enhanced GDAL tool integration for better performance with
    experimental support for the new `gdal raster tiles` (but this will really need
    a new release of gdal to be stable as the feature is still under development there)

### Dependencies

- **Added**: `scikit-learn>=1.7.1` for PCA functionality
- **Added**: `scikit-image>=0.25.2` for advanced image processing
- **Added**: `geodatasets>=2024.8.0` for geographic data access
- **Enhanced**: `rich` and `typer` for improved CLI experience
- **Updated**: Various dependencies to latest stable versions

### Migration Notes

From v0.4.0 to v0.5.0:
- **API changes**: Update code to handle new return values from `fetch_embedding()` and `fetch_embeddings()`
- **CLI workflow changes**: `visualize` command now operates on existing GeoTIFF files
- **Module imports**: Update imports for modules that have been restructured
- **Dependencies**: Run `uv sync` or equivalent to update to new dependency versions

Deprecated Features:
- **Old visualization workflow**: Previous inline visualization during download is replaced by separate `download` → `visualize` workflow
- **Legacy export functions**: Old export utilities replaced by enhanced core methods
- **Direct embedding visualization**: Now requires separate PCA step for optimal results

## v0.4.0

### Enhanced Full-Band GeoTIFF Support

- **Simplified GeoTIFF export**: Always uses float32 precision without normalization
  - **Removed normalization logic**: All outputs preserve dequantized embedding values exactly
  - **Consistent data type**: Always float32 to maintain precision regardless of band count
  - **Band selection**: Still supports selecting specific bands (e.g., `--bands 0 1 2`) while preserving raw values
  - **Backward compatible**: Existing scripts continue to work unchanged
- **Enhanced CLI**: `geotessera visualize` now defaults to full 128-band export when `--bands` is not specified
  - Default: `geotessera visualize --region area.json --output full.tif` (128 bands, float32)
  - Selected bands: `geotessera visualize --region area.json --bands 0 1 2 --output subset.tif` (3 bands, float32)

### CLI Improvements and Bug Fixes

- **Fixed `visualize` command**: Resolved "Unknown geometry type: 'featurecollection'" error
  - Fixed condition order bug in `find_tiles_for_geometry()` that incorrectly handled GeoDataFrames
  - Command now works reliably with GeoJSON, Shapefile, GeoPackage, and other region file formats
- **Improved performance**: Made `find_tiles_for_geometry()` efficient by loading only needed registry blocks
  - Previously loaded entire 400+ block registry, now loads only 1-4 blocks for typical regions
  - Faster startup and reduced memory usage for both `visualize` and `serve` commands
- **Enhanced tile generation**: Fixed `serve` command's gdal2tiles compatibility
  - Automatically converts float32 TIFF to 8-bit using `gdal_translate -scale` before tile generation
- **Better logging**: Improved registry loading messages
  - Clear distinction between newly loaded vs. already cached registry blocks
  - More informative progress reporting during region processing
- **Code rationalization**: Created shared logic between `visualize` and `serve` commands
  - Added `merge_embeddings_for_region_file()` method to core library for region file handling
  - Eliminated code duplication while maintaining full functionality

### Infrastructure Improvements

- **Natural Earth integration**: Set proper user agent when downloading world map data
- **Cleanup**: Removed accidentally committed world map files to reduce repository size

## v0.3.0

- Moved the map updating CI to https://github.com/ucam-eo/tessera-coverage.
  This results in a reset main branch with a cleaner git history.
- Modified `export_single_tile_to_tiff` so it can take not just 3 bands,
  allowing exporting of all 128 bands to a TIFF (#3 @epingchris)
- Fix degrees for georeferencing (#3 @nkarasiak and @avsm)
- Improve GDAL compatibility with different versions (#3 @nkarasiak)
- Fix map coverage generation with geopandas>1.0 (#4 @avsm, reported by @epingchris)
- Remove unnecessary registry directory existence check that prevented custom TESSERA_REGISTRY_DIR usage (#5 @avsm, reported by @epingchris)

## v0.2.0

### Breaking Changes

- **API**: `get_embedding()` method renamed to `fetch_embedding()` for clarity
- **Registry**: Switched from year-based to block-based (5x5 degree) registry system
- **Package**: Individual year registry files (`registry_2017.txt` through `registry_2024.txt`)
  removed as they are now tracked in https://github.com/ucam-eo/tessera-manifests

### New Features

- **Tessera utilities**:
  - `find_tiles_for_geometry()` - Find tiles intersecting with regions of interest
  - `extract_points()` - multi-point embedding extraction
  - Georeferencing utilities: `get_tile_bounds()`, `get_tile_crs()`, `get_tile_transform()`

- **New modules**:
  - `io.py` - Flexible I/O supporting JSON, CSV, GeoJSON, Shapefile, and Parquet formats
  - `spatial.py` - Spatial utilities for bounding boxes, grids, and raster stitching
  - `parallel.py` - Parallel processing for efficient tile operations
  - `export.py` - Export utilities for georeferenced GeoTIFFs

- **Registry improvements**:
  - Block-based registry system (5x5 degree blocks) for faster startup
  - Support for local registry via `TESSERA_REGISTRY_DIR` environment variable
  - Auto-cloning of tessera-manifests repository when no local registry specified
  - SHA256 checksum verification
  - New `geotessera-registry` CLI tool for registry management

### API Additions

- **GeoTessera constructor now autoclones manifests**:
  - `registry_dir` - Optional local registry directory path
  - `auto_update` - Auto-update tessera-manifests repository
  - `manifests_repo_url` - Custom manifests repository URL

- **New methods**:
  - `get_available_years()` - List available years in the dataset
  - Multiple georeferencing helper methods

### CLI Enhancements

The `geotessera` tool has also been improved.

- **New arguments**:
  - `--registry-dir` - Specify local registry directory
  - `--auto-update` - Auto-update tessera-manifests repository
  - `--manifests-repo-url` - Custom manifests repository URL

- **Command improvements**:
  - `info` command shows detailed registry and year information
  - `map` command displays year distribution
  - Better progress reporting and error messages

### Infrastructure

- Added `TESSERA_DATA_DIR` environment variable to override cache location
- Lazy loading of registry blocks for improved performance

### Dependencies

- Added `rich` for enhanced CLI output and progress bars
- Updated package metadata with license information and PyPI classifiers

### Bug Fixes

- Fixed tile alignment issues
- Improved landmask and TIFF file handling
- Better error handling and user feedback via exceptions
- Fixed coverage map generation
- Resolved coordinate formatting issues

## v0.1.0

Initial release to GitHub
