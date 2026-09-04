GeoTessera CLI Tests
=====================

These are tests for the `geotessera` command-line interface.

Setup
-----

Set environment variable to disable fancy terminal output (ANSI codes, boxes, colors):

  $ export TERM=dumb

Create a temporary directory for test outputs and cache:

  $ export TESTDIR="$CRAMTMP/test_outputs"
  $ mkdir -p "$TESTDIR"

Override XDG cache directory to use temporary location (for test isolation):

  $ export XDG_CACHE_HOME="$CRAMTMP/cache"
  $ mkdir -p "$XDG_CACHE_HOME"

Test: Version Command
---------------------

The version command should print the version number.

  $ geotessera version
  0.10.2

Test: Info Command (Library Info)
----------------------------------

Test the info command without arguments to see library information.
We just verify key information is present, ignoring formatting:

  $ geotessera info --dataset-version v1 2>&1 | grep -E 'Available years'
   Available years: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 

Test: Download Dry Run for UK Tile
-----------------------------------

Test downloading a single tile covering London, UK using --dry-run to avoid actual downloads.
Verify key information is present:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format tiff \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '(Format|Year|Compression|Dataset version|Found|Files to download|Total download|Tiles in region)' | sed 's/ *$//'
   Format:          TIFF
   Year:            2024
   Compression:     lzw
   Dataset version: v1
  Found 16 tiles for region in year 2024
   Files to download:   16
   Total download size: 6.4 GB
   Tiles in region:     16
   Year:                2024
   Format:              TIFF

Test: Download Single UK Tile (TIFF format)
--------------------------------------------

Download a single tile in TIFF format to a temporary directory:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format tiff \
  >   --output "$TESTDIR/uk_tiles_tiff" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Exported 16 GeoTIFF files

Verify TIFF files were created in the registry structure:

  $ [ -n "$(find "$TESTDIR/uk_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" 2>/dev/null)" ] && echo "TIFF files created"
  TIFF files created

  $ find "$TESTDIR/uk_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" | wc -l | tr -d ' '
  16

Test: Download Single UK Tile (NPY format)
-------------------------------------------

Download the same tile in NPY format (quantized arrays with scales):

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format npy \
  >   --output "$TESTDIR/uk_tiles_npy" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Downloaded 16 tiles (48 files, 1.7 GB)

Verify NPY directory structure was created:

  $ test -d "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" && echo "Embeddings directory created"
  Embeddings directory created

  $ test -d "$TESTDIR/uk_tiles_npy/global_0.1_degree_tiff_all" && echo "Landmasks directory created"
  Landmasks directory created

Verify NPY files exist in grid subdirectories:

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "grid_*.npy" ! -name "*_scales.npy" 2>/dev/null)" ] && echo "Embedding NPY files created"
  Embedding NPY files created

  $ find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "*.npy" | wc -l | tr -d ' '
  32

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "*_scales.npy" 2>/dev/null)" ] && echo "Scales NPY files created"
  Scales NPY files created

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_tiff_all" -name "*.tif*" 2>/dev/null)" ] && echo "Landmask TIFF files created"
  Landmask TIFF files created

Test: Info Command on Downloaded TIFF Tiles
--------------------------------------------

Test the info command on the downloaded TIFF tiles.
Both TIFF and NPY formats should be present (NPY files are retained for efficient reprocessing):

  $ geotessera info --tiles "$TESTDIR/uk_tiles_tiff" | sed 's/ *$//'
   Total tiles: 16
   Format:      GEOTIFF, NPY (USING NPY)
   Years:       2024
   CRS:         EPSG:32630, EPSG:32631
   Longitude: -0.200000 to 0.200000
   Latitude:  51.200000 to 51.600000
   Band Count Files
   128 bands     16

  $ geotessera info --tiles "$TESTDIR/uk_tiles_tiff" | sed 's/ *$//'
   Total tiles: 16
   Format:      GEOTIFF, NPY (USING NPY)
   Years:       2024
   CRS:         EPSG:32630, EPSG:32631
   Longitude: -0.200000 to 0.200000
   Latitude:  51.200000 to 51.600000
   Band Count Files
   128 bands     16

Test: Info Command on Downloaded NPY Tiles
-------------------------------------------

Test the info command on the downloaded NPY tiles:

  $ geotessera info --tiles "$TESTDIR/uk_tiles_npy"
   Total tiles: 16                     
   Format:      NPY                    
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.200000 to 0.200000  
   Latitude:  51.200000 to 51.600000 
   Band Count Files 
   128 bands     16 

  $ geotessera info --tiles "$TESTDIR/uk_tiles_npy"
   Total tiles: 16                     
   Format:      NPY                    
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.200000 to 0.200000  
   Latitude:  51.200000 to 51.600000 
   Band Count Files 
   128 bands     16 

Test: Resume Capability for NPY Downloads
------------------------------------------

Test that re-running the NPY download skips existing files:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format npy \
  >   --output "$TESTDIR/uk_tiles_npy" \
  >   --dataset-version v1 2>&1 | grep -E '(Skipped|existing files)'
     Skipped 48 existing files (resume capability)

Test: Tile Discovery Ignores Temporary Files
---------------------------------------------

Test that temporary files left from interrupted downloads are silently ignored.
Create temporary files manually in the NPY tiles directory to simulate interrupted downloads:

  $ touch "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024/.grid_0.05_51.25.npy_tmp_abc123"
  $ touch "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024/.grid_0.05_51.35_tmp_xyz789.npy"
  $ touch "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024/invalid_file.npy"

Verify that the info command still works correctly and doesn't show warnings about temp files.
The tile count should remain 16 (unchanged) and no warnings should appear in stderr:

  $ uv run -m geotessera.cli info --tiles "$TESTDIR/uk_tiles_npy" 2>&1 | grep -E '(Total tiles|WARNING|Failed to load|Cannot parse)'
   Total tiles: 16                     

