GeoTessera Single Tile Tests
=============================

These are tests for the single tile download functionality using --tile and 2-coord --bbox.

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

Test: Single Tile with --tile Option (Dry Run)
----------------------------------------------

Test downloading a single tile using the --tile option with a point coordinate.
The point (0.17, 52.23) should resolve to tile grid_0.15_52.25:

  $ geotessera download \
  >   --tile "0.17,52.23" \
  >   --year 2024 \
  >   --format tiff \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '(Point|tile grid_|Found|Files to download|Tiles in region)'
  Point (0.17, 52.23) -> tile grid_0.15_52.25
  Found 1 tiles for region in year 2024
   Files to download:   1        
   Tiles in region:     1        

Test: Single Tile with 2-coord --bbox (Dry Run)
-----------------------------------------------

Test downloading a single tile using --bbox with only 2 coordinates.
This should behave identically to --tile:

  $ geotessera download \
  >   --bbox "0.17,52.23" \
  >   --year 2024 \
  >   --format tiff \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '(Point|tile grid_|Found|Files to download|Tiles in region)'
  Point (0.17, 52.23) -> tile grid_0.15_52.25
  Found 1 tiles for region in year 2024
   Files to download:   1        
   Tiles in region:     1        

Test: Mutual Exclusivity of Region Options
------------------------------------------

Test that specifying multiple region options produces an error:

  $ geotessera download \
  >   --tile "0.17,52.23" \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E 'Cannot specify multiple region'
  Error: Cannot specify multiple region options. Choose one of: --bbox, --tile, 

Test: Invalid --tile Format (Wrong Number of Coords)
-----------------------------------------------------

Test that --tile with wrong number of coordinates produces an error:

  $ geotessera download \
  >   --tile "0.17,52.23,0.20" \
  >   --year 2024 \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E "Error.*--tile must be"
  Error: --tile must be 'lon,lat'

Test: Invalid --bbox Format (Wrong Number of Coords)
-----------------------------------------------------

Test that --bbox with 3 coordinates produces an error:

  $ geotessera download \
  >   --bbox "0.17,52.23,0.20" \
  >   --year 2024 \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E "Error.*bbox must be"
  Error: bbox must be 'lon,lat' (single tile) or 'min_lon,min_lat,max_lon,max_lat'

Test: Download Single Tile (Actual Download)
---------------------------------------------

Download a single tile using --tile option:

  $ geotessera download \
  >   --tile "0.17,52.23" \
  >   --year 2024 \
  >   --format tiff \
  >   --output "$TESTDIR/single_tile_tiff" \
  >   --dataset-version v1 2>&1 | grep -E '(Point|SUCCESS)' | sed 's/ *$//'
  Point (0.17, 52.23) -> tile grid_0.15_52.25
  SUCCESS: Exported 1 GeoTIFF files

Verify that exactly one TIFF file was created:

  $ find "$TESTDIR/single_tile_tiff/global_0.1_degree_representation/2024" -name "*.tif*" | wc -l | tr -d ' '
  1

Verify the tile is named correctly (grid_0.15_52.25):

  $ find "$TESTDIR/single_tile_tiff/global_0.1_degree_representation/2024" -type d -name "grid_*" | xargs -I {} basename {}
  grid_0.15_52.25

Test: Coverage Command with --tile Option
-----------------------------------------

Test that coverage command also accepts --tile option and parses the tile correctly:

  $ geotessera coverage \
  >   --tile "0.17,52.23" \
  >   --output "$TESTDIR/single_tile_coverage.png" \
  >   --dataset-version v1 2>&1 | head -2
  Point (0.17, 52.23) -> tile grid_0.15_52.25
  Region bounding box: [0.150000\xc2\xb0E, 52.250000\xc2\xb0N] - [0.150000\xc2\xb0E, 52.250000\xc2\xb0N] (esc)

Test that an explicit --dataset-version all forces the multi-source view
without needing --by-source (as the flag help documents):

  $ geotessera coverage \
  >   --tile "0.17,52.23" \
  >   --output "$TESTDIR/all_datasets_coverage.png" \
  >   --dataset-version all 2>&1 | head -3
  Point (0.17, 52.23) -> tile grid_0.15_52.25
  Region bounding box: [0.150000\xc2\xb0E, 52.250000\xc2\xb0N] - [0.150000\xc2\xb0E, 52.250000\xc2\xb0N] (esc)
  Explicit 'all' requested: enabling --by-source rendering
