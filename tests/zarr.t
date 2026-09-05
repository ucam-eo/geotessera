Zarr Remote Store Tests
=======================

These cover the pieces that let `geotessera-registry zarr-fill` write to a
remote object store while streaming tiles from another one, and the state
that makes per-zone fills safe to run in parallel.

Setup
-----

  $ export TERM=dumb

Test: zarr-fill accepts remote locations
-----------------------------------------

Both the tile source and the store may be URLs:

  $ geotessera-registry zarr-fill --help | grep -c 'Path or URL of an existing tessera store'
  1

Per-zone sweeps are documented on the command itself:

  $ geotessera-registry zarr-fill --help | grep -c 'concurrently'
  1

Object-store credentials come from flags or the environment, never from
positional arguments:

  $ geotessera-registry zarr-fill --help | grep -oE '\-\-(source|store)-endpoint-url' | sort -u
  --source-endpoint-url
  --store-endpoint-url

Consolidation is opt-out for a zone-restricted fill:

  $ geotessera-registry zarr-fill --help | grep -oE '\-\-(no-)?consolidate' | sort -u
  --consolidate
  --no-consolidate

Test: resume from the store itself
-----------------------------------

Shard objects outlive the bookkeeping, so a fill scans for them and skips
what is already there. Rebuilding is the opt-in:

  $ geotessera-registry zarr-fill --help | grep -o '\-\-rewrite-existing-shards' | sort -u
  --rewrite-existing-shards

Test: zarr-scan reports outstanding work
-----------------------------------------

  $ geotessera-registry zarr-scan --help | grep -c 'written, missing, or'
  1

Shards with no manifest tiles are counted separately, so percentages are
over land rather than over the zone's bounding box:

  $ geotessera-registry zarr-scan --help | grep -c 'ocean or outside coverage'
  1

  $ geotessera-registry zarr-scan --help | grep -o '\-\-output OUTPUT' | sort -u
  --output OUTPUT

The tile mirror is optional -- scanning a remote store needs only the
landmask registry for its land denominator:

  $ geotessera-registry zarr-scan --help | grep -c 'Optional tile mirror'
  1

  $ geotessera-registry zarr-scan --help | grep -oE '\[base_dir\] store_path'
  [base_dir] store_path

Test: stretch statistics are collected at fill time
----------------------------------------------------

Fills fold each shard's stretch statistics into the zone group, so the
global stretch never re-reads embeddings:

  $ geotessera-registry zarr-fill --help | grep -o '\-\-no-stretch-stats' | sort -u
  --no-stretch-stats

  $ geotessera-registry zarr-fill --help | grep -o '\-\-backfill-stretch-stats' | sort -u
  --backfill-stretch-stats

  $ geotessera-registry zarr-init --help | grep -o '\-\-stretch-sample-size' | sort -u
  --stretch-sample-size

zarr-stretch aggregates them by default, remote-capable; the legacy
shard-sampling path is the opt-in:

  $ geotessera-registry zarr-stretch --help | grep -o '\-\-from-shards' | sort -u
  --from-shards

  $ geotessera-registry zarr-stretch --help | grep -o '\-\-drift-threshold' | sort -u
  --drift-threshold

Test: zarr-extend grows the time axis
--------------------------------------

Adding a year is a metadata-only edit, and only ever appends:

  $ geotessera-registry zarr-extend --help | grep -c 'metadata-only edit'
  1

  $ geotessera-registry zarr-extend --help | grep -o '\-\-years YEARS' | sort -u
  --years YEARS

Test: zarr-consolidate finishes a sweep
----------------------------------------

  $ geotessera-registry zarr-consolidate --help | grep -c 'single-writer step'
  1

  $ geotessera-registry zarr-consolidate --help | grep -o '\-\-no-merge-registry' | sort -u
  --no-merge-registry

Test: exit status propagates
-----------------------------

A failing command must report failure so a sweep orchestrator notices:

  $ geotessera-registry zarr-consolidate /nonexistent/store.zarr > /dev/null 2>&1
  [1]
