Zarr Remote Store Tests
=======================

These cover the pieces that let `geotessera-registry zarr-fill` write to a
remote object store while streaming tiles from another one, and the state
that makes per-zone fills safe to run in parallel.

Setup
-----

  $ export TERM=dumb

Test: Location-transparent I/O and parallel-fill state
------------------------------------------------------

Everything here runs offline against a temporary directory, using `file://`
URLs to exercise the same fsspec path an `s3://` store takes:

  $ python "$TESTDIR/zarr_remote_check.py" | tail -1
  all checks passed

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

A dead sibling job's lock can be taken over explicitly:

  $ geotessera-registry zarr-fill --help | grep -o '\-\-force-lock' | sort -u
  --force-lock

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
