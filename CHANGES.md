## Unreleased

### Breaking Changes
- **`convertv2.sh STRETCH_STATS=0`**: skip fill-time stretch statistics. v2's
  preview reads bands 0-2 of `embeddings_d4` directly, so the statistics that
  exist to support a 128-band PCA earn much less there. (@avsm)

- **`zarr-init --no-landmask`**: for datasets whose inference covers every
  pixel of a tile it emits, so a present tile is data all the way to its
  edges. No landmask registry is fetched and no landmask GeoTIFF is read
  during a fill; the mask would be all ones, so consulting it is a per-tile
  round trip that can only confirm what the tile's presence already says. The
  flag is recorded on the root as `geoemb:landmask: false` and `zarr-fill`
  follows the store, so the two can never disagree. Absent means masked, which
  is every store predating the flag.

  The zone grid is then sized from the embeddings rather than the landmask.
  That is required, not an optimisation: the published `landmasks/v2` is a
  copy of v1.1's and stops at lat -59.55 while v2 reaches -89.95, so sizing
  from it put 880 Antarctic tiles outside the grid — 49 of 60 zones failed
  with `IndexError: index 376 is out of bounds for axis 0 with size 315` from
  the stretch bookkeeping, after the shard writes had already run. Measured
  over six v2 zones: 83 tiles outside the landmask-derived grid, 0 outside the
  embedding-derived one.

  The scale sanity check still runs. Rejecting out-of-range scales written
  into a tile is a different thing from masking water, and only the latter is
  the landmask's job. (@avsm)

- **Convention metadata now comes from `zarr-cm`**, replacing
  `geozarr-toolkit`. Stores stamp `spatial:` and `proj:` at revision **r3**
  and `multiscales` at **r2** (no r3 exists upstream), each pinned to the
  spec commit that defined it. The previous `refs/tags/v1` URLs were dead —
  every schema URL written by earlier builds returns 404, since those tags
  were never cut and `geo-proj` has since moved to `zarr-conventions/proj`.
  Convention UUIDs and every `spatial:`/`proj:`/`multiscales` attribute
  keep their existing names and shapes, so readers are unaffected; only the
  `spec_url`/`schema_url` registrations change. Existing published stores
  keep their dead URLs until their metadata is rewritten. Drops the
  `pydantic` and `structlog` transitive dependencies. (@avsm)
- **`zarr-init --matryoshka-depths 4,16`**: also store the first N dimensions
  of every embedding as their own arrays, so a client can read a 4- or
  16-dimensional prefix without decoding all 128 bands. `embeddings` is one
  chunk wide on the band axis, so a prefix read from it currently costs a full
  read and discards 96.9% of what it decoded. Requires matryoshka-ordered
  dimensions and is refused for v1/v1.1, whose dimensions are not ordered by
  importance — a prefix of those is an arbitrary slice, and the store would
  look correct while being meaningless.

  The depth arrays share the shard grid with `embeddings` exactly, so one
  source read fills them all from the same buffer and one shard coordinate
  addresses the same pixels everywhere; only the inner chunk differs
  (`128x128` at depth 4, `64x64` at 16), sized to hold chunk bytes at or below
  the depth-128 budget. Keeping `32x32` at depth 4 would have meant a 256 KiB
  sharding-index fetch to reach a 1-2 KiB chunk. `scales` is *not* duplicated:
  quantisation is per-pixel, so every depth dequantises against the same
  array. Depths are written before the full depth so that "the `embeddings`
  shard exists" implies every prefix exists, leaving `_existing_shards` the
  single resume oracle. Storage overhead measured at 16.2%. `zarr-fill`,
  `zarr-extend` and `zarr-scan` all follow the root's `geoemb:depths`
  declaration. See `docs/specs/zarr-matryoshka-depths.md`. (@avsm)
- **`zarr-stretch --from-sample`**: take the covariance from the stored
  per-zone reservoir instead of the summed sufficient statistics. The sums
  are exact but unrepairable in place once a few out-of-range pixels have
  dominated them — the only other remedy is a full rescan of the store. A
  uniform reservoir almost never contains such pixels, and 1.2M pooled
  samples is ample for a 128x128 covariance, so this recovers a usable
  stretch in seconds. The drift figure still reports how far the sums have
  strayed rather than comparing the sample against itself. (@avsm)
- **`zarr-global-preview --state-url`**: per-zone resume markers no longer
  have to sit in `<output>.build` beside the pyramid, so a build writing to a
  published bucket can keep its bookkeeping on local disk. Takes the usual
  `--state-*` object-store flags. (@avsm)
### Bug Fixes

- **Out-of-range scales no longer poison the stretch statistics**:
  `MAX_VALID_SCALE` was `1e6`, set to reject only the `~FLT_MAX` sentinel and
  pass everything below. Measured over the published v1 store's 1.2M pooled
  sample pixels, real scales run median 0.064 / p99.9 0.119 / p99.99 0.137,
  and only 5 in 1.2M exceed 1.0 — but pixels with scales just under `1e6`
  survived the filter and contributed ~1e16 apiece to the second moment,
  poisoning **47 of 60 zones** (max|prod|/n of 1e6–1.9e8 against 27–653 for a
  clean zone) and so every colour derived from the PCA. The limit is now
  `1.0`: seven times the p99.99 of real data, six orders of magnitude below
  the corrupt values. The pooled sample is also re-filtered on read, since a
  reservoir written under the old limit can still hold a few. (@avsm)
- **`zarr-stretch` refuses to persist a stretch that fails its own drift
  check** unless `--allow-drift`. It previously warned and saved anyway,
  which is how a covariance known to be wrong reached the store and then
  every preview built from it. (@avsm)
- **`zarr-global-preview` no longer races itself creating the pyramid**:
  zarr's create is check-then-write, so a parallel zone sweep had several
  callers pass the existence check before any wrote, and the losers died with
  `A group exists ... at path 'global_rgb/2'` or `An array exists ... at path
  'global_rgb/0/rgb'`. Every creation step is now idempotent, so all callers
  build the identical structure and converge. (@avsm)
- **Antimeridian zones no longer coarsen the whole grid width**: the
  reprojection work list was tightened (below), but the coarsening still took
  a single enclosing rectangle, which for a zone with chunks at both grid
  edges spans every column — 16.0M chunk slots for utm01's 18.8k real chunks
  and 16.7M for utm60's 37.8k, each one read and rewritten by
  `_coarsen_tile`. Footprints now split on a column gap wider than half the
  grid, giving two tight rectangles: utm01 falls to 347k slots (853x -> 18.5x
  overhead) and utm60 to 364k (442x -> 9.6x), with every other zone still a
  single rectangle and bit-identical. (@avsm)
- **Antimeridian shards no longer enqueue the whole globe**: a shard
  straddling 180° samples corners near -180 and +180, and taking the naive
  min/max of those made `_chunks_for_shards` claim every chunk column at that
  latitude. utm60 enqueued 1,395,753 level-0 chunks from 360 shards (~3,900
  per shard, against ~65 for a normal zone) and utm01 877,648 from 146 — some
  22% of a year's reprojection work list, all of it reprojecting to nothing.
  The wrap is now detected by re-measuring the span with longitudes shifted
  to `[0, 360)` and split into two column ranges; a polar shard, which
  genuinely does span most longitudes, keeps the full range. For 2024 this
  cuts utm60 to 37,791 chunks and utm01 to 18,766, leaves every other zone
  bit-identical, and reduces the cross-zone conflict graph from 172 pairs to
  60 — 59 adjacent plus utm01↔utm60, which really are neighbours. (@avsm)
- **Pyramid levels keep their per-level geometry**: `zarr-global-preview`
  computed `spatial:shape` and `spatial:transform` for each multiscale level,
  but `geozarr-toolkit`'s layout builder silently dropped both, so every
  level advertised only its scale factor. `zarr-cm` preserves them, and the
  multiscales schema permits them (`additionalProperties: true`). (@avsm)

- **Remote zarr builds**: `zarr-init` and `zarr-fill` now take locations
  rather than paths — both the tile source and the output store may be
  fsspec URLs (`s3://bucket/prefix`), so a store on one S3 node can be
  filled from tiles on another with no local mirror. Remote tiles are read
  with byte-range GETs sized to the rows each shard needs (an `.npy` is a
  header plus a flat C-ordered buffer), so no scratch disk is involved.
  `--source-*` and `--store-*` flags configure the two endpoints
  independently; credentials come from the environment, a named profile
  (`--store-profile`), or an instance role rather than argv, and
  `--store-acl` stamps a canned ACL such as `bucket-owner-full-control` on
  every object written. `s3://` locations need the new optional
  `s3` extra (`pip install 'geotessera[s3]'`), which is what pulls in
  `s3fs` and `botocore`; the core install stays free of both and `https://`
  sources work without it. (@avsm)
- **Parallel per-zone fills**: `zarr-fill --zones N` is now safe to run as
  one process per UTM zone against a shared store. Ingestion tracking is one
  object per zone/year, each zone/year takes an advisory lock
  (`--force-lock` to take over a dead run's), and root-metadata
  consolidation is skipped by default for a zone-restricted fill. See
  the architecture guide for the sweep recipe. (@avsm)
- **The store now contains only Zarr**: build bookkeeping — the ingestion
  registry, fill locks and global-preview resume markers — moved out of the
  store into a sibling location, `<store>.build` by default and relocatable
  with `--state-url`. Previously these sat at the store root, where every
  hierarchy listing and `consolidate_metadata` call warned about
  unrecognised objects and readers saw non-Zarr entries. A `_registry.parquet`
  left inside an older store is still read, so existing stores resume
  correctly; nothing is written back into them. (@avsm)
- **`geotessera-registry zarr-scan`**: New subcommand that inventories a
  store's shards without writing anything, classifying each as `written`,
  `missing`, or `empty` (no land falls in it, so it is ocean or outside
  coverage and will never be filled). Prints per-zone/year and per-year
  summaries of how much is left to fill — percentages are over land, not
  over each zone's bounding box — and optionally writes the per-shard index
  as parquet. Takes the store alone: the land denominator comes from the
  landmask registry (~19 MB, cached), so no tile mirror or manifest is
  needed. An optional tile mirror switches the denominator to each year's
  actual embedding coverage from the manifest. (@avsm)
- **`zarr-fill` scans before writing and uploads only what is missing.**
  The ingestion registry is written only when a (zone, year) finishes, so a
  run killed partway loses that year's bookkeeping and would re-upload
  everything — for a zone that is 97% done, rebuilding 1,398 shards to add
  48. The shard objects survive anything and a shard is always written from
  every tile covering it, so their presence is proof of completeness. Fills
  now scan for them by default and skip what is there; falls back to
  probing just the shards in hand where the credentials cannot list the
  store. `--rewrite-existing-shards` forces a rebuild, needed only when the
  tile inventory has grown, since a newly-added tile falls inside an
  existing shard. `--skip-existing-shards` is still accepted as a no-op.
  (@avsm)
- **Sentinel scales no longer count as data.** Some published v1 scales
  files carry a huge-finite nodata sentinel (~FLT_MAX) that passes
  `isfinite()`: those pixels inflated valid-pixel counts up to 100x,
  drove stretch sums to 1e37 and overflowed the product matrices to inf —
  which surfaced as a `nan` drift check and garbage PCA. Validity is now
  finite, positive and below `MAX_VALID_SCALE` (1e6), applied at stats
  collection, at fill time (the store now records such pixels as NaN
  nodata), and in preview rendering. `zarr-stretch` refuses poisoned
  statistics outright, naming the zones to rebuild with
  `--backfill-stretch-stats`; statistics collected before this fix need
  that rebuild (shards themselves are unaffected). The aiohttp
  "Unclosed client session" destructor noise that flooded preview output
  is also silenced. (@avsm)
- **`zarr-global-preview` streams from remote stores.** The pyramid source
  and destination are now separate: `--output <local-store>` builds the
  EPSG:4326 RGB pyramid locally while reading zone embeddings straight from
  a remote store (anonymously or with credentials) as sub-shard byte
  ranges — no copy of the source is made. If the store has no persisted
  `geoemb:stretch` for the year, one is derived on the fly from the
  per-zone stretch statistics (a few MiB of reads, not persisted), so a
  read-only consumer can preview a store it cannot write to without any
  prior `zarr-stretch` step. Reprojection workers now spawn rather than
  fork, for the same fsspec event-loop reason as the fill workers. The
  work list is now derived from the footprints of the shards that exist,
  not the zone's bounding rectangle — at high latitude a zone's rectangle
  back-projects across most longitudes, and utm02 enqueued ~5.6 million
  candidate chunks (22 GiB of queued futures) to render 28 shards of
  actual data. (@avsm)
- **Fills are stateless and stretch statistics collect themselves.** The
  ingestion registry and advisory locks are gone from `zarr-fill`: the
  store's shard objects are the only record of progress, so a preemptible
  (spot) instance that dies mid-run leaves nothing to clean up or take
  over — relaunching the same command scans and continues. A per-zone
  coverage mask (`stretch_stats_shards`) records which shards are folded
  into the stretch sums; the fill diffs it against the same scan and reads
  back any shard whose statistics are missing, so interrupted runs and
  stores from older builds converge automatically — no separate backfill
  step, which now exists only as the explicit repair for suspected
  double-counting. `--state-url` and `--force-lock` are accepted as no-ops
  for existing scripts; `zarr-consolidate` still reads `--state-url` to
  merge registries written by older builds. (@avsm)
- **Per-zone stretch statistics, collected at fill time** (see
  `docs/specs/zarr-stretch-stats.md`): each zone group gains six arrays —
  exact mean/covariance sufficient statistics per (zone, year), additive
  across zones, plus a weighted 20k-pixel sample for quantiles — folded in
  by `zarr-fill` from the shard buffers it already holds, at no extra I/O.
  `zarr-stretch` now aggregates these by default: a few MiB of reads and an
  *exact* global PCA (verified |cos| = 1.0 against a full-population fit)
  instead of terabytes of shard re-reads, and it works against remote
  stores. A drift check compares the stats-derived covariance against one
  refitted from the stored sample and warns when rewritten shards have
  double-counted. `--from-shards` keeps the legacy path;
  `zarr-fill --backfill-stretch-stats` rebuilds statistics for stores
  filled before this existed (and is required before `zarr-extend` will
  touch such stores); `zarr-init --stretch-sample-size` tunes the sample.
  (@avsm)
- **`zarr-fill` reports the shard arithmetic**: `Shards: 1,373 land, 48
  recorded done, 43 found in store, 1,282 to write`. The previous line
  showed only the last figure, which could not be reconciled with
  `zarr-scan` — that counts every land shard, whereas a fill considers only
  those covering tiles the registry has not already recorded. (@avsm)
- **Object-store libraries no longer log over the progress bar**: botocore
  logs "Found credentials in shared credentials file" at INFO every time a
  client is built, once per worker process, and the workers share a
  terminal with the progress bar. botocore, boto3, aiobotocore, s3fs,
  urllib3 and aiohttp are now capped at WARNING in both the parent and the
  workers. (@avsm)
- **Fills no longer deadlock against an object store.** `s3fs` runs its
  client on a background event-loop thread and `fsspec` caches both the loop
  and its filesystem instances globally; a forked worker inherits those
  objects but not the thread running the loop, so its first call to the
  store waits forever (main thread in `futex_do_wait`, loop thread idle in
  `ep_poll`). Workers are now started with "spawn" rather than the Linux
  default of "fork", and reset any inherited fsspec state on startup.
  (@avsm)
- **Workers die with their parent.** A fill killed outright left its
  workers running, each holding gigabytes and accumulating across runs —
  19 orphans holding 8 GB were observed on one host. Workers now set
  `PR_SET_PDEATHSIG` on Linux. (@avsm)
- **Ctrl-C during a fill exits cleanly** with status 130 instead of two
  tracebacks. The process pool was shut down with `wait=True`, so an
  interrupt blocked until every in-flight shard finished — with
  multi-gigabyte workers that looks like a hang and invites a second
  Ctrl-C. (@avsm)
- **`zarr-fill` warns when the worker count will not fit in memory**, using
  `MemAvailable` rather than physical RAM since these hosts are usually
  shared. The estimate covers the real peak — a worker holds a 2.1 GiB
  shard buffer, then zarr's sharding codec compresses every inner chunk and
  assembles the shard while s3fs holds the upload body, measured at 4.3 GiB
  and observed being OOM-killed at 12 GiB — so it budgets ~6.2 GiB each
  rather than the raw buffer. An OOM kill leaves no traceback, so the
  warning is the only diagnosis. (@avsm)
- **`geotessera-registry zarr-extend`**: New subcommand that appends years
  to an existing store's time axis, so a new year can be added without
  rebuilding. Time is chunked one year per chunk, making this a
  metadata-only edit — existing chunks are never rewritten — and the new
  slice reads back with the same sentinels a freshly initialised year has.
  Years may only be appended (inserting an earlier one would renumber every
  chunk, so it is refused), and it will not run while a fill lock is held.
  (@avsm)
### Bug Fixes

- **Incremental fills no longer erase neighbouring tiles**: a shard write
  replaces the whole shard, so a fill that touched a shard already holding
  data would zero out the tiles it did not re-read. Touched shards are now
  rebuilt from every tile overlapping them. (@avsm)
- **Failed shards are no longer recorded as written**: tiles are only added
  to the ingestion registry once every shard covering them succeeded, so
  re-running a fill retries exactly the unfinished work. A fill with any
  failed shard now reports an error. (@avsm)
- **`geotessera-registry` propagates exit status**: command return codes
  were discarded, so failures reported success to the shell. (@avsm)

- **`zarr-consolidate` merges registries and works remotely**: the
  subcommand (introduced in 0.10.0) now also merges the per-zone ingestion
  registries into `_registry.parquet` — the single-writer step that
  finishes a parallel sweep — and accepts a remote store URL as well as a
  local path. (@avsm)

## 0.10.0 (2026-08-20)

### Breaking Changes

- All NPY downloads now come from the Source Cooperative, fronted by CloudFlare.
  Embeddings, landmasks and manifests are served from the public
  `https://data.source.coop/tessera/tessera` repository over HTTPS,
  replacing the retired `tessera-embeddings` AWS S3 bucket.

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
