# Per-zone stretch statistics and the global preview pipeline

**Status:** Implemented (except the two-phase parallel preview mode, which
remains proposed; `zarr-global-preview` gained remote-source support with a
local `--output` pyramid, plus on-the-fly stretch derivation from the
statistics when none is persisted). Deviations from the draft are marked
**[as built]**.
**Scope:** `geotessera-registry` subcommands `zarr-init`, `zarr-fill`, `zarr-extend`, `zarr-stretch`, `zarr-global-preview`, `zarr-consolidate`
**Store convention:** additions to the per-zone group layout of the Tessera Zarr v3 store

## Motivation

`zarr-stretch` today computes the global RGB stretch (percentiles, optional histogram-equalisation CDF, optional 3-component PCA over all 128 bands) by reading randomly chosen shards from every zone until roughly 2 M valid pixels have been sampled. Each shard is a `(1, 128, 4096, 4096)` int8 object — 2 GiB — and typically a few hundred shards must be fetched to reach the sample target. That is on the order of a terabyte of embedding reads to derive a stretch object that serialises to a few kilobytes of root attributes. Against a remote store the cost is prohibitive and the command does not currently work at all.

The waste is structural: every one of those pixels was already resident in a fill worker's shard buffer, exactly once, at `zarr-fill` time. This specification moves stretch-statistic collection into the fill, so that the global stretch is derived from a few MiB of per-zone summaries and never re-reads embeddings.

Secondary goals folded into the same change:

- `zarr-stretch` and `zarr-global-preview` must work against remote (S3/fsspec) stores, as `zarr-init`/`zarr-fill` already do.
- `zarr-global-preview` resume markers currently live at `_preview/zone_{N}_done` in the build-state sibling; the migration below moves any legacy in-store markers there too, preserving the rule that the store contains only Zarr.
- The preview's parallelism constraints are documented precisely, with a safe default and an explicit opt-in two-phase mode.

## Design

### New arrays

Five arrays are added to every zone group `utm{zz}/`. They are ordinary Zarr v3 arrays with dimension names, consolidated with everything else — not sidecar files. `T` is the length of the zone's `time` axis; `K` is the per-zone-year sample capacity (default 20 000); `B` = `N_BANDS` = 128.

| Array | Shape | Dtype | Dimensions | Semantics |
|---|---|---|---|---|
| `stretch_stats_count` | `(T,)` | int64 | `time` | N: valid pixels contributed to the sums; -1 while invalid or updating |
| `stretch_stats_sum` | `(T, 128)` | float64 | `time, band` | S: per-band sum of dequantised values |
| `stretch_stats_prod` | `(T, 128, 128)` | float64 | `time, band, band2` | M: sum of outer products xxᵀ |
| `stretch_sample` | `(T, K, 128)` | int8 | `time, sample, band` | raw sampled embedding vectors |
| `stretch_sample_scales` | `(T, K)` | float32 | `time, sample` | per-sample dequant scale |

**[as built]** Two structural deviations. First, the number of filled slots is a sixth array, `stretch_sample_count` `(T,)` int64, not a zone-group attribute: the `geoemb:` convention keeps its attributes on the root group only, and zone groups deliberately carry nothing beyond `proj:`/`spatial:`. Unfilled slots are zero-valued with scale `+inf`, matching the existing "not yet filled" sentinel, so a reader that ignores the count still cannot mistake padding for data.

**[as built]** Second, a seventh array `stretch_stats_shards` `(T, n_shard_rows, n_shard_cols)` uint8 — the **coverage mask**, 1 where a shard's pixels are folded into the sums. It is what makes collection *automatic*: the fill's normal store scan diffed against the mask yields the shards whose statistics are missing (crash between write and fold, or shards written by older builds), and they are read back as catch-up tasks in the same worker pool. This replaced both the fill-time lock/registry state and the manual backfill workflow: fills are now fully stateless (no `--state-url`, no locks — the store is the only record), `--backfill-stretch-stats` survives only as the explicit repair for suspected double-counting, and `ensure_stretch_arrays` heals pre-feature stores on their next fill (resetting sums whose provenance the missing mask makes unknowable).

Samples are stored in **source representation** — the int8 embedding vector plus its float32 scale — which is lossless with respect to the store itself and lets any downstream statistic be recomputed exactly as if the pixels had been read from `embeddings`/`scales`.

### Semantics of the additive statistics

For each (zone, year), over every valid pixel x = int8 vector × scale (float64 accumulation):

- `count` N — number of valid pixels (scale finite, i.e. not NaN water, not +inf unfilled),
- `sum` S = Σ x (length 128),
- `prod` M = Σ x xᵀ (128 × 128).

These are the sufficient statistics for mean and covariance and are additive across zones and across shards within a zone. Global aggregation is exact:

```
μ = ΣS / ΣN
Σcov = ΣM / ΣN − μ μᵀ
```

### Size budget

Per (zone, year): 8 + 1 024 + 131 072 bytes of statistics (≈ 0.13 MiB) plus 20 000 × 132 bytes of sample (≈ 2.5 MiB) ≈ **2.6 MiB**. For 60 zones × 9 years the whole-world total is ≈ **1.4 GiB**, negligible against the multi-terabyte store. For one year across all 60 zones, the additive statistics alone are ≈ 8 MiB; statistics plus samples ≈ 160 MiB — a few seconds from S3 either way.

Chunking: one chunk per year on the `time` axis for all five arrays (matching the existing one-year-per-chunk convention), so a (zone, year) update touches exactly one chunk per array and `zarr-extend` can append years as a metadata-only edit.

### Creation: at init, not lazily

The arrays are created by `zarr-init`, alongside `embeddings` and `scales`. Lazy creation by `zarr-fill` was rejected: fills run in parallel across zones but the *decision* to create arrays would race with `zarr-consolidate` snapshots and would leave stores in mixed states that every reader must handle. With init-time creation, a store either predates the feature entirely (handled by backfill, below) or has the full set. `zarr-extend` grows all five arrays' `time` axis when appending years, exactly as it does for `embeddings`.

## Statistical soundness

The design rests on three empirically verified facts (all measured on real Tessera tile data):

1. **Sufficient statistics give exact PCA.** Eigenvectors computed from summed (N, S, M) match a full-population PCA at |cos| = 1.000000 per component — exact to float64 — whereas a 20 k-pixel sampled PCA achieves |cos| = 0.9977. Summation beats sampling because no information is discarded: the covariance *is* the sufficient statistic. Every valid pixel in the store contributes.

2. **Quantiles are not additive**, and worse, the quantities we need quantiles of — the principal-component projections — do not exist until the global PCA axes are known, which happens only after cross-zone aggregation. Hence the raw 128-dimensional samples: at stretch time they are projected onto the freshly derived PCs and percentiles/CDF are computed in PC space from the pooled sample.

3. **Sampled quantiles are accurate enough.** Measured worst-case error of sampled p2/p98 against the full population of a real tile, expressed as a fraction of the stretch span over 20 trials: 5 000 samples → 3.89 %, 20 000 → 1.90 %, 50 000 → 2.28 % (no further improvement past ~20 k; the residual is population tail noise). Default **K = 20 000** per (zone, year), giving a pooled global sample of up to 1.2 M pixels — comparable to today's 2 M target, but drawn once and stored.

**Weighting.** Samples must be drawn proportionally to each shard's valid-pixel count. Uniform per-shard draws would over-represent sparse coastal shards (a shard with 500 valid pixels would carry the same weight as a full 16.7 M-pixel interior shard). The implementation uses a weighted reservoir over the zone's shards: each worker draws a per-shard sub-sample proportional to its valid count, and the parent merges reservoirs with weights, so the final K samples approximate a uniform draw over all valid pixels of the (zone, year).

## Fill-time collection

Collection is free in I/O terms: workers already hold every decoded pixel in the 2 GiB shard buffer.

**Worker side.** After assembling a shard and before writing it, the worker computes the shard's (n, s, m) triple over its valid pixels and draws its weighted pixel sub-sample (int8 vectors + scales). This adds one 128×128 GEMM-style pass per shard — negligible against tile decode — and a bounded few MiB of memory, preserving the ~6.2 GiB (or ~4.1 GiB with `--spill-dir`) per-worker budget. Partial results are returned to the parent over the existing result queue.

**Parent side.** The parent process owns the (zone, year) advisory lock in the state dir for the duration of the fill. It sums the (n, s, m) triples, merges the reservoirs, and — once the (zone, year) sweep completes — performs a **single write per (zone, year)** into the five arrays plus the `geoemb:stretch_sample_count` attribute. Each write touches one chunk per array (one-year chunking), so the update is as atomic as Zarr offers; the lock guarantees no concurrent writer for that (zone, year).

**Cross-zone safety.** Parallel `zarr-fill --zones N` processes touch disjoint zone groups, disjoint locks, and never the root metadata, exactly as today. The stats arrays live inside the zone group, so the existing parallelism contract is unchanged.

**Resumed fills.** Complete shards have both embeddings and scales. The coverage mask identifies complete shards whose statistics have not been collected; the fill reads them back as catch-up tasks. A negative `stretch_stats_count` means an update or rewrite was interrupted, so the fill rebuilds that zone/year from complete shards before using its statistics.

## Aggregation: the `zarr-stretch` fast path

When the stats arrays are present (detected from consolidated metadata), `zarr-stretch` defaults to:

1. Read `stretch_stats_{count,sum,prod}` for the requested year from every selected zone (≈ 8 MiB for the world; seconds against S3).
2. Sum: Nᵍ = ΣN, Sᵍ = ΣS, Mᵍ = ΣM. Compute μ = Sᵍ/Nᵍ and Σcov = Mᵍ/Nᵍ − μμᵀ in float64.
3. Eigendecompose Σcov; take the top `--pca-components` eigenvectors (for `--mode pca`; `--mode bands` skips PCA and uses bands 0–2, with percentiles still computed from the pooled sample).
4. Read `stretch_sample`/`stretch_sample_scales` for the year (≈ 150 MiB for the world), dequantise, pool across zones (each zone's sample already approximates a uniform draw of that zone; pooling weights zones by their filled sample counts), project into PC space.
5. Compute `--p-low`/`--p-high` percentiles and, unless `--no-equalise`, the per-channel CDF with `--breakpoints` breakpoints.
6. Run the **drift check** (see Caveats) and write the result to the root attribute `geoemb:stretch.{year}`.

Total runtime: seconds to low minutes, dominated by S3 GETs of small chunks. This is a **single-writer** step (it rewrites root attributes): run it after all zone fills have finished and before any preview, never concurrently with `zarr-consolidate`.

`--from-shards` forces the legacy shard-sampling mode for stores without stats (and remains useful as an independent cross-check). If stats arrays are absent the command falls back to legacy mode with a warning.

## Caveats and failure modes

**Shard rewrites.** Before overwriting shards, the fill sets `stretch_stats_count` to -1. After successful writes it rebuilds the affected zone/year statistics from the store, replacing the old sums, sample, and coverage mask. This costs one additional read of that zone/year but avoids per-shard statistics storage and double-counting. With `--no-stretch-stats`, the count remains negative until a later fill or explicit backfill repairs it; `zarr-stretch` refuses incomplete statistics. The covariance drift check remains an independent check of the resulting aggregates.

**Stores initialised before this feature.** Their zone groups lack the arrays. `zarr-fill` gains `--backfill-stretch-stats`: for the selected zone(s) it creates the five arrays (under the zone lock; a zone-group edit, not a root edit, so it composes with parallel fills of *other* zones) and populates them by scanning that zone's existing shards once — this is the one path that does re-read embeddings, but it is per-zone, opt-in, and runs at full shard-streaming bandwidth. A subsequent `zarr-consolidate` publishes the new arrays.

**Interrupted fills.** Updates publish a negative count before changing sums or samples and publish the final count last. If interrupted, the next fill rebuilds the affected zone/year. Shards written before any statistics update are handled by the normal coverage-mask catch-up.

**`zarr-extend`.** All seven statistics arrays grow with the data arrays. The zone records intended coordinates in `tessera:pending_years` before resizing; retries finish those coordinates even if a prior time-array write failed. Legacy trailing zero coordinates are reused. The pending attribute is removed after all coordinate writes succeed. Consolidation runs even when every requested year is already present, so retrying also repairs an interrupted final consolidation.

**Preview marker migration.** Older `zarr-global-preview` runs wrote `.zone_{N}_done` markers inside the store. Markers now live at `<state>/preview/{year}/zone_{N}_done`. On startup the command migrates any legacy in-store markers into the state dir and deletes them from the store, restoring the "store contains only Zarr" invariant.

## CLI surface

The running example is the source.coop deployment. Tile source: the public mirror `s3://tessera/tessera` via the gateway (`--source-endpoint-url https://data.source.coop --source-anon`), or in-region anonymous reads of the backing bucket `s3://us-west-2.opendata.source.coop/tessera/tessera` (`--source-anon --source-region us-west-2` — preferred for large fills, no gateway hop). Store: `s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1`, written with credentials (`--store-profile sc-writer --store-region us-west-2 --store-acl bucket-owner-full-control`). Note there is **no** `--store-endpoint-url`: `data.source.coop` is a read-only gateway and writes go straight to the backing bucket.

```sh
STORE=s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1
SRC=s3://us-west-2.opendata.source.coop/tessera/tessera
SRCFLAGS="--source-anon --source-region us-west-2"
STOREFLAGS="--store-profile sc-writer --store-region us-west-2 --store-acl bucket-owner-full-control"
STATE="--state-url s3://sc-build-state/tessera-v1.build"
```

### `zarr-init` — changed

Unchanged interface; now also creates the five stats arrays (year-chunked, `K` from `--stretch-sample-size`, default 20 000, recorded in zone attrs) in every zone group.

```sh
geotessera-registry zarr-init "$SRC" --years 2017-2025 --output "$STORE" \
    $SRCFLAGS $STOREFLAGS $STATE
```

### `zarr-fill` — changed

Existing behaviour and flags (`--year`, `--zones`, `--workers`, `--spill-dir`, `--rewrite-existing-shards`, `--force-lock`, `--state-url`, source/store storage flags) unchanged. New behaviour: accumulates (N, S, M) and the weighted sample in workers, aggregates in the parent, writes once per (zone, year) at sweep end. New flags:

| Flag | Meaning |
|---|---|
| `--no-stretch-stats` | Skip collection; rewrites invalidate old aggregates until a later fill or backfill |
| `--backfill-stretch-stats` | Create/rebuild the stats arrays for the selected zones by scanning existing shards; implies no tile ingestion |

```sh
# one zone per VM / process; safe in parallel across zones
geotessera-registry zarr-fill "$SRC" "$STORE" --zones 30 --workers 4 \
    --spill-dir /scratch/spill $SRCFLAGS $STOREFLAGS $STATE
```

### `zarr-stretch` — changed

New default: stats fast path when the arrays are present (detected per zone; zones lacking them are reported and skipped, or the run aborts under `--strict`). Existing flags (`--year`, `--p-low`, `--p-high`, `--no-equalise`, `--breakpoints`, `--mode`, `--pca-components`, `--pca-total-bands`, `--pca-rgb-order`, `--zones`) keep their meaning. `--target-samples`/`--max-shards`/`--workers` apply only to legacy mode. Gains store storage flags so it works remotely. New flags:

| Flag | Meaning |
|---|---|
| `--from-shards` | Force the legacy shard-sampling path |
| `--drift-threshold` | Maximum relative Frobenius distance between stats- and sample-derived covariances before warning (default 0.25; the limit is never below 3× the sample noise floor) |

```sh
geotessera-registry zarr-stretch "$STORE" --year 2024 --mode pca $STOREFLAGS
```

(`zarr-stretch` keeps no build state, so it takes no `--state-url`.)

### `zarr-global-preview` — changed

Gains store storage flags and `--state-url`; resume markers move to the state dir (with in-store legacy migration). Zone-level parallelism over the shared pyramid is **unsafe**: adjacent zones share level-0 edge chunks (read-modify-write composites) and coarse levels overlap almost totally. Two modes:

- **Default (safe): sequential zones**, exactly today's semantics, now remote-capable.
- **Two-phase (opt-in):** phase 1, `--reproject-only --zones N`, writes a zone's *interior* level-0 chunks and stages its edge-column contributions in the state dir; runnable in parallel across zones. Phase 2, `--coarsen-only`, run once after **all** zone level-0 writes are complete: composites the staged edge columns into the shared level-0 edge chunks, then builds levels 1–9. Running `--coarsen-only` before every zone has finished produces a silently incomplete pyramid; the command refuses unless every selected zone's phase-1 marker exists (`--force` overrides).

```sh
# safe default
geotessera-registry zarr-global-preview "$STORE" --year 2024 \
    --gamma 0.7 --saturation 1.8 $STOREFLAGS $STATE

# opt-in two-phase
for z in $(seq 1 60); do
  geotessera-registry zarr-global-preview "$STORE" --year 2024 \
      --reproject-only --zones $z $STOREFLAGS $STATE &   # parallel, one per slot
done; wait
geotessera-registry zarr-global-preview "$STORE" --year 2024 \
    --coarsen-only $STOREFLAGS $STATE                    # single-writer
```

### `zarr-consolidate` — unchanged

Single-writer; merges per-zone registries and rewrites root consolidated metadata (which now includes the stats arrays). Run when no fill or preview is in flight.

## End-to-end runbook

Fresh store, 2017–2025, source.coop deployment. Parallel-safety and cost annotated per step.

1. **Init** — single-writer, minutes.
   ```sh
   geotessera-registry zarr-init "$SRC" --years 2017-2025 --output "$STORE" \
       $SRCFLAGS $STOREFLAGS $STATE
   ```
2. **Fill, one zone per slot** — parallel-safe across zones; hours per zone, memory-bound (~6.2 GiB/worker, ~4.1 GiB with `--spill-dir`). On one large box:
   ```sh
   parallel -j 8 geotessera-registry zarr-fill "$SRC" "$STORE" \
       --zones {} --workers 4 --spill-dir /scratch/spill \
       $SRCFLAGS $STOREFLAGS $STATE ::: $(seq 1 60)
   ```
   Across VMs, give each VM a zone range (`--zones 1-8`, `--zones 9-16`, …). Fills are resumable: re-running skips existing shards.
3. **Consolidate** — single-writer, minutes; publishes filled arrays and stats.
   ```sh
   geotessera-registry zarr-consolidate "$STORE" $STOREFLAGS $STATE
   ```
4. **Stretch (fast path)** — single-writer (root attrs), seconds–minutes per year.
   ```sh
   for y in $(seq 2017 2025); do
     geotessera-registry zarr-stretch "$STORE" --year $y --mode pca $STOREFLAGS
   done
   ```
5. **Global preview** — sequential default (hours; the remaining sequential bottleneck) or the two-phase opt-in above (phase 1 parallel, phase 2 single-writer).
   ```sh
   geotessera-registry zarr-global-preview "$STORE" --year 2024 \
       --gamma 0.7 --saturation 1.8 $STOREFLAGS $STATE
   ```
6. **Final consolidate** — single-writer, minutes; publishes the pyramid.
   ```sh
   geotessera-registry zarr-consolidate "$STORE" $STOREFLAGS $STATE
   ```

Appending a year later: `zarr-extend --years 2026` (single-writer, no fills in flight; grows all arrays including stats) → step 2 for the new year → steps 3–6.

## Open questions

1. **K tuning.** 20 k/zone-year is justified by the p2/p98 error measurements, but equalisation CDFs with 257 breakpoints may benefit from larger pooled samples. Should `--stretch-sample-size` be raised for stores intended primarily for equalised previews?
2. **Sample refresh policy on partial re-fills.** The reservoir-replacement rule keeps the sample duplicate-free but means a small re-fill can churn slots contributed by unrelated shards. Is slot churn acceptable, or should re-fills merge weighted against the stored `geoemb:stretch_sample_count`?
3. **Water/land class statistics.** N/S/M cover valid land pixels only. Per-class counts (water, unfilled) per (zone, year) would be nearly free to collect and useful for coverage dashboards — worth adding now while the array set is being defined?
4. **Preview edge-staging format.** ~~Phase 1 stages edge-column composites in the state dir; the serialisation is unspecified.~~ **[likely resolved]**: staging may be unnecessary. The contended region is only the chunk column containing each zone boundary meridian (~1–2 columns; a zone's data pixels stay within its own 6° even though its iterated rectangle bulges far wider), so only *immediate* neighbours conflict and the conflict graph is 2-colourable. Scheduling zones in two waves (odd zones in parallel, then even) gives 30-way parallelism with no concurrent writer per chunk and no staged state — only the single-writer `--coarsen-only` barrier survives. To verify at implementation time: whether resampling smear at extreme latitudes can push a zone's data more than half a chunk past its meridian, which would require a third wave.
5. **Drift-check threshold.** ~~0.99 |cos|~~ **[resolved as built]**: the metric changed to covariance Frobenius distance with a noise-floor-aware limit, after eigenvector comparison false-alarmed on near-isotropic data. The 0.25 default should still be revisited once real rewrite workloads exist.
