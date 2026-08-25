# Pipeline scripts

Standalone drivers for building and publishing the Tessera Zarr stores on a
deployment host. Each wraps a sequence of `geotessera-registry zarr-*`
subcommands with per-zone parallelism, retries and resume, so an interrupted
run can simply be re-run. They are configured entirely through environment
variables; the defaults target the published Source Cooperative repository.

Every script verifies up front that the installed geotessera carries the
fixes it depends on, and refuses to run otherwise — an older build would
write a plausible-looking but wrong store.

## zarr-convert.sh — npy → Zarr, v1/v1.1

Builds a Zarr store from a published npy dataset (defaults to
v1.1/cambridge). Installs geotessera into its own venv with `uv`, so the
host needs only `uv`, `curl` and AWS credentials, then runs `zarr-init`,
a pool of per-zone `zarr-fill` processes, and a final `zarr-consolidate`.

```sh
./scripts/zarr-convert.sh                                # v1.1 cambridge
VERSION=v1 VARIANT=vultr YEARS=2017-2025 ./scripts/zarr-convert.sh
```

Key variables: `VERSION`/`VARIANT`/`YEARS` select the dataset; `REF` pins
the git ref to install; `PROFILE` names the AWS profile with write access;
`ZONE_JOBS`×`WORKERS` size the fill pool against RAM (not cores);
`SRC`/`DST` override the source repository and destination store.

## zarr-convert-v2.sh — npy → Zarr, v2

The same pipeline for the v2 `2B-L~beta1` dataset, with the v2 differences
baked in: no landmask (v2 inference covers every pixel of a tile it emits),
nested matryoshka depth arrays (`MATRYOSHKA_DEPTHS`, default `4,16`), and
optional fill-time stretch statistics (`STRETCH_STATS=0` to skip — v2
previews read bands 0–2 of `embeddings_d4` directly, so the statistics
earn much less than for a 128-band PCA).

```sh
./scripts/zarr-convert-v2.sh                             # the whole dataset
ZONES="30 31 32" ./scripts/zarr-convert-v2.sh            # just those zones
```

## zarr-sync.sh — incremental npy → manifest → Zarr → preview

Thin wrapper over `geotessera-registry s3sync`, which owns the whole
incremental pipeline: rescan the bucket for tiles, diff against the
manifest as of the last successful sync, publish the fresh manifest, fill
only the `(zone, year)` pairs that gained or changed tiles
(`--rewrite-existing-shards`, since a new tile can fall inside an existing
shard), and re-render just the changed zones of the preview year's
pyramid. A run that finds nothing new stops after the scan; an interrupted
run is simply re-run. On a store that does not exist yet, the first run is
the full build (init + fill + preview).

```sh
./scripts/zarr-sync.sh                                   # v2 2B-L~beta2
DRY_RUN=1 ./scripts/zarr-sync.sh                         # plan only
VARIANT=2B-L~beta1 ZONE_JOBS=30 WORKERS=8 ./scripts/zarr-sync.sh
```

Key variables: `VERSION`/`VARIANT` select the dataset; `PREVIEW_YEAR`
(default 2024) the published pyramid year; `REF` pins the git ref to
install; `PROFILE` the AWS profile with write access; `ZONE_JOBS`×`WORKERS`
size the fill pool; `WORKDIR` holds the sync baseline, logs and preview
markers — point every run at the same one. Anything after the environment
is forwarded: `./scripts/zarr-sync.sh --force-rewrite` etc.

New *years* are deliberately not appended: run
`geotessera-registry zarr-extend` by hand, and the deferred tiles sync on
the next run.

## zarr-preview.sh — global RGB preview pyramid

Renders one year of a store into the EPSG:4326 RGB pyramid at
`<store>/global_rgb`: per-zone stretch statistics (optional), a global
`zarr-stretch`, per-zone stretches blended across zone boundaries, then a
neighbour-aware parallel `zarr-global-preview --reproject-only` sweep over
all 60 zones and a single-writer `--coarsen-only` pass. Unlike the convert
scripts it never installs anything: it runs against an existing venv
(`VENV`, default `~/.venvs/geotessera`).

```sh
./scripts/zarr-preview.sh 2024                           # v1 store
VERSION=v2-2B-L~beta1 ./scripts/zarr-preview.sh 2024
```

Key variables: `VERSION` selects the store; `STRETCH_MODE` (`pca` for
v1/v1.1, `bands` for v2) and `BLEND_ZONES` control colour; `RESET_PREVIEW=1`
discards the per-zone resume markers, which you must do whenever the stretch
has changed. Zone markers live under `STATE` on local disk, so nothing
non-Zarr lands in the published bucket.
