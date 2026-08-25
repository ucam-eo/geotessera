#!/usr/bin/env bash
#
# Incrementally sync one npy dataset into its Zarr store and RGB preview.
#
#     npy/<dataset>  ->  npy/<dataset>/manifest.parquet
#                    ->  zarr/<dataset>
#                    ->  zarr/<dataset>/global_rgb   (the preview year)
#
#     ./scripts/zarr-sync.sh                        # v2 2B-L~beta2
#     DRY_RUN=1 ./scripts/zarr-sync.sh              # print the plan, change nothing
#     VARIANT=2B-L~beta1 ./scripts/zarr-sync.sh     # another dataset
#     ZONE_JOBS=30 WORKERS=8 ./scripts/zarr-sync.sh # monteverde fill settings
#
# Thin wrapper: all the logic lives in `geotessera-registry s3sync`, which
# scans the bucket, diffs against the manifest as of the last successful
# sync, publishes the fresh manifest, fills only the changed (zone, year)
# pairs, and re-renders just the changed zones of the preview year's
# pyramid. A run that finds nothing new stops after the scan; an
# interrupted run is simply re-run. New years are NOT appended — run
# `geotessera-registry zarr-extend` by hand once a year, and the deferred
# tiles sync on the next run.
#
# This wrapper only pins the install (uv + a git ref, so the host needs
# nothing else) and the monteverde-shaped defaults. Sync state lives in
# $WORKDIR: point every run at the same one.
#
set -euo pipefail

VERSION="${VERSION:-v2}"
VARIANT="${VARIANT:-2B-L~beta2}"
REF="${REF:-main}"                        # git ref of ucam-eo/geotessera to install
PROFILE="${PROFILE:-sc-writer}"           # AWS profile with write access
PREVIEW_YEAR="${PREVIEW_YEAR:-2024}"

# Fill pool: bounded by memory, not cores (see zarr-convert-v2.sh).
ZONE_JOBS="${ZONE_JOBS:-8}"
WORKERS="${WORKERS:-4}"

SAFE_TAG="$VERSION-${VARIANT//\~/-}"
VENV="${VENV:-$HOME/.venvs/geotessera-$SAFE_TAG}"
WORKDIR="${WORKDIR:-$HOME/tessera-$SAFE_TAG.sync}"

echo "==> Installing geotessera@$REF into $VENV"
uv venv --python 3.13 "$VENV" >/dev/null
uv pip install --python "$VENV" --quiet \
    "geotessera[s3] @ git+https://github.com/ucam-eo/geotessera@$REF"
GT="$VENV/bin/geotessera-registry"
[ -x "$GT" ] || { echo "!! install produced no geotessera-registry at $GT" >&2; exit 1; }
"$GT" s3sync --help | grep -q -- --force-rewrite \
    || { echo "!! this build has no s3sync — set REF to a ref carrying it" >&2; exit 1; }

EXTRA=()
if [ "${DRY_RUN:-0}" = "1" ]; then
    EXTRA+=(--dry-run)
fi

# Read the source anonymously through the CDN (path-style: data.source.coop
# has no wildcard DNS); write to the backing bucket with credentials.
exec "$GT" s3sync s3://tessera/tessera \
    --dataset-version "$VERSION" --dataset-variant "$VARIANT" \
    --workdir "$WORKDIR" --preview-year "$PREVIEW_YEAR" \
    --zone-jobs "$ZONE_JOBS" --workers "$WORKERS" \
    --source-endpoint-url https://data.source.coop --source-anon --source-path-style \
    --store-profile "$PROFILE" --store-region us-west-2 \
    --store-acl bucket-owner-full-control \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"
