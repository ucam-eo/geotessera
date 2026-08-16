#!/usr/bin/env bash
#
# Build a Zarr store from a Tessera npy dataset on Source Cooperative.
#
# Defaults to v1.1/cambridge -> zarr/v1.1, but any published dataset works:
#
#     ./convert.sh                       # v1.1 cambridge, years 2015-2025
#     VERSION=v1 VARIANT=vultr YEARS=2017-2025 ./convert.sh
#
# Installs geotessera from git with uv, so the host needs only uv, curl and
# AWS credentials. Unlike the preview build this is safe to install on every
# run: the code comes from a pinned ref rather than files patched in by hand.
#
# Zone fills are independent — each writes only its own zone group — so they
# run in a simple pool with no ordering constraint. A fill also resumes from
# the store itself: shard objects are the record of what is done, so re-running
# after any interruption retries exactly the unfinished work.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION="${VERSION:-v1.1}"
VARIANT="${VARIANT:-cambridge}"
YEARS="${YEARS:-2015-2025}"
REF="${REF:-switch-source-coop}"          # git ref of avsm/geotessera to install
PROFILE="${PROFILE:-sc-writer}"           # AWS profile with write access

# Dataset directory in the npy/ tree. v1.1/cambridge lives in npy/v1.1-cam/,
# so this cannot be derived from the version alone — the preflight asks the
# installed library rather than guessing.
SRC="${SRC:-s3://tessera/tessera}"                                          # repo root, via gateway
DST="${DST:-s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/$VERSION}"

VENV="${VENV:-$HOME/.venvs/geotessera-$VERSION}"
WORKDIR="${WORKDIR:-$HOME/tessera-$VERSION.build}"
LOGDIR="${LOGDIR:-$WORKDIR/logs}"
SPILL="${SPILL:-$WORKDIR/spill}"

# Each fill worker holds a (128, 4096, 4096) int8 shard buffer plus scales —
# ~2.2 GiB, and peak is several times that. --spill-dir memory-maps them, which
# moves the cost to disk and off the OOM killer's radar. Size ZONE_JOBS x
# WORKERS against RAM, not cores: this is bounded by memory, not CPU.
ZONE_JOBS="${ZONE_JOBS:-8}"               # zones filled concurrently
WORKERS="${WORKERS:-4}"                   # shard workers per zone

RETRIES="${RETRIES:-3}"                   # whole-zone retries

# The gateway returns 429/525 under load; adaptive mode adds client-side rate
# limiting that backs off when throttled.
export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"

# Read the source anonymously through the CDN; path-style because
# data.source.coop has no wildcard DNS. Writes go straight to the backing
# bucket — the gateway is read-only, so there is no --store-endpoint-url.
SRC_FLAGS=(--source-endpoint-url https://data.source.coop --source-anon --source-path-style)
STORE_FLAGS=(--store-profile "$PROFILE" --store-region us-west-2
             --store-acl bucket-owner-full-control)
DATASET=(--dataset-version "$VERSION" --dataset-variant "$VARIANT")

fail() { echo; echo "!! $*" >&2; exit 1; }

mkdir -p "$WORKDIR" "$LOGDIR" "$SPILL"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

echo "==> Installing geotessera@$REF into $VENV"
uv venv --python 3.13 "$VENV" >/dev/null
uv pip install --python "$VENV" --quiet \
    "geotessera[s3] @ git+https://github.com/avsm/geotessera@$REF"
GT="$VENV/bin/geotessera-registry"
[ -x "$GT" ] || fail "install produced no geotessera-registry at $GT"

# Resolve the dataset directory and confirm the build carries the fixes this
# pipeline depends on. Each of these fails silently rather than loudly, so
# check rather than trust: an older ref would write a plausible-looking store.
DATASET_DIR=$("$VENV/bin/python" - "$VERSION" "$VARIANT" <<'PY' || fail "preflight failed (see above)"
import sys
from geotessera.registry import dataset_path, _parse_dataset_version
version, variant = sys.argv[1], sys.argv[2]
_, norm = _parse_dataset_version(version)
d = dataset_path(norm, variant)
if not d:
    print(f"   {version}/{variant} is not a published dataset", file=sys.stderr)
    sys.exit(1)
from geotessera.zarr import MAX_VALID_SCALE, _require_group  # noqa: F401
if MAX_VALID_SCALE != 1.0:
    print(f"   MAX_VALID_SCALE is {MAX_VALID_SCALE}, expected 1.0 — this build "
          f"still admits the scales that poison the stretch statistics",
          file=sys.stderr)
    sys.exit(1)
print(d)
PY
)
echo "    dataset dir: npy/$DATASET_DIR   commit: $("$VENV/bin/python" -c 'import geotessera;print(getattr(geotessera,"__version__","?"))')"

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
# Fetched with curl rather than left to fsspec: fsspec times out on these
# through the gateway (FSTimeoutError after ~6 min) where curl takes seconds.

MANIFEST="$WORKDIR/manifest-$VERSION.parquet"
LANDMASKS="$WORKDIR/landmasks-$VERSION.parquet"
BASE="https://data.source.coop/tessera/tessera"

fetch() {  # url dest
    [ -s "$2" ] && { echo "    cached $(basename "$2")"; return 0; }
    echo "    fetching $(basename "$2")"
    curl -fsS --retry 5 --retry-delay 5 --max-time 1800 -o "$2.part" "$1" \
        || fail "could not fetch $1"
    mv "$2.part" "$2"
}

echo "==> Registries"
fetch "$BASE/npy/$DATASET_DIR/manifest.parquet" "$MANIFEST"
fetch "$BASE/landmasks/$VERSION/landmasks.parquet" "$LANDMASKS"
REG=(--manifest-url "$MANIFEST" --landmasks-url "$LANDMASKS")

# Zones that actually carry tiles. Filling a zone with no tiles is harmless but
# pointless, and the list makes the run's scope explicit in the log.
ZONES="${ZONES:-$("$VENV/bin/python" - "$MANIFEST" <<'PY'
import math, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["lon"])
z = {max(1, min(60, int(math.floor((lon + 180) / 6)) + 1)) for lon in df["lon"]}
print(" ".join(str(v) for v in sorted(z)))
PY
)}"

echo "==> $VERSION/$VARIANT  years=$YEARS  zones=$(echo "$ZONES" | wc -w)"
echo "    source $SRC (npy/$DATASET_DIR)"
echo "    dest   $DST"
echo "    fill   $ZONE_JOBS zones x $WORKERS workers, spilling to $SPILL"
echo

# ---------------------------------------------------------------------------
# 1. Store skeleton — single writer
# ---------------------------------------------------------------------------

# Cheap existence probe rather than zarr-scan, which lists every shard.
if "$VENV/bin/python" - "$DST" "$PROFILE" <<'PY' >/dev/null 2>&1; then
import sys
from geotessera.zarr import StoreLocation
from geotessera.remote import build_storage_options
so = build_storage_options(profile=sys.argv[2], region="us-west-2")
sys.exit(0 if StoreLocation.resolve(sys.argv[1], so).exists("zarr.json", on_denied=False) else 1)
PY
    echo "==> [1/3] Store already initialised, skipping zarr-init"
else
    echo "==> [1/3] Initialising store"
    "$GT" zarr-init "$SRC" --years "$YEARS" --output "$DST" \
        "${DATASET[@]}" "${REG[@]}" "${SRC_FLAGS[@]}" "${STORE_FLAGS[@]}" \
        2>&1 | tee "$LOGDIR/init.log"
fi

# ---------------------------------------------------------------------------
# 2. Fill, one zone per process
# ---------------------------------------------------------------------------

run_zone() {
    local z="$1" log="$LOGDIR/zone-z$1.log" attempt
    for attempt in $(seq 1 "$RETRIES"); do
        if "$GT" zarr-fill "$SRC" "$DST" --zones "$z" --workers "$WORKERS" \
                --spill-dir "$SPILL/z$z" --no-consolidate \
                "${DATASET[@]}" "${REG[@]}" "${SRC_FLAGS[@]}" "${STORE_FLAGS[@]}" \
                >>"$log" 2>&1; then
            return 0
        fi
        echo "    [z$z] attempt $attempt failed, see $log" >&2
        sleep $((attempt * 30))
    done
    return 1
}

# Reap whichever zone finished rather than the oldest: waiting on the oldest
# leaves finished children unreaped and slots idle behind one slow zone.
if (sleep 0 & wait -n -p _probe) >/dev/null 2>&1; then HAVE_WAIT_N=1; else HAVE_WAIT_N=0; fi

echo "==> [2/3] Filling zones"
failed=0 inflight="" n=0
reap() {
    local fpid="" rc=0 zone="" newin="" p pair
    if [ "$HAVE_WAIT_N" = "1" ]; then wait -n -p fpid; rc=$?; fi
    if [ -z "$fpid" ]; then
        pair="${inflight%% *}"
        if [ "$pair" = "$inflight" ]; then inflight=""; else inflight="${inflight#* }"; fi
        fpid="${pair%%:*}"; zone="${pair##*:}"; wait "$fpid"; rc=$?
    else
        for p in $inflight; do
            if [ "${p%%:*}" = "$fpid" ] && [ -z "$zone" ]; then zone="${p##*:}"; else newin="$newin $p"; fi
        done
        inflight=$(echo $newin)
        [ -z "$zone" ] && return 0
    fi
    if [ "$rc" = "0" ]; then echo "    zone $zone done"; else
        echo "  !! zone $zone FAILED (see $LOGDIR/zone-z$zone.log)" >&2; failed=1; fi
    n=$((n - 1))
}

for z in $ZONES; do
    while [ "$n" -ge "$ZONE_JOBS" ]; do reap; done
    run_zone "$z" &
    if [ -z "$inflight" ]; then inflight="$!:$z"; else inflight="$inflight $!:$z"; fi
    n=$((n + 1))
done
while [ -n "$inflight" ]; do reap; done
[ "$failed" = "0" ] || fail "one or more zones failed; re-run to retry just those (fills resume from the store)"

# ---------------------------------------------------------------------------
# 3. Publish — single writer, after every zone has finished
# ---------------------------------------------------------------------------

echo "==> [3/3] Consolidating metadata"
"$GT" zarr-consolidate "$DST" "${STORE_FLAGS[@]}" 2>&1 | tee "$LOGDIR/consolidate.log"

echo
echo "==> Done. Store at $DST"
echo "    logs:  $LOGDIR"
echo "    next:  zarr-stretch then zarr-global-preview, if you want a preview pyramid"
