#!/usr/bin/env bash
#
# Build a Zarr store from the v2 2B-L~beta1 Tessera dataset on Source
# Cooperative.
#
#     npy/v2-2B-L~beta1  ->  zarr/v2-2B-L~beta1
#
#     ./convertv2.sh                              # the whole dataset
#     ZONE_JOBS=30 WORKERS=8 ./convertv2.sh       # monteverde settings
#     ZONES="30 31 32" ./convertv2.sh             # just those zones
#     MATRYOSHKA_DEPTHS= ./convertv2.sh           # without nested depths
#
# Same shape as convert.sh — installs geotessera from a pinned git ref with
# uv, so the host needs only uv, curl and AWS credentials — with three v2
# differences:
#
#   * The npy directory is `v2-2B-L~beta1` (version *and* variant), while the
#     landmasks are keyed by version alone at `landmasks/v2/`. Neither is
#     derivable from the other, so the preflight asks the library rather than
#     guessing.
#   * v2 dimensions are matryoshka-ordered, so the store also carries the
#     first 4 and first 16 dimensions as their own arrays for ~16% extra
#     storage (docs/specs/zarr-matryoshka-depths.md). A client can then read a
#     4-dimensional prefix without decoding all 128 bands.
#   * The dataset is small: 207,502 tiles over 9 years, against the millions
#     in v1.1. Expect hours rather than days.
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

VERSION="${VERSION:-v2}"
VARIANT="${VARIANT:-2B-L~beta1}"
YEARS="${YEARS:-2017-2025}"
REF="${REF:-switch-source-coop}"          # git ref of avsm/geotessera to install
PROFILE="${PROFILE:-sc-writer}"           # AWS profile with write access

# Nested embedding depths. v2 is matryoshka-trained, so a prefix of the
# dimensions is a usable embedding in its own right; empty disables them.
# zarr-init refuses these for v1/v1.1, whose dimensions are not ordered.
MATRYOSHKA_DEPTHS="${MATRYOSHKA_DEPTHS-4,16}"

# The destination keeps the full dataset name, so v2 variants do not collide
# in zarr/ the way they would if it were keyed by version alone.
DATASET_TAG="${DATASET_TAG:-$VERSION-$VARIANT}"

SRC="${SRC:-s3://tessera/tessera}"                                          # repo root, via gateway
DST="${DST:-s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/$DATASET_TAG}"

# Local paths drop the tilde: it is legal in a filename but invites trouble in
# any command line that later forgets to quote it.
SAFE_TAG="${DATASET_TAG//\~/-}"
VENV="${VENV:-$HOME/.venvs/geotessera-$SAFE_TAG}"
WORKDIR="${WORKDIR:-$HOME/tessera-$SAFE_TAG.build}"
LOGDIR="${LOGDIR:-$WORKDIR/logs}"
SPILL="${SPILL:-$WORKDIR/spill}"

# Each fill worker holds a (128, 4096, 4096) int8 shard buffer plus scales —
# ~2.2 GiB, and peak is several times that. Nested depths add ~320 MiB (they
# are slices of that same buffer, so no extra source reads, just the extra
# writes). --spill-dir memory-maps the buffers, which moves the cost to disk
# and off the OOM killer's radar. Size ZONE_JOBS x WORKERS against RAM, not
# cores: this is bounded by memory, not CPU.
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
DATASET_DIR=$("$VENV/bin/python" - "$VERSION" "$VARIANT" "$MATRYOSHKA_DEPTHS" <<'PY' || fail "preflight failed (see above)"
import sys
from geotessera.registry import dataset_path, _parse_dataset_version
version, variant, depths = sys.argv[1], sys.argv[2], sys.argv[3]
_, norm = _parse_dataset_version(version)
d = dataset_path(norm, variant)
if not d:
    print(f"   {version}/{variant} is not a published dataset", file=sys.stderr)
    sys.exit(1)
from geotessera.zarr import MAX_VALID_SCALE, N_BANDS, _require_group  # noqa: F401
if MAX_VALID_SCALE != 1.0:
    print(f"   MAX_VALID_SCALE is {MAX_VALID_SCALE}, expected 1.0 — this build "
          f"still admits the scales that poison the stretch statistics",
          file=sys.stderr)
    sys.exit(1)
if N_BANDS != 128:
    print(f"   N_BANDS is {N_BANDS}; the v2 tiles are (H, W, 128) int8",
          file=sys.stderr)
    sys.exit(1)
if depths:
    # Nested depths landed after the v1.1 conversion, so a ref that predates
    # them would silently build a single-depth store.
    try:
        from geotessera.zarr import validate_matryoshka_depths
    except ImportError:
        print(f"   this build has no nested-depth support, so "
              f"--matryoshka-depths {depths} would be rejected by argparse. "
              f"Push the branch carrying it and set REF, or re-run with "
              f"MATRYOSHKA_DEPTHS= to build a single-depth store.",
              file=sys.stderr)
        sys.exit(1)
    validate_matryoshka_depths([int(p) for p in depths.split(",")], norm)
print(d)
PY
)
echo "    dataset dir: npy/$DATASET_DIR   commit: $("$VENV/bin/python" -c 'import geotessera;print(getattr(geotessera,"__version__","?"))')"

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
# Fetched with curl rather than left to fsspec: fsspec times out on these
# through the gateway (FSTimeoutError after ~6 min) where curl takes seconds.
#
# Note the asymmetry — the manifest lives under the dataset directory
# (version + variant), the landmasks under the version alone.

MANIFEST="$WORKDIR/manifest-$SAFE_TAG.parquet"
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
echo "    depths ${MATRYOSHKA_DEPTHS:-none (full 128 only)}"
echo "    fill   $ZONE_JOBS zones x $WORKERS workers, spilling to $SPILL"
echo

# ---------------------------------------------------------------------------
# 1. Store skeleton — single writer
# ---------------------------------------------------------------------------

# An explicit if, not `[ -n .. ] && arr+=(..)`: under `set -e` that idiom
# exits the script whenever the test is false and it happens to be the last
# command in its context. Expanded below with the ${arr[@]+..} guard, because
# bash before 4.4 treats an empty array as unset under `set -u`.
INIT_FLAGS=()
if [ -n "$MATRYOSHKA_DEPTHS" ]; then
    INIT_FLAGS+=(--matryoshka-depths "$MATRYOSHKA_DEPTHS")
fi

# Cheap existence probe rather than zarr-scan, which lists every shard.
if "$VENV/bin/python" - "$DST" "$PROFILE" <<'PY' >/dev/null 2>&1; then
import sys
from geotessera.zarr import StoreLocation
from geotessera.remote import build_storage_options
so = build_storage_options(profile=sys.argv[2], region="us-west-2")
sys.exit(0 if StoreLocation.resolve(sys.argv[1], so).exists("zarr.json", on_denied=False) else 1)
PY
    echo "==> [1/3] Store already initialised, skipping zarr-init"
    # Depths are fixed at init: every fill writes what the root declares, and
    # adding one later would mean rewriting every shard. Say what the store
    # actually has rather than what this run asked for.
    "$VENV/bin/python" - "$DST" "$PROFILE" <<'PY' || true
import sys
from geotessera.zarr import StoreLocation, store_depths
from geotessera.remote import build_storage_options
so = build_storage_options(profile=sys.argv[2], region="us-west-2")
d = store_depths(StoreLocation.resolve(sys.argv[1], so))
print(f"    store declares nested depths: {', '.join(map(str, d)) if d else 'none'}")
PY
else
    echo "==> [1/3] Initialising store"
    "$GT" zarr-init "$SRC" --years "$YEARS" --output "$DST" \
        ${INIT_FLAGS[@]+"${INIT_FLAGS[@]}"} \
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
    # `|| rc=$?` rather than a bare `wait`: under `set -e` a wait that
    # reports a failed child aborts this script on the spot, orphaning every
    # zone still in flight instead of reporting the failure at the end.
    if [ "$HAVE_WAIT_N" = "1" ]; then wait -n -p fpid || rc=$?; fi
    if [ -z "$fpid" ]; then
        pair="${inflight%% *}"
        if [ "$pair" = "$inflight" ]; then inflight=""; else inflight="${inflight#* }"; fi
        fpid="${pair%%:*}"; zone="${pair##*:}"; wait "$fpid" || rc=$?
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
echo "    check: $GT zarr-scan $DST --dataset-version $VERSION \\"
echo "             --dataset-variant $VARIANT ${REG[*]} ${STORE_FLAGS[*]}"
echo "    next:  zarr-stretch then zarr-global-preview, if you want a preview pyramid"
