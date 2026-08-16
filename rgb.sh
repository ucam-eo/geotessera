#!/usr/bin/env bash
#
# Global RGB preview pyramid for one year of the Tessera v1 zarr store.
#
# Standalone: runs against an existing venv and never installs anything, so
# it cannot silently swap the code underneath itself. It verifies up front
# that the venv carries the fixes this pipeline depends on and refuses to run
# otherwise — running the unfixed code produces a mosaic coloured from a
# poisoned covariance, which looks plausible and is wrong.
#
# Reads embeddings anonymously through the source.coop gateway and writes the
# pyramid straight to the backing bucket (the gateway is read-only). Build
# bookkeeping stays on local disk so nothing non-Zarr lands in the published
# repository.
#
# Re-running is cheap: each zone drops a marker in $STATE when it finishes and
# a later run skips it. Set RESET_PREVIEW=1 to discard those markers, which
# you must do whenever the stretch has changed — otherwise zones rendered with
# the old colours are skipped and the mosaic ends up half one stretch, half
# another.
#
# Usage:  ./rgb.sh [YEAR]
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YEAR="${1:-2024}"
PROFILE="${PROFILE:-sc-writer}"           # AWS profile with write access
VENV="${VENV:-$HOME/.venvs/geotessera}"

SRC="s3://tessera/tessera/zarr/v1"                                # via gateway
DST="s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1" # backing bucket

STATE="${STATE:-$HOME/tessera-preview-$YEAR.state}"   # local build state
LOGDIR="${LOGDIR:-$HOME/tessera-preview-$YEAR.logs}"

# I/O bound on gateway round-trips, not CPU: workers sit ~99.8% idle pulling
# ~2.4 MB/s each, so throughput scales with in-flight requests. 30 zones x 40
# workers = 1200 requests ~= 490 GB RSS (measured 0.41 GB/worker) against
# 2.5 TB free. 30 zones is the hard ceiling — no two neighbours may run at
# once. Back WORKERS off if 429/503 start appearing in the logs.
ZONE_JOBS="${ZONE_JOBS:-30}"
WORKERS="${WORKERS:-40}"
COARSEN_WORKERS="${COARSEN_WORKERS:-32}"  # threads, I/O bound

# Colour appearance. --mode pca decorrelates the embedding bands, which is
# what stops the mosaic looking grey.
STRETCH_MODE="${STRETCH_MODE:-pca}"
GAMMA="${GAMMA:-0.7}"
SATURATION="${SATURATION:-1.0}"

# The summed statistics for the published v1 store are poisoned by a handful
# of out-of-range scales and cannot be repaired without a multi-day rescan.
# --from-sample takes the covariance from the stored reservoir instead, which
# those rare pixels almost never land in. Set to 0 only after a rebuild.
FROM_SAMPLE="${FROM_SAMPLE:-1}"

# Step 1 rescans every shard in the store — days of work, and already done for
# 2024. It is off by default; set SKIP_STATS=0 for a year that has never had
# its statistics built.
SKIP_STATS="${SKIP_STATS:-1}"
STATS_ZONES="${STATS_ZONES:-$(seq 1 60)}"

RETRIES="${RETRIES:-3}"                   # whole-zone retries, outermost safety net

# The gateway returns 429/525 under load, and at high worker counts that is
# routine rather than exceptional. Adaptive mode adds client-side rate
# limiting that backs off when throttled, which is exactly the 429 case.
# geotessera retries individual chunks on top of this and refuses to mark a
# zone complete if any chunk is still failing.
# `wait -n` (bash 4.3+) lets the scheduler reap whichever zone finished
# rather than only the oldest; without it we fall back to FIFO order.
if (sleep 0 & wait -n -p _probe_pid) >/dev/null 2>&1; then HAVE_WAIT_N=1; else HAVE_WAIT_N=0; fi

export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"

# Read the source anonymously through the CDN; path-style because
# data.source.coop has no wildcard DNS.
SRC_FLAGS=(--store-endpoint-url https://data.source.coop --store-anon --store-path-style)
# Write to the backing bucket with credentials. No endpoint-url: writes must
# not go through the read-only gateway.
DST_FLAGS=(--output-profile "$PROFILE" --output-region us-west-2
           --output-acl bucket-owner-full-control)
# Same bucket addressed as the *store*, for the steps that modify the
# embeddings store itself rather than the pyramid.
STORE_W_FLAGS=(--store-profile "$PROFILE" --store-region us-west-2
               --store-acl bucket-owner-full-control)

GT="$VENV/bin/geotessera-registry"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

fail() { echo; echo "!! $*" >&2; exit 1; }

[ -x "$GT" ] || fail "no geotessera-registry at $GT (set VENV=...)"

# Each of these guards a bug that silently corrupts the output rather than
# failing, so check for them explicitly instead of trusting the install.
"$VENV/bin/python" - <<'PY' || fail "venv is missing required fixes (see above)"
import sys
try:
    from geotessera.zarr import MAX_VALID_SCALE, _require_group  # noqa: F401
except ImportError as e:
    print(f"   missing symbol: {e}", file=sys.stderr); sys.exit(1)
if MAX_VALID_SCALE != 1.0:
    print(f"   MAX_VALID_SCALE is {MAX_VALID_SCALE}, expected 1.0 — this build "
          f"still admits the out-of-range scales that poison the covariance",
          file=sys.stderr)
    sys.exit(1)
PY

for flag in --from-sample --allow-drift; do
    "$GT" zarr-stretch --help 2>&1 | grep -q -- "$flag" \
        || fail "zarr-stretch has no $flag — venv predates the stretch fixes"
done
for flag in --reproject-only --coarsen-only --state-url; do
    "$GT" zarr-global-preview --help 2>&1 | grep -q -- "$flag" \
        || fail "zarr-global-preview has no $flag — venv predates the parallel fixes"
done

mkdir -p "$STATE" "$LOGDIR"

if [ "${RESET_PREVIEW:-0}" = "1" ]; then
    echo "==> Discarding preview markers in $STATE"
    rm -rf "${STATE:?}/_preview"
elif [ -d "$STATE/_preview" ] && [ -n "$(ls -A "$STATE/_preview" 2>/dev/null)" ]; then
    echo "==> $(ls -A "$STATE/_preview" | wc -l) zone(s) already marked complete;"
    echo "    they will be skipped. Re-run with RESET_PREVIEW=1 if the stretch"
    echo "    has changed since they were rendered."
fi

echo "==> year=$YEAR  zones=1-60  parallel=$ZONE_JOBS x $WORKERS workers"
echo "    venv   $VENV"
echo "    source $SRC"
echo "    dest   $DST"
echo "    state  $STATE"
echo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Run one zone with retries; log to its own file. Returns non-zero on failure.
run_zone() {
    local label="$1"; shift
    local log="$LOGDIR/$label.log"
    local attempt
    for attempt in $(seq 1 "$RETRIES"); do
        if "$@" >>"$log" 2>&1; then
            return 0
        fi
        echo "    [$label] attempt $attempt failed, see $log" >&2
        sleep $((attempt * 10))
    done
    return 1
}

# Run a list of zones with at most $ZONE_JOBS in flight, waiting on each as the
# throttle advances rather than collecting every status at the end.
#
# That ordering is the point. Waiting only at the end leaves up to 60 finished
# children un-reaped for days, and bash then loses some of their exit statuses
# ("wait: pid N is not a child of this shell"), which reads as a failure for a
# zone that in fact succeeded. Reaping as we go keeps at most $ZONE_JOBS
# statuses outstanding. Also avoids `wait -n` and arrays (absent in bash 3.2)
# and polling `jobs` (which perturbs the job table).
run_zones() {
    local label_prefix="$1"; shift
    local zones="$1"; shift
    local failed=0 inflight="" n=0 z pair pid zone

    _reap_one() {
        pair="${inflight%% *}"
        if [ "$pair" = "$inflight" ]; then inflight=""; else inflight="${inflight#* }"; fi
        pid="${pair%%:*}"; zone="${pair##*:}"
        if wait "$pid"; then
            echo "    zone $zone done"
        else
            echo "  !! zone $zone FAILED (see $LOGDIR/$label_prefix-z$zone.log)" >&2
            failed=1
        fi
        n=$((n - 1))
    }

    for z in $zones; do
        run_zone "$label_prefix-z$z" "$@" --zones "$z" &
        if [ -z "$inflight" ]; then inflight="$!:$z"; else inflight="$inflight $!:$z"; fi
        n=$((n + 1))
        while [ "$n" -ge "$ZONE_JOBS" ]; do _reap_one; done
    done
    while [ -n "$inflight" ]; do _reap_one; done
    return $failed
}

# Zones composite into shared level-0 chunks with a read-modify-write, so two
# zones touching the same chunk concurrently would lose each other's pixels.
# Only *adjacent* zones share chunks (measured against the real footprints:
# every conflicting pair differs by one, plus utm01/utm60 across the
# antimeridian). So a zone is safe to start whenever neither neighbour is
# running.
#
# The old odd-then-even rounds satisfied that but wasted most of the machine:
# zones differ hugely in size (z19 is 254k chunks, the smallest ~20k), so a
# round drained from 30 zones to a handful and concurrency collapsed with it
# — measured load fell 197 -> 27 -> 4 across one round. This scheduler keeps
# starting whatever is currently safe, holding ~30 zones busy throughout
# instead of trailing off twice.
#
# 30 concurrent is the ceiling regardless: no two neighbours may run at once.
neighbours_busy() {
    local z="$1" r
    for r in $RUNNING; do
        [ "$r" = "$(( z - 1 ))" ] && return 0
        [ "$r" = "$(( z + 1 ))" ] && return 0
        # utm01 and utm60 meet at the antimeridian.
        { [ "$z" = 1 ] && [ "$r" = 60 ]; } && return 0
        { [ "$z" = 60 ] && [ "$r" = 1 ]; } && return 0
    done
    return 1
}

# Run every zone in $1 with at most $ZONE_JOBS in flight, never starting a
# zone whose neighbour is running. Any failure is fatal once the queue drains.
run_zones_scheduled() {
    local label_prefix="$1"; shift
    local queue="$1"; shift
    local failed=0 RUNNING="" inflight="" n=0 z pair pid zone started
    # Normalise separators: callers pass $(seq ...), which is newline-separated.
    queue=$(echo $queue)

    # Reap whichever zone finished, not merely the oldest. Blocking on the
    # oldest leaves finished-but-unreaped zones still counted in RUNNING, so
    # their neighbours stay blocked and slots go unused: measured 13 of 30
    # slots busy with 9 zones startable, and throughput down 1813 -> 648 MB/s.
    _reap_one() {
        local fpid="" rc=0 keep="" newin="" r p
        if [ "$HAVE_WAIT_N" = "1" ]; then
            wait -n -p fpid; rc=$?
        fi
        if [ -z "$fpid" ]; then
            pair="${inflight%% *}"
            if [ "$pair" = "$inflight" ]; then inflight=""; else inflight="${inflight#* }"; fi
            fpid="${pair%%:*}"; zone="${pair##*:}"
            wait "$fpid"; rc=$?
        else
            zone=""
            for p in $inflight; do
                if [ "${p%%:*}" = "$fpid" ] && [ -z "$zone" ]; then
                    zone="${p##*:}"
                else
                    newin="$newin $p"
                fi
            done
            inflight=$(echo $newin)
            [ -z "$zone" ] && return 0
        fi
        if [ "$rc" = "0" ]; then
            echo "    zone $zone done"
        else
            echo "  !! zone $zone FAILED (see $LOGDIR/$label_prefix-z$zone.log)" >&2
            failed=1
        fi
        for r in $RUNNING; do [ "$r" = "$zone" ] || keep="$keep $r"; done
        RUNNING="$keep"
        n=$((n - 1))
    }

    while [ -n "$queue" ] || [ -n "$inflight" ]; do
        started=0
        if [ "$n" -lt "$ZONE_JOBS" ]; then
            for z in $queue; do
                neighbours_busy "$z" && continue
                run_zone "$label_prefix-z$z" "$@" --zones "$z" &
                if [ -z "$inflight" ]; then inflight="$!:$z"; else inflight="$inflight $!:$z"; fi
                RUNNING="$RUNNING $z"
                local keepq="" q
                for q in $queue; do [ "$q" = "$z" ] || keepq="$keepq $q"; done
                queue="$keepq"
                n=$((n + 1)); started=1
                [ "$n" -ge "$ZONE_JOBS" ] && break
            done
        fi
        # Nothing could start (all remaining blocked, or at capacity): wait
        # for the oldest in-flight zone to free its neighbours.
        if [ "$started" = "0" ] && [ -n "$inflight" ]; then _reap_one; fi
    done
    while [ -n "$inflight" ]; do _reap_one; done
    return $failed
}

ALL_ZONES=$(seq 1 60)

# ---------------------------------------------------------------------------
# 1. Per-zone stretch statistics. Rescans every shard, so this is the
#    expensive step by a wide margin, and it has no resume: a zone writes its
#    arrays only after scanning all ~2000 of its shards, so an interrupted
#    zone starts over and a finished one is rescanned in full on a re-run.
# ---------------------------------------------------------------------------

if [ "$SKIP_STATS" = "1" ]; then
    echo "==> [1/5] Skipping stretch-statistics rebuild (SKIP_STATS=1)"
else
    echo "==> [1/5] Rebuilding stretch statistics"
    run_zones stats "$STATS_ZONES" \
        "$GT" zarr-fill "$DST" --backfill-stretch-stats --year "$YEAR" \
        "${STORE_W_FLAGS[@]}" \
        || fail "stretch-statistics rebuild failed; fix before continuing"
    echo "==> [2/5] Consolidating store metadata"
    "$GT" zarr-consolidate "$DST" "${STORE_W_FLAGS[@]}" 2>&1 \
        | tee "$LOGDIR/consolidate.log"
fi

# ---------------------------------------------------------------------------
# 2. One cross-zone stretch for the whole globe, persisted to the store root.
#    Sharing a single stretch is what removes colour seams between zones, so
#    it must be settled before any reprojection. zarr-stretch refuses to save
#    a stretch that fails its drift check, so a bad one stops here.
# ---------------------------------------------------------------------------

STRETCH_FLAGS=()
[ "$FROM_SAMPLE" = "1" ] && STRETCH_FLAGS+=(--from-sample)

echo "==> [3/5] Computing global stretch (mode=$STRETCH_MODE${FROM_SAMPLE:+, from-sample=$FROM_SAMPLE})"
"$GT" zarr-stretch "$DST" --year "$YEAR" --mode "$STRETCH_MODE" \
    "${STRETCH_FLAGS[@]}" "${STORE_W_FLAGS[@]}" 2>&1 | tee "$LOGDIR/stretch.log"

# ---------------------------------------------------------------------------
# 3. Reproject every zone into level 0, in two rounds.
#
#    Zones composite into shared level-0 chunks with a read-modify-write, so
#    two zones touching the same chunk concurrently would lose each other's
#    pixels. Only adjacent zones share chunks, so odd-numbered zones are
#    mutually disjoint, as are even-numbered ones. --reproject-only stops each
#    zone's coarsening at the depth where that disjointness still holds.
# ---------------------------------------------------------------------------

echo "==> [4/5] Reprojecting all 60 zones (neighbour-aware, up to $ZONE_JOBS at once)"
run_zones_scheduled zone "$ALL_ZONES" \
    "$GT" zarr-global-preview "$SRC" --year "$YEAR" --reproject-only \
    --workers "$WORKERS" --gamma "$GAMMA" --saturation "$SATURATION" \
    --output "$DST" --state-url "$STATE" "${SRC_FLAGS[@]}" "${DST_FLAGS[@]}" \
    || fail "one or more zones failed; re-run to resume before coarsening"

# ---------------------------------------------------------------------------
# 4. Build the coarse levels where every zone's data overlaps. Single-writer,
#    one pass over the whole grid, and it consolidates the pyramid at the end.
# ---------------------------------------------------------------------------

echo "==> [5/5] Coarsening the shared upper levels"
"$GT" zarr-global-preview "$SRC" --year "$YEAR" --coarsen-only \
    --workers "$COARSEN_WORKERS" \
    --output "$DST" --state-url "$STATE" "${SRC_FLAGS[@]}" "${DST_FLAGS[@]}" \
    2>&1 | tee "$LOGDIR/coarsen.log"

echo
echo "==> Done. Pyramid for $YEAR is at $DST/global_rgb"
echo "    logs:  $LOGDIR"
echo "    state: $STATE"
