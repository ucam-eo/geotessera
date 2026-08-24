# Nested embedding depths (matryoshka prefixes) in the Zarr store

**Status:** Implemented. The storage layer covers `zarr-init`, `zarr-fill`,
`zarr-extend` and `zarr-scan`; `zarr-global-preview` reads its colour bands
from the shallowest adequate depth array. The stretch-machinery retirement
described under "Consequences for the RGB preview" is **partly done** —
step 2 (percentiles from a coarse pyramid level) is not built, because the
existing fill-time statistics already answer it without reading embeddings.
**Scope:** `geotessera-registry` subcommands `zarr-init`, `zarr-fill`, `zarr-extend`
**Store convention:** additions to the per-zone group layout and to the root `geoemb:` attributes
**Applies to:** v2 embeddings and later. Refused for v1/v1.1.

## Motivation

v2 embeddings are trained matryoshka-style: the dimensions are ordered by
importance, so the first 4 and the first 16 of the 128 are each a usable
embedding in their own right rather than an arbitrary slice. Consumers want
that choice — a 4-dimensional embedding for browsing, wide-area screening and
RGB rendering; 16 for moderate-fidelity analysis; the full 128 for ML work on
a bounded AOI.

The store must serve all three without making the cheap cases pay for the
expensive one.

### Why the existing layout cannot serve prefixes

`embeddings` is chunked `(1, 128, 32, 32)` — the band axis is **one chunk
wide**. Reading dimensions 0–3 therefore decodes all 128 bands and discards
96.9% of them. A prefix read costs exactly as much as a full read.

### Why not chunk along the band axis

The obvious fix is `chunks=(1, 4, …)`, so a prefix read touches one chunk and
nothing is duplicated. Rejected:

- A full-128 read becomes 32 chunk fetches instead of 1. Over a CDN, request
  count dominates wall-clock; this is the access pattern we would be
  penalising to speed up the one that is already cheapest to serve.
- The three depths want different *spatial* chunk granularity (below). One
  array cannot have three.
- Only the 4-dimensional array wants a spatial pyramid. Coarsening a
  band-chunked array drags all 128 dimensions through a reduction whose
  output uses 3 of them.

### Cost of duplication

Duplication is the cheap axis. The uncompressed floor is `(4 + 16) / 128` =
**15.6%** on top of `embeddings`, plus roughly 1% if the depth-4 array carries
a pyramid. Compression moves it only slightly: the smaller arrays have less
cross-band redundancy per chunk to exploit, but also wider spatial chunks to
compensate. Measured at **16.2%** over a 512×512×128 block of spatially
correlated int8 (`embeddings` 2 208 178 bytes, `embeddings_d4` 64 468,
`embeddings_d16` 293 200) — synthetic, so treat it as the right order rather
than a production figure, but it is nowhere near a multiple of the floor.

Crucially **`scales` is not duplicated**. Quantisation is
`per_pixel_scale`: `scales` is `(T, H, W)`, one scale per pixel shared by all
128 bands, so every depth array dequantises against the same array. A reader
that opens `embeddings_d4` uses the identical `scales` it would have used for
`embeddings`.

## Design

### One shard grid, three chunk granularities

All depth arrays share the **same shard grid** as `embeddings`:
`(1, D, 4096, 4096)`, addressed by the same `(t, shard_row, shard_col)`. This
is the property worth protecting, because it means:

- one source read fills every depth from the same in-memory buffer;
- a shard coordinate means the same thing in every array;
- `_existing_shards` — which lists `embeddings/c/{t}/0/{sr}/{sc}` — keeps
  working unchanged as the resume oracle;
- `stretch_stats_shards` and any future coverage mask stay indexed by one grid.

What varies is the **inner chunk**, sized to hold chunk bytes roughly constant
at or below the 128 KiB the depth-128 array already uses:

| Array | `chunks` | chunk bytes | chunks/shard | shard index | shard object |
|---|---|---|---|---|---|
| `embeddings` (D=128) | `(1,128,32,32)` | 128 KiB | 16 384 | 256 KiB | 2 GiB |
| `embeddings_d16` | `(1,16,64,64)` | 64 KiB | 4 096 | 64 KiB | 256 MiB |
| `embeddings_d4` | `(1,4,128,128)` | 64 KiB | 1 024 | 16 KiB | 16 MiB |

Shard *byte* sizes differ by 128×. That is fine and intended; what must not
differ is the grid.

The rule (`depth_inner_chunk`) is the largest power-of-two side `s` with
`D · s² ≤ 128 KiB`, clamped to `[32, 4096]`. Every power of two ≤ 4096 divides
4096, so `shards % chunks == 0` holds on every axis by construction.

#### Why not keep 32×32 everywhere

At D=4 a `32×32` chunk is 4 KiB uncompressed, perhaps 1–2 KiB compressed.
Range requests that small are dominated by HTTP overhead. Worse, the sharding
codec's index is a fixed 16 bytes per inner chunk *including absent ones*, so
`32×32` at D=4 means a **256 KiB index fetch to reach a 1–2 KiB chunk**. The
`128×128` chunk drops the index to 16 KiB — which matters precisely because
depth-4 is the array interactive clients hit hardest.

#### Why not shrink the shard for the smaller arrays

Keeping 32×32 chunks and shrinking the shard to, say, `(1,4,1024,1024)` would
also bound the index. It would also give the depth arrays a different shard
grid from `embeddings`, forfeiting every property listed above. Rejected.

### Write order is load-bearing

For each shard, depths are written **ascending, with the full 128 last**:

```
embeddings_d4  ← emb_buf[:4]
embeddings_d16 ← emb_buf[:16]
embeddings     ← emb_buf          (last)
scales         ← scales_buf
```

All are slices of the one buffer already in memory, so there are no extra
source reads. The ordering makes "the `embeddings` shard object exists" imply
every shallower depth for that coordinate also exists, which is what lets
`_existing_shards` remain the single resume oracle with no changes. A crash
mid-shard leaves shallow depths written and the full depth absent; resume sees
the full depth missing and rewrites all of them, idempotently.

Reversing the order would strand shallow depths permanently: resume would see
`embeddings` present and skip the shard forever.

### Discovery: root attributes only

The zone-group convention is that zone groups carry `proj:` and `spatial:`
only, with all `geoemb:` attributes on the root (see
`zarr-stretch-stats.md`). Depths follow that rule — nothing is added to zone
groups or to the arrays themselves:

```json
"geoemb:dimensions": 128,
"geoemb:depths": [
  {"dimensions": 4,   "array": "embeddings_d4"},
  {"dimensions": 16,  "array": "embeddings_d16"},
  {"dimensions": 128, "array": "embeddings"}
]
```

The list is always complete: the full depth is included as its own entry so a
client picking a depth never has to special-case the base array's name. Absence
of `geoemb:depths` means a single-depth store, which is every v1/v1.1 store.

`geoemb:depths` is metadata *about the layout*, not about coverage, so it
cannot drift the way a per-year coverage attribute would: it is written once
at init and every fill writes all listed depths or none.

### Creation at init, never lazily

Depth arrays are created by `zarr-init` alongside `embeddings`, for the same
reason the stretch arrays are: fills run concurrently across zones, so a
lazy create/decide step would race with `zarr-consolidate` snapshots and leave
mixed stores that every reader must handle. A store either declares
`geoemb:depths` and has all of them, or declares none.

`zarr-extend` grows every depth array's time axis with `embeddings`, and
refuses if a declared depth array is missing — the same guard already applied
to the stretch arrays.

### Gating

`--matryoshka-depths` is refused for dataset versions below 2.0. v1 and v1.1
dimensions are not ordered by importance, so a prefix of them is an arbitrary
slice: the store would look correct and be quietly meaningless. Depths must
also be strictly increasing, ≥ 1, and strictly less than 128 (the full depth is
implied, not listed).

## Consequences for the RGB preview

**Implemented, except where noted.**

`zarr-stretch` currently derives the preview colouring by reducing 128 bands to
3 via PCA, which is why the store carries `stretch_stats_count`,
`stretch_stats_sum`, `stretch_stats_prod` (a 128×128 covariance),
`stretch_sample`, `stretch_sample_scales`, `stretch_sample_count` and
`stretch_stats_shards`, and why fills carry a reservoir sampler.

With ordered dimensions, bands 0–2 *are* the colour channels and the reduction
disappears. `zarr-stretch --mode bands` computes per-channel percentiles from
the fill-time statistics without reading a single embedding, and
`zarr-global-preview` then reprojects from `embeddings_d4` instead of
`embeddings` — a thirty-second of the bytes and a sixteenth of the sharding
index for byte-identical pixels.

The array is chosen from the stretch's own `mode` and `bands`, so the two
cannot disagree: a `pca` stretch projects all 128 dimensions and always reads
the full array, while a `bands` stretch reads the shallowest depth that still
holds the bands it names.

Step 2 of the original sketch — percentiles from a coarse pyramid level — was
not needed. The fill-time statistics already cover it at lower cost, since
they are collected while every pixel is resident rather than read back.

That removes the sampling apparatus and with it a whole bug class: there is no
covariance to poison, so a bad dequantisation scale can no longer contaminate a
global statistic shared by every zone. `MAX_VALID_SCALE` still matters for
dequantisation itself.

For v2 stores the stretch arrays are therefore expected to become vestigial.
They are still created, so that a v2 store remains readable by the existing
`zarr-stretch` path until the replacement lands.

## Verification

`zarr-scan` lists each declared depth's shard objects and flags any coordinate
present in `embeddings` but absent from a prefix. Given the write order that
cannot happen, so a hit means shards were written by a build with the wrong
order; the repair is `zarr-fill --zones N --rewrite-existing-shards`, which
rewrites every depth idempotently.

Depth arrays also add a dimension name per depth (`band_d4`, `band_d16`) with a
matching coordinate array. Reusing the plain `band` name was rejected: an
xarray reader opening the zone group would see one `band` dimension with
conflicting sizes (4 vs 128) and refuse to build the dataset.
