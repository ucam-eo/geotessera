"""Checks for seam-aware point reads in ``geotessera.store``.

Run by tests/store.t.  Every check works on a small in-memory zone Dataset
carrying the three pixel states of a real one — a scale for land, ``NaN``
for water, ``+inf`` for never written — so the suite stays offline and fast.

Prints one ``ok - <name>`` line per check and exits non-zero if any failed.
"""

import logging
import sys

import numpy as np
import xarray as xr

# Expected warnings (NaN-padded patches) must not reach cram's output.
logging.getLogger("geotessera.store").setLevel(logging.ERROR)

from geotessera.store import (
    NODATA,
    OUTSIDE,
    VALID,
    WATER,
    GeoTesseraZarr,
    _patch_crs,
    _resolve_zone,
    _seam_neighbours,
    _zone_for_lon,
    _zones_spanned,
)

FAILED = []


def check(name, condition):
    if condition:
        print(f"ok - {name}")
    else:
        print(f"FAIL - {name}")
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Zone seams
# ---------------------------------------------------------------------------

check("zone interior needs no neighbours", _seam_neighbours(3.0) == [])
check("just east of a seam looks west", _seam_neighbours(138.03) == [53])
check("just west of a seam looks east", _seam_neighbours(137.97) == [54])
# The band is half a tile wide each side, since a tile belongs to the zone
# holding its centre.  A point a whole tile from the seam is served by its own
# zone alone.
check("a tile away from a seam needs no neighbours", _seam_neighbours(138.2) == [])
# On a seam the natural zone is the eastern one, so the list holds only its
# western neighbour: what probe() has not already tried.
check("exactly on a seam adds the western zone", _seam_neighbours(138.0) == [53])
check("zone 1 wraps west to 60", _seam_neighbours(-179.97) == [60])
check("zone 60 wraps east to 1", _seam_neighbours(179.97) == [1])

check(
    "zone routing still agrees with the plain longitude rule away from seams",
    _zone_for_lon(2.35) == 31 and _zone_for_lon(-120.5) == 10,
)


# ---------------------------------------------------------------------------
# Zone selection (open_zone / GeoTesseraZarr.open_zone share this)
# ---------------------------------------------------------------------------

check("an explicit zone passes straight through", _resolve_zone(30, None, None) == 30)
check("a longitude selects its zone", _resolve_zone(None, -3.0, None) == 30)
# A whole-number longitude is the one people type; the old structural match
# accepted only float and rejected this as if nothing had been given.
check("a whole-number longitude works too", _resolve_zone(None, -3, None) == 30)
check(
    "a bbox selects the zone holding its centre",
    _resolve_zone(None, None, (-3.0, 53.4, -2.9, 53.5)) == 30,
)


def _raises_type_error(*args):
    try:
        _resolve_zone(*args)
    except TypeError:
        return True
    return False


check("no selector at all is an error", _raises_type_error(None, None, None))
check("two selectors at once is an error", _raises_type_error(30, -3.0, None))


# ---------------------------------------------------------------------------
# Tile seams
# ---------------------------------------------------------------------------


def _fake_zone(scales_2d, epsg=32653, px=10.0, ox=300000.0, oy=4050000.0):
    """A minimal zone Dataset the .tessera accessor can read."""
    h, w = scales_2d.shape
    b = 4
    emb = np.tile(np.arange(1, b + 1, dtype=np.int8)[:, None, None], (1, h, w))
    return xr.Dataset(
        {
            "embeddings": (("time", "band", "y", "x"), emb[None]),
            "scales": (("time", "y", "x"), np.asarray(scales_2d)[None]),
        },
        coords={
            "time": [2024],
            "band": np.arange(b),
            "x": ox + (np.arange(w) + 0.5) * px,
            "y": oy - (np.arange(h) + 0.5) * px,
        },
        attrs={
            "proj:code": f"EPSG:{epsg}",
            "spatial:transform": [px, 0.0, ox, 0.0, -px, oy],
            "geoemb:dimensions": b,
        },
    )


INF, NAN = np.float32("inf"), np.float32("nan")

# The artifact found at tile corners in the published v1 store: a patch of
# data with a single unwritten pixel at its centre.
sc = np.full((5, 5), np.float32(0.05))
sc[2, 2] = INF
holed = _fake_zone(sc)
cx = float(holed.x[2])
cy = float(holed.y[2])

v, st = holed.tessera.probe(cx, cy, 2024, search_px=0)
check("without repair a one-pixel hole reads as nodata", v is None and st == NODATA)

v, st = holed.tessera.probe(cx, cy, 2024, search_px=1)
check("with repair the hole resolves to a neighbour", v is not None and st == VALID)
check(
    "the repaired value is a real embedding, not a fill value",
    v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
)

# A NaN centre is a real answer and must never be searched away.
sw = np.full((5, 5), np.float32(0.05))
sw[2, 2] = NAN
watery = _fake_zone(sw)
v, st = watery.tessera.probe(float(watery.x[2]), float(watery.y[2]), 2024, search_px=1)
check("water is reported as water, not repaired into land", v is None and st == WATER)

# An entirely unwritten neighbourhood stays nodata however wide the search.
blank = _fake_zone(np.full((5, 5), INF))
v, st = blank.tessera.probe(float(blank.x[2]), float(blank.y[2]), 2024, search_px=2)
check("an all-unwritten window stays nodata", v is None and st == NODATA)

# Snapping must not silently answer for a point outside the grid.
v, st = holed.tessera.probe(float(holed.x[0]) - 10_000.0, cy, 2024)
check("a point far outside the grid reports outside", v is None and st == OUTSIDE)

# The nearest valid pixel wins, not the first one scanned.
sc2 = np.full((5, 5), INF)
sc2[2, 3] = np.float32(0.07)  # immediately east of centre
sc2[0, 0] = np.float32(0.09)  # further away
near = _fake_zone(sc2)
v, st = near.tessera.probe(float(near.x[2]), float(near.y[2]), 2024, search_px=2)
check(
    "repair picks the nearest valid pixel",
    v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
)

# sample_at() is the collapsing wrapper: no value of any kind becomes NaN.
check(
    "sample_at repairs the hole the same way probe does",
    np.allclose(
        holed.tessera.sample_at(cx, cy, 2024),
        np.array([1, 2, 3, 4]) * 0.05,
        atol=1e-6,
    ),
)
# probe and sample_at are the same lookup, so on one zone they must read a
# coordinate pair the same way.  They did not: probe took UTM while sample_at
# took lon/lat, so the same numbers meant different places on adjacent methods.
v_probe, _ = holed.tessera.probe(cx, cy, 2024)
v_sample = holed.tessera.sample_at(cx, cy, 2024)
check(
    "probe and sample_at agree on the same coordinates",
    v_probe is not None and np.allclose(v_probe, v_sample, atol=1e-6),
)

check(
    "sample_at returns a NaN row for water",
    np.all(
        np.isnan(watery.tessera.sample_at(float(watery.x[2]), float(watery.y[2]), 2024))
    ),
)

# ---------------------------------------------------------------------------
# Patch reads (read_patch)
# ---------------------------------------------------------------------------

from pyproj import Transformer  # noqa: E402  (import here keeps the header light)


def _fake_store(zone_datasets, n_bands=4):
    """A GeoTesseraZarr with its zone cache pre-filled — no store opened."""
    gt = GeoTesseraZarr.__new__(GeoTesseraZarr)
    gt.url = "fake://"
    gt.model_version = ""
    gt.build_version = ""
    gt.n_bands = n_bands
    gt.years = [2024]
    gt._cache = dict(zone_datasets)
    return gt


# Zone enumeration walks the ring the short way round.
check("a patch inside one zone spans one zone", _zones_spanned([2.0, 2.5], 2.2) == [31])
check(
    "a patch across a seam spans both zones",
    _zones_spanned([-0.01, 0.01], 0.0) == [30, 31],
)
check(
    "a patch across the antimeridian wraps 60 to 1",
    _zones_spanned([179.97, -179.97], 179.99) == [60, 1],
)

# The patch-centred CRS puts its false easting on the patch itself.
_to_patch = Transformer.from_crs("EPSG:4326", _patch_crs(0.5, 52.0), always_xy=True)
_pe, _pn = _to_patch.transform(0.5, 52.0)
check("the patch CRS centres its meridian on the patch", abs(_pe - 500000.0) < 1e-3)
_to_south = Transformer.from_crs("EPSG:4326", _patch_crs(0.5, -30.0), always_xy=True)
check(
    "a southern patch gets the southern false northing",
    _to_south.transform(0.5, -30.0)[1] > 5_000_000,
)

# -- Native fast path: one zone, pure slice, no resampling ------------------

flat = _fake_zone(np.full((12, 12), np.float32(0.05)))
_to_wgs = Transformer.from_crs(flat.attrs["proj:code"], "EPSG:4326", always_xy=True)
_clon, _clat = _to_wgs.transform(float(flat.x[6]), float(flat.y[6]))
gt_one = _fake_store({_zone_for_lon(_clon): flat})

patch, transform, crs = gt_one.read_patch(_clon, _clat, 2024, 6)
check("a one-zone patch has the exact shape asked for", patch.shape == (6, 6, 4))
check("a one-zone patch keeps the zone's own CRS", crs == flat.attrs["proj:code"])
check(
    "a one-zone patch holds native values, unresampled",
    np.allclose(patch, np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
)
# The transform must place the patch centre within half a pixel of the point.
_ce, _cn = transform * (3, 3)  # centre pixel's corner
check(
    "the native grid lands within half a pixel of the requested centre",
    abs(_ce + 5.0 - float(flat.x[6])) <= 5.0 and abs(_cn - 5.0 - float(flat.y[6])) <= 5.0,
)

# A patch larger than the zone's data is NaN-padded, never truncated.
patch, transform, crs = gt_one.read_patch(_clon, _clat, 2024, 20)
inside = np.isfinite(patch).any(axis=2)
check("an over-size patch keeps its shape and pads with NaN", patch.shape == (20, 20, 4))
check(
    "the padding surrounds intact native data",
    0 < inside.sum() == 144 and np.allclose(
        patch[inside], np.tile(np.array([1, 2, 3, 4]) * 0.05, (144, 1)), atol=1e-6
    ),
)

# -- Merge path: a patch straddling the utm30/utm31 seam --------------------


def _seam_zone(epsg, west_of_seam, lat=52.0, px=10.0, h=140, w=110, shift_px=0):
    """A fake zone whose data stops at the lon=0 seam, like real tiles do.

    ``shift_px`` moves the data edge east: overhang past the seam for the
    western zone, a gap after it for the eastern one.
    """
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    seam_e, seam_n = to_utm.transform(0.0, lat)
    if west_of_seam:
        ox, width, scale = seam_e - w * px, w + shift_px, np.float32(0.05)
    else:
        ox, width, scale = seam_e + shift_px * px, w, np.float32(0.07)
    oy = seam_n + h * px / 2
    return _fake_zone(np.full((h, width), scale), epsg=epsg, px=px, ox=ox, oy=oy)


gt_seam = _fake_store({30: _seam_zone(32630, True), 31: _seam_zone(32631, False)})
patch, transform, crs = gt_seam.read_patch(0.0, 52.0, 2024, 64)

check("a seam patch has the exact shape asked for", patch.shape == (64, 64, 4))
check("a seam patch comes back in a patch-centred CRS", "+proj=tmerc" in crs)
coverage = np.isfinite(patch).any(axis=2).mean()
check("a seam patch is covered from both zones", coverage > 0.98)
check(
    "west of the seam the western zone's values survive relocation",
    np.allclose(patch[32, 5], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
)
check(
    "east of the seam the eastern zone's values survive relocation",
    np.allclose(patch[32, 58], np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
)
# Both paths centre the requested point on pixel [size // 2, size // 2].
_ce, _cn = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(0.0, 52.0)
_pe, _pn = transform * (32 + 0.5, 32 + 0.5)
check(
    "the merged grid centres the requested point on its centre pixel",
    abs(_pe - _ce) < 1e-6 and abs(_pn - _cn) < 1e-6,
)

# Zone 30's data overhangs 5 px past the seam; zone 31's starts 3 px after it.
gt_overlap = _fake_store(
    {30: _seam_zone(32630, True, shift_px=5), 31: _seam_zone(32631, False, shift_px=3)}
)
patch, transform, crs = gt_overlap.read_patch(0.0, 52.0, 2024, 64)
check(
    "a sliver the owner lacks is filled by its neighbour",
    np.allclose(patch[32, 33], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
)
check(
    "where zones overlap the owning zone wins",
    np.allclose(patch[32, 36], np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
)

# Pinning dst_crs forces one shared grid even within a single zone.
patch, transform, crs = gt_one.read_patch(_clon, _clat, 2024, 6, dst_crs="EPSG:3857")
check("an explicit dst_crs is honoured", crs == "EPSG:3857" and patch.shape == (6, 6, 4))


if FAILED:
    print(f"{len(FAILED)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("all checks passed")
