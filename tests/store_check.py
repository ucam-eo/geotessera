"""Checks for seam-aware point reads in ``geotessera.store``.

Run by tests/store.t.  Every check works on a small in-memory zone Dataset
carrying the three pixel states of a real one — a scale for land, ``NaN``
for water, ``+inf`` for never written — so the suite stays offline and fast.

Prints one ``ok - <name>`` line per check and exits non-zero if any failed.
"""

import sys

import numpy as np
import xarray as xr

from geotessera.store import (
    NODATA,
    OUTSIDE,
    VALID,
    WATER,
    _resolve_zone,
    _seam_neighbours,
    _zone_for_lon,
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

if FAILED:
    print(f"{len(FAILED)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("all checks passed")
