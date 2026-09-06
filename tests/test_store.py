"""Offline tests for point, patch, region, and store reads."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from pyproj import Transformer
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, LocalStore, MemoryStore, ObjectStore

from geotessera.registry import zarr_store_url
from geotessera.store import (
    NODATA,
    OUTSIDE,
    RETRY_CONFIG,
    VALID,
    WATER,
    GeoTesseraZarr,
    _patch_crs,
    _resolve_zone,
    _seam_neighbours,
    _store_cache_key,
    _zone_for_lon,
    _zones_spanned,
    zarr_store,
)


def assert_check(label, condition, *details):
    """Retain the old assertion labels while failing at the first mismatch."""
    assert condition, " ".join((label, *(str(detail) for detail in details)))


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


def _fake_store(zone_datasets, n_bands=4):
    """A GeoTesseraZarr over in-memory zarr groups built from the datasets."""
    gt = GeoTesseraZarr.__new__(GeoTesseraZarr)
    gt.url = "fake://"
    gt.model_version = ""
    gt.build_version = ""
    gt.n_bands = n_bands
    gt.years = [2024]
    gt.depths = {n_bands: "embeddings"}
    gt._cache = dict(zone_datasets)
    gt._root = zarr.group()
    for zone, ds in zone_datasets.items():
        group = gt._root.create_group(f"utm{zone:02d}")
        for name, var in ds.data_vars.items():
            arr = group.create_array(name, shape=var.shape, dtype=var.dtype)
            arr[:] = var.values
    return gt


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


def test_zone_routing_and_selection():
    assert_check("zone interior needs no neighbours", _seam_neighbours(3.0) == [])
    assert_check("just east of a seam looks west", _seam_neighbours(138.03) == [53])
    assert_check("just west of a seam looks east", _seam_neighbours(137.97) == [54])
    # The band is half a tile wide each side, since a tile belongs to the zone
    # holding its centre.  A point a whole tile from the seam is served by its own
    # zone alone.
    assert_check(
        "a tile away from a seam needs no neighbours", _seam_neighbours(138.2) == []
    )
    # On a seam the natural zone is the eastern one, so the list holds only its
    # western neighbour: what probe() has not already tried.
    assert_check(
        "exactly on a seam adds the western zone", _seam_neighbours(138.0) == [53]
    )
    assert_check("zone 1 wraps west to 60", _seam_neighbours(-179.97) == [60])
    assert_check("zone 60 wraps east to 1", _seam_neighbours(179.97) == [1])

    assert_check(
        "zone routing still agrees with the plain longitude rule away from seams",
        _zone_for_lon(2.35) == 31 and _zone_for_lon(-120.5) == 10,
    )

    # ---------------------------------------------------------------------------
    # Zone selection (open_zone / GeoTesseraZarr.open_zone share this)
    # ---------------------------------------------------------------------------

    assert_check(
        "an explicit zone passes straight through", _resolve_zone(30, None, None) == 30
    )
    assert_check("a longitude selects its zone", _resolve_zone(None, -3.0, None) == 30)
    # A whole-number longitude is the one people type; the old structural match
    # accepted only float and rejected this as if nothing had been given.
    assert_check(
        "a whole-number longitude works too", _resolve_zone(None, -3, None) == 30
    )
    assert_check(
        "a bbox selects the zone holding its centre",
        _resolve_zone(None, None, (-3.0, 53.4, -2.9, 53.5)) == 30,
    )

    def _raises_type_error(*args):
        try:
            _resolve_zone(*args)
        except TypeError:
            return True
        return False

    assert_check("no selector at all is an error", _raises_type_error(None, None, None))
    assert_check(
        "two selectors at once is an error", _raises_type_error(30, -3.0, None)
    )


def test_point_reads_repair_only_nodata():
    # The artifact found at tile corners in the published v1 store: a patch of
    # data with a single unwritten pixel at its centre.
    sc = np.full((5, 5), np.float32(0.05))
    sc[2, 2] = INF
    holed = _fake_zone(sc)
    cx = float(holed.x[2])
    cy = float(holed.y[2])

    v, st = holed.tessera.probe(cx, cy, 2024, search_px=0)
    assert_check(
        "without repair a one-pixel hole reads as nodata", v is None and st == NODATA
    )

    v, st = holed.tessera.probe(cx, cy, 2024, search_px=1)
    assert_check(
        "with repair the hole resolves to a neighbour", v is not None and st == VALID
    )
    assert_check(
        "the repaired value is a real embedding, not a fill value",
        v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )

    # A NaN centre is a real answer and must never be searched away.
    sw = np.full((5, 5), np.float32(0.05))
    sw[2, 2] = NAN
    watery = _fake_zone(sw)
    v, st = watery.tessera.probe(
        float(watery.x[2]), float(watery.y[2]), 2024, search_px=1
    )
    assert_check(
        "water is reported as water, not repaired into land", v is None and st == WATER
    )

    # An entirely unwritten neighbourhood stays nodata however wide the search.
    blank = _fake_zone(np.full((5, 5), INF))
    v, st = blank.tessera.probe(float(blank.x[2]), float(blank.y[2]), 2024, search_px=2)
    assert_check("an all-unwritten window stays nodata", v is None and st == NODATA)

    # Snapping must not silently answer for a point outside the grid.
    v, st = holed.tessera.probe(float(holed.x[0]) - 10_000.0, cy, 2024)
    assert_check(
        "a point far outside the grid reports outside", v is None and st == OUTSIDE
    )

    # The nearest valid pixel wins, not the first one scanned.
    sc2 = np.full((5, 5), INF)
    sc2[2, 3] = np.float32(0.07)  # immediately east of centre
    sc2[0, 0] = np.float32(0.09)  # further away
    near = _fake_zone(sc2)
    v, st = near.tessera.probe(float(near.x[2]), float(near.y[2]), 2024, search_px=2)
    assert_check(
        "repair picks the nearest valid pixel",
        v is not None and np.allclose(v, np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
    )

    # sample_at() is the collapsing wrapper: no value of any kind becomes NaN.
    assert_check(
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
    assert_check(
        "probe and sample_at agree on the same coordinates",
        v_probe is not None and np.allclose(v_probe, v_sample, atol=1e-6),
    )

    assert_check(
        "sample_at returns a NaN row for water",
        np.all(
            np.isnan(
                watery.tessera.sample_at(float(watery.x[2]), float(watery.y[2]), 2024)
            )
        ),
    )


def test_patch_reads_across_zone_seams():
    # Zone enumeration walks the ring the short way round.
    assert_check(
        "a patch inside one zone spans one zone",
        _zones_spanned([2.0, 2.5], 2.2) == [31],
    )
    assert_check(
        "a patch across a seam spans both zones",
        _zones_spanned([-0.01, 0.01], 0.0) == [30, 31],
    )
    assert_check(
        "a patch across the antimeridian wraps 60 to 1",
        _zones_spanned([179.97, -179.97], 179.99) == [60, 1],
    )

    # The patch-centred CRS puts its false easting on the patch itself.
    _to_patch = Transformer.from_crs("EPSG:4326", _patch_crs(0.5, 52.0), always_xy=True)
    _pe, _pn = _to_patch.transform(0.5, 52.0)
    assert_check(
        "the patch CRS centres its meridian on the patch", abs(_pe - 500000.0) < 1e-3
    )
    _to_south = Transformer.from_crs(
        "EPSG:4326", _patch_crs(0.5, -30.0), always_xy=True
    )
    assert_check(
        "a southern patch gets the southern false northing",
        _to_south.transform(0.5, -30.0)[1] > 5_000_000,
    )

    # -- Native fast path: one zone, pure slice, no resampling ------------------

    flat = _fake_zone(np.full((12, 12), np.float32(0.05)))
    _to_wgs = Transformer.from_crs(flat.attrs["proj:code"], "EPSG:4326", always_xy=True)
    _clon, _clat = _to_wgs.transform(float(flat.x[6]), float(flat.y[6]))
    gt_one = _fake_store({_zone_for_lon(_clon): flat})

    patch, transform, crs = gt_one.read_patch(_clon, _clat, 2024, 6)
    assert_check(
        "a one-zone patch has the exact shape asked for", patch.shape == (6, 6, 4)
    )
    assert_check(
        "a one-zone patch keeps the zone's own CRS", crs == flat.attrs["proj:code"]
    )
    assert_check(
        "a one-zone patch holds native values, unresampled",
        np.allclose(patch, np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )
    # The transform must place the patch centre within half a pixel of the point.
    _ce, _cn = transform * (3, 3)  # centre pixel's corner
    assert_check(
        "the native grid lands within half a pixel of the requested centre",
        abs(_ce + 5.0 - float(flat.x[6])) <= 5.0
        and abs(_cn - 5.0 - float(flat.y[6])) <= 5.0,
    )

    # A patch larger than the zone's data is NaN-padded, never truncated.
    patch, transform, crs = gt_one.read_patch(_clon, _clat, 2024, 20)
    inside = np.isfinite(patch).any(axis=2)
    assert_check(
        "an over-size patch keeps its shape and pads with NaN",
        patch.shape == (20, 20, 4),
    )
    assert_check(
        "the padding surrounds intact native data",
        0 < inside.sum() == 144
        and np.allclose(
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

    assert_check(
        "a seam patch has the exact shape asked for", patch.shape == (64, 64, 4)
    )
    assert_check(
        "a seam patch comes back in a named patch-centred CRS",
        "Transverse Mercator" in crs and "Tessera patch" in crs,
    )
    coverage = np.isfinite(patch).any(axis=2).mean()
    assert_check("a seam patch is covered from both zones", coverage > 0.98)
    assert_check(
        "west of the seam the western zone's values survive relocation",
        np.allclose(patch[32, 5], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )
    assert_check(
        "east of the seam the eastern zone's values survive relocation",
        np.allclose(patch[32, 58], np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
    )
    # Both paths centre the requested point on pixel [size // 2, size // 2].
    _ce, _cn = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        0.0, 52.0
    )
    _pe, _pn = transform * (32 + 0.5, 32 + 0.5)
    assert_check(
        "the merged grid centres the requested point on its centre pixel",
        abs(_pe - _ce) < 1e-6 and abs(_pn - _cn) < 1e-6,
    )

    # Zone 30's data overhangs 5 px past the seam; zone 31's starts 3 px after it.
    gt_overlap = _fake_store(
        {
            30: _seam_zone(32630, True, shift_px=5),
            31: _seam_zone(32631, False, shift_px=3),
        }
    )
    patch, transform, crs = gt_overlap.read_patch(0.0, 52.0, 2024, 64)
    assert_check(
        "a sliver the owner lacks is filled by its neighbour",
        np.allclose(patch[32, 33], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )
    assert_check(
        "where zones overlap the owning zone wins",
        np.allclose(patch[32, 36], np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
    )

    # Pinning dst_crs forces one shared grid even within a single zone.
    patch, transform, crs = gt_one.read_patch(
        _clon, _clat, 2024, 6, dst_crs="EPSG:3857"
    )
    assert_check(
        "an explicit dst_crs is honoured",
        crs == "EPSG:3857" and patch.shape == (6, 6, 4),
    )


def test_bulk_point_sampling():
    sc = np.full((5, 5), np.float32(0.05))
    sc[2, 2] = np.float32("inf")
    holed = _fake_zone(sc)
    cx, cy = float(holed.x[2]), float(holed.y[2])
    sw = np.full((5, 5), np.float32(0.05))
    sw[2, 2] = np.float32("nan")
    watery = _fake_zone(sw)
    gt_seam = _fake_store({30: _seam_zone(32630, True), 31: _seam_zone(32631, False)})
    gt_overlap = _fake_store(
        {
            30: _seam_zone(32630, True, shift_px=5),
            31: _seam_zone(32631, False, shift_px=3),
        }
    )

    # One vectorised read must answer like the per-point path.
    _far = float(holed.x[0]) - 10_000.0
    vals = holed.tessera.sample_points(
        [(cx - 10, cy), (cx, cy), (_far, cy)], 2024, progress=False
    )
    assert_check(
        "bulk sampling reads land natively and repairs the hole",
        np.allclose(vals[0], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6)
        and np.allclose(vals[1], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )
    assert_check(
        "bulk sampling reports beyond-grid points as NaN", np.all(np.isnan(vals[2]))
    )
    assert_check(
        "bulk sampling answers water as NaN without repairing it",
        np.all(
            np.isnan(
                watery.tessera.sample_points(
                    [(float(watery.x[2]), float(watery.y[2]))], 2024, progress=False
                )
            )
        ),
    )

    # Store level: one bulk read per zone, seam gaps served by the neighbour.
    _t30 = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    _t31 = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    _seam30 = _t30.transform(0.0, 52.0)
    _seam31 = _t31.transform(0.0, 52.0)
    _back30 = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
    _back31 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    _west_pt = _back30.transform(_seam30[0] - 500.0, _seam30[1])
    _east_pt = _back31.transform(_seam31[0] + 500.0, _seam31[1])
    _gap_pt = _back31.transform(_seam31[0] + 15.0, _seam31[1])  # zone 31 starts 3 px in

    vals = gt_seam.sample_points([_west_pt, _east_pt], 2024, progress=False)
    assert_check(
        "store-level bulk sampling routes zones and preserves order",
        np.allclose(vals[0], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6)
        and np.allclose(vals[1], np.array([1, 2, 3, 4]) * 0.07, atol=1e-6),
    )
    # Filtering invalid coordinates must preserve the original row order
    # when the remaining points are grouped by zone and read through Zarr.
    mixed = [(np.nan, 52.0), _east_pt, (0.0, np.inf), _west_pt, (-np.inf, 52.0)]
    invalid_row = np.full(4, np.nan, np.float32)
    np.testing.assert_array_equal(
        gt_seam.sample_points(mixed, 2024),
        np.stack([invalid_row, vals[1], invalid_row, vals[0], invalid_row]),
    )
    vals = gt_overlap.sample_points([_gap_pt], 2024, progress=False)
    assert_check(
        "a point in the owner's gap is served by the zone next door",
        np.allclose(vals[0], np.array([1, 2, 3, 4]) * 0.05, atol=1e-6),
    )


def test_matryoshka_and_quantized_reads():
    flat = _fake_zone(np.full((12, 12), np.float32(0.05)))
    # A store that declares a 2-dimension prefix array, as v2 stores do.
    deep = flat.copy()
    deep["embeddings_d2"] = (
        ("time", "band_d2", "y", "x"),
        deep["embeddings"].values[:, :2],
    )
    gt_deep = _fake_store({53: deep})
    gt_deep.depths = {2: "embeddings_d2", 4: "embeddings"}

    _ll53 = Transformer.from_crs(deep.attrs["proj:code"], "EPSG:4326", always_xy=True)
    _lon53, _lat53 = _ll53.transform(float(deep.x[6]), float(deep.y[6]))
    _w53, _s53 = _ll53.transform(float(deep.x[1]), float(deep.y[10]))
    _e53, _n53 = _ll53.transform(float(deep.x[10]), float(deep.y[1]))
    _box53 = (_w53, _s53, _e53, _n53)

    full, _, _ = gt_deep.read_region(_box53, 2024)
    d2, _, _ = gt_deep.read_region(_box53, 2024, depth=2)
    assert_check(
        "a depth read equals the prefix of the full read",
        d2.shape[2] == 2 and np.array_equal(d2, full[:, :, :2], equal_nan=True),
    )
    assert_check(
        "depth point samples match the full prefix",
        np.array_equal(
            gt_deep.sample_points([(_lon53, _lat53)], 2024, progress=False, depth=2)[0],
            gt_deep.sample_points([(_lon53, _lat53)], 2024, progress=False)[0][:2],
        ),
    )
    _d2cat = np.concatenate(
        [b for b, _, _ in gt_deep.iter_region(_box53, 2024, depth=2, strip_rows=4)],
        axis=0,
    )
    assert_check(
        "depth strips equal the depth region",
        np.array_equal(_d2cat, d2, equal_nan=True),
    )

    def _bad_depth():
        try:
            gt_deep.read_region(_box53, 2024, depth=16)
        except ValueError as problem:
            return "available: [2, 4]" in str(problem)
        return False

    assert_check("an undeclared depth raises and lists what exists", _bad_depth())

    emb_q, scales_q, _transform_q, _crs_q = gt_deep.read_region_quantized(_box53, 2024)
    assert_check(
        "a quantised read dequantises to exactly read_region",
        emb_q.dtype == np.int8
        and np.array_equal(
            deep.tessera.dequantise(emb_q.transpose(2, 0, 1), scales_q),
            full,
            equal_nan=True,
        ),
    )


def test_streamed_region_reads():
    flat = _fake_zone(np.full((12, 12), np.float32(0.05)))
    _box = (float(flat.x[1]), float(flat.y[10]), float(flat.x[10]), float(flat.y[1]))
    _whole, _t_whole = flat.tessera.read_region(_box, 2024)
    _strips = list(flat.tessera.iter_region(_box, 2024, strip_rows=4))
    assert_check(
        "iter_region strips concatenate to exactly read_region",
        np.array_equal(
            np.concatenate([b for b, _ in _strips], axis=0), _whole, equal_nan=True
        ),
    )
    assert_check(
        "the first strip shares read_region's transform and later ones step down",
        _strips[0][1] == _t_whole
        and _strips[1][1].f == _t_whole.f - 4 * flat.tessera.pixel_size,
    )


def test_zarr_store_adapters_and_errors(tmp_path: Path):
    flat = _fake_zone(np.full((12, 12), np.float32(0.05)))

    from geotessera.store import zarr_store

    _loc = zarr_store("https://example.org/store")
    assert_check(
        "an http url reads through an obstore store with retries configured",
        isinstance(_loc, ObjectStore)
        and _loc.store.retry_config["max_retries"] == RETRY_CONFIG["max_retries"],
    )

    assert_check(
        "a local path becomes a LocalStore",
        isinstance(zarr_store(tmp_path / "store.zarr"), LocalStore),
    )
    assert_check(
        "another url scheme resolves through fsspec",
        isinstance(zarr_store((tmp_path / "store.zarr").as_uri()), FsspecStore),
    )
    assert_check("a zarr Store passes through untouched", zarr_store(_loc) is _loc)

    # A full GeoTesseraZarr over a Store instance, and the same store wrapped
    # in zarr's experimental cache, must read identically.

    _mem = MemoryStore()
    with warnings.catch_warnings():
        # zarr warns that v3 consolidated metadata is not yet in the spec
        warnings.simplefilter("ignore")
        flat.to_zarr(_mem, group="utm53", zarr_format=3, mode="w")
        _g = zarr.open_group(_mem, mode="a")
        _g.attrs.update({"geoemb:dimensions": 4, "geoemb:model": "fake"})
        zarr.consolidate_metadata(_mem)

    gt_store = GeoTesseraZarr(_mem)
    assert_check(
        "a Store instance initialises the full store",
        gt_store.n_bands == 4 and gt_store.years == [2024],
    )
    _pt53 = Transformer.from_crs(
        flat.attrs["proj:code"], "EPSG:4326", always_xy=True
    ).transform(float(flat.x[6]), float(flat.y[6]))
    _want = gt_store.sample_points([_pt53], 2024, progress=False)

    gt_cached = GeoTesseraZarr(
        CacheStore(_mem, cache_store=MemoryStore(), max_size=64 * 1024 * 1024)
    )
    assert_check(
        "a cache-wrapped store reads identically",
        np.array_equal(gt_cached.sample_points([_pt53], 2024, progress=False), _want),
    )

    from geotessera.registry import zarr_store_url

    assert_check(
        "version names resolve to their store path",
        zarr_store_url("v1").endswith("/zarr/v1")
        and zarr_store_url("v1.1").endswith("/zarr/v1.1")
        and zarr_store_url("v2").endswith("/zarr/v2-2B-L~beta1")
        and zarr_store_url("v2-2B-L~beta1").endswith("/zarr/v2-2B-L~beta1"),
    )

    # A missing remote store is represented by an empty in-memory store: this
    # exercises the same error translation without performing an HTTP request.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "geotessera.store.zarr_store", lambda *args, **kwargs: MemoryStore()
        )
        with pytest.raises(ValueError) as exc_info:
            GeoTesseraZarr("https://invalid.example/store")
    msg = str(exc_info.value)
    assert "Failed to open zarr store" in msg
    assert "Known versions:" in msg
    assert "zarr_store_url()" in msg


def test_persistent_cache_keying(tmp_path: Path):
    assert_check(
        "canonical store urls key by their dataset path",
        _store_cache_key(zarr_store_url("v1")) == "v1"
        and _store_cache_key(zarr_store_url("v2")) == "v2-2B-L_beta1"
        and _store_cache_key(zarr_store_url("v1") + "/") == "v1",
    )
    assert_check(
        "non-canonical locations key by slug plus digest",
        _store_cache_key("https://mirror.example.org/zarr/v1")
        != _store_cache_key(zarr_store_url("v1"))
        and _store_cache_key("/data/local-v1-copy")
        != _store_cache_key(zarr_store_url("v1")),
    )

    for name, model in (("s1", "1.0"), ("s2", "2.0")):
        group = zarr.open_group(str(tmp_path / name), mode="w")
        group.attrs["geoemb:model"] = model
        array = group.create_array("data", shape=(4,), dtype="f4")
        array[:] = float(model[0])

    cache = tmp_path / "cache"
    group1 = zarr.open_group(
        zarr_store(str(tmp_path / "s1"), cache_dir=cache), mode="r"
    )
    group2 = zarr.open_group(
        zarr_store(str(tmp_path / "s2"), cache_dir=cache), mode="r"
    )
    assert group1.attrs["geoemb:model"] == "1.0"
    assert group2.attrs["geoemb:model"] == "2.0"
    assert float(group1["data"][0]) == 1.0
    assert float(group2["data"][0]) == 2.0
    assert len(list(cache.iterdir())) == 2

    with pytest.raises(ValueError):
        zarr_store(zarr_store(str(tmp_path / "s1")), cache_dir=cache)
