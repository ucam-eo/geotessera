from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from zarr.storage import LocalStore

import geotessera.store as store_module
from geotessera.store import GeoTesseraZarr, _utm_envelope, zarr_store


def _point_dataset(scales, *, xs=None, ys=None, alternate_names=False):
    scales = np.asarray(scales, dtype=np.float32)
    h, w = scales.shape
    xs = np.arange(w, dtype=float) * 10.0 + 5.0 if xs is None else np.asarray(xs)
    ys = 15.0 - np.arange(h, dtype=float) * 10.0 if ys is None else np.asarray(ys)
    xname, yname = ("xc", "yc") if alternate_names else ("x", "y")
    embedding = np.arange(1, h * w + 1, dtype=np.int8).reshape(1, 1, h, w)
    return xr.Dataset(
        {
            "embeddings": (("time", "band", yname, xname), embedding),
            "scales": (("time", yname, xname), scales[None]),
        },
        coords={
            "time": [2024],
            "band": [0],
            xname: xs,
            yname: ys,
        },
        attrs={
            "proj:code": "EPSG:32631",
            "spatial:transform": [10.0, 0.0, 0.0, 0.0, -10.0, 20.0],
            "geoemb:dimensions": 1,
        },
    )


def test_zarr_store_accepts_path(tmp_path: Path):
    assert isinstance(zarr_store(tmp_path / "store.zarr"), LocalStore)


def test_geo_tessera_zarr_accepts_path(monkeypatch, tmp_path: Path):
    location = tmp_path / "store.zarr"
    seen = {}

    def fake_store(value, **kwargs):
        seen["location"] = value
        return object()

    class Root:
        def __init__(self):
            self.attrs = {}

        @staticmethod
        def keys():
            return []

    monkeypatch.setattr(store_module, "zarr_store", fake_store)
    monkeypatch.setattr(store_module.zarr, "open_group", lambda *args, **kwargs: Root())

    result = GeoTesseraZarr(location)

    assert seen["location"] == str(location)
    assert result.url == str(location)


@pytest.mark.parametrize("alternate_names", [False, True], ids=["xy", "xcyc"])
def test_point_sampling_uses_same_first_coordinate_tie(alternate_names):
    ds = _point_dataset(
        [[1.0, 1.0], [1.0, 1.0]],
        xs=[5.0, 15.0],
        ys=[15.0, 5.0],
        alternate_names=alternate_names,
    )
    acc = ds.tessera

    # A tie follows the first coordinate in the original axis: west for X,
    # north for descending Y.  Values on either side select the expected
    # adjacent pixel in both scalar and bulk paths.
    expected = {
        (10.0, 10.0): 1.0,
        (9.9, 10.0): 1.0,
        (10.1, 10.0): 2.0,
        (10.0, 10.1): 1.0,
        (10.0, 9.9): 3.0,
        (np.nextafter(10.0, -np.inf), 10.0): 1.0,
        (np.nextafter(10.0, np.inf), 10.0): 2.0,
        (10.0, np.nextafter(10.0, np.inf)): 1.0,
        (10.0, np.nextafter(10.0, -np.inf)): 3.0,
    }
    for point, value in expected.items():
        assert acc.sample_at(*point, 2024)[0] == value
    bulk = acc.sample_points(list(expected), 2024)
    np.testing.assert_array_equal(bulk[:, 0], list(expected.values()))

    # The published failure mode: a north/valid and south/water pair on a
    # Y tie must choose the north pixel in both paths.
    mixed = _point_dataset(
        [[1.0, 1.0], [np.nan, 1.0]],
        xs=[5.0, 15.0],
        ys=[15.0, 5.0],
        alternate_names=alternate_names,
    )
    assert mixed.tessera.sample_at(5.0, 10.0, 2024)[0] == 1.0
    np.testing.assert_array_equal(
        mixed.tessera.sample_points([(5.0, 10.0)], 2024)[:, 0], [1.0]
    )


@pytest.mark.parametrize(
    ("scales", "target", "status"),
    [
        ([[1.0]], (5.0, 15.0), "valid"),
        ([[np.nan]], (5.0, 15.0), "water"),
        ([[np.inf]], (5.0, 15.0), "nodata"),
    ],
)
def test_point_sampling_preserves_valid_water_and_unwritten(scales, target, status):
    acc = _point_dataset(scales).tessera
    value, actual = acc.probe(*target, 2024, search_px=0)
    assert actual == status
    assert (value is not None) == (status == "valid")

    bulk = acc.sample_points([target], 2024)
    if status == "valid":
        assert np.isfinite(bulk).all()
    else:
        assert np.isnan(bulk).all()


def test_water_is_not_repaired_but_unwritten_is():
    acc = _point_dataset([[np.nan], [np.inf], [0.5]]).tessera

    value, status = acc.probe(5.0, 15.0, 2024, search_px=1)
    assert value is None and status == "water"
    assert np.isnan(acc.sample_at(5.0, 15.0, 2024)).all()

    value, status = acc.probe(5.0, 5.0, 2024, search_px=1)
    assert status == "valid" and value[0] == 1.5
    np.testing.assert_array_equal(acc.sample_points([(5.0, 5.0)], 2024), [[1.5]])


def test_point_sampling_rejects_nonfinite_targets():
    ds = _point_dataset(
        np.ones((3, 3)),
        xs=[5.0, 15.0, 25.0],
        ys=[25.0, 15.0, 5.0],
    )
    acc = ds.tessera

    assert acc.sample_at(15.0, 15.0, 2024)[0] == 5.0
    np.testing.assert_array_equal(
        acc.sample_points([(15.0, 15.0)], 2024)[:, 0], [5.0]
    )
    invalid = [(v, 15.0) for v in (np.nan, np.inf, -np.inf)]
    invalid += [(15.0, v) for v in (np.nan, np.inf, -np.inf)]
    for point in invalid:
        value, status = acc.probe(*point, 2024)
        assert value is None and status == "outside"
        assert np.isnan(acc.sample_at(*point, 2024)).all()
    # An invalid row must not disturb a valid row in the same batch.
    values = acc.sample_points(invalid[:3] + [(15.0, 15.0)] + invalid[3:], 2024)
    np.testing.assert_array_equal(values[:, 0], [np.nan] * 3 + [5.0] + [np.nan] * 3)


def test_point_sampling_handles_singleton_axes():
    acc = _point_dataset([[2.0]], xs=[5.0], ys=[15.0]).tessera
    np.testing.assert_array_equal(acc.sample_at(5.0, 15.0, 2024), [2.0])
    np.testing.assert_array_equal(acc.sample_points([(5.0, 15.0)], 2024), [[2.0]])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_point_sampling_handles_empty_axes(axis):
    acc = _point_dataset(np.ones((2, 2))).isel({axis: slice(0, 0)}).tessera
    assert acc.probe(5.0, 15.0, 2024) == (None, "outside")
    assert np.isnan(acc.sample_points([(5.0, 15.0)], 2024)).all()


def test_point_sampling_preserves_one_pixel_outside_tolerance():
    acc = _point_dataset(np.ones((2, 2))).tessera
    points = [(-5.0, 15.0), (25.0, 15.0), (-5.01, 15.0), (25.01, 15.0)]
    expected = [1.0, 2.0, np.nan, np.nan]
    np.testing.assert_array_equal(acc.sample_points(points, 2024)[:, 0], expected)
    np.testing.assert_array_equal([acc.sample_at(*p, 2024)[0] for p in points], expected)


def test_geo_point_sampling_rejects_nonfinite_coordinates(monkeypatch):
    gt = GeoTesseraZarr.__new__(GeoTesseraZarr)
    gt.n_bands = 1
    gt.depths = {1: "embeddings"}

    def should_not_route(*args, **kwargs):
        raise AssertionError("non-finite points must not be routed to a zone")

    monkeypatch.setattr(gt, "open_zone", should_not_route)
    invalid = [(v, 52.0) for v in (np.nan, np.inf, -np.inf)]
    invalid += [(0.0, v) for v in (np.nan, np.inf, -np.inf)]
    for point in invalid:
        assert gt.probe(*point, 2024) == (None, "outside")
        assert np.isnan(gt.sample_at(*point, 2024)).all()
    values = gt.sample_points(invalid, 2024)
    assert values.shape == (6, 1)
    assert np.isnan(values).all()


@pytest.mark.parametrize(
    ("bbox", "crs"),
    [
        ((2.0, 50.0, 4.0, 51.0), "EPSG:32631"),
        ((2.0, -51.0, 4.0, -50.0), "EPSG:32731"),
    ],
)
def test_utm_envelope_includes_curved_edge_extrema(bbox, crs):
    envelope = _utm_envelope(bbox, crs)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Sample the four edges densely enough to expose extrema between corners.
    edge = np.linspace(0.0, 1.0, 1001)
    lon0, lat0, lon1, lat1 = bbox
    points = np.concatenate(
        [
            np.column_stack((lon0 + edge * (lon1 - lon0), np.full(edge.size, lat0))),
            np.column_stack((lon0 + edge * (lon1 - lon0), np.full(edge.size, lat1))),
            np.column_stack((np.full(edge.size, lon0), lat0 + edge * (lat1 - lat0))),
            np.column_stack((np.full(edge.size, lon1), lat0 + edge * (lat1 - lat0))),
        ]
    )
    eastings, northings = transformer.transform(points[:, 0], points[:, 1])

    assert envelope[0] <= min(eastings)
    assert envelope[1] <= min(northings)
    assert envelope[2] >= max(eastings)
    assert envelope[3] >= max(northings)

    corner_northings = transformer.transform(
        [lon0, lon0, lon1, lon1], [lat0, lat1, lat0, lat1]
    )[1]
    assert envelope[1] < min(corner_northings) - 400.0 or envelope[3] > max(
        corner_northings
    ) + 400.0
