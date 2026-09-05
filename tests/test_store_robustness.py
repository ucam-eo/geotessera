from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer
from zarr.storage import LocalStore

import geotessera.store as store_module
from geotessera.store import GeoTesseraZarr, _utm_envelope, zarr_store


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
