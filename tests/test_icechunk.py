"""Offline tests for reading a dClimate-layout Icechunk store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
import zarr

icechunk = pytest.importorskip("icechunk")

from geotessera.icechunk import IcechunkStore, TileRegistry
from geotessera.registry import dataset_path, icechunk_dataset
from geotessera.store import NODATA, VALID, GeoTesseraZarr

YEARS = [2024, 2025]
BANDS = 4
PX = 10.0
WEST = 500000.0
# North covers northings 0..80, south 9_999_940..10_000_020 in its own
# CRS, which overlaps north by two rows above the equator.
GROUPS = {
    "31N": {"top": 80.0, "rows": 8, "cols": 6, "west": WEST, "complete": YEARS},
    "31S": {
        "top": 10_000_020.0,
        "rows": 8,
        "cols": 4,
        "west": WEST + 20.0,
        "complete": [2025],
    },
}


def _value(group, t, row, col):
    """Distinct per-pixel content, so misplaced reads are detectable."""
    base = 1 if group.endswith("N") else 51
    return base + 10 * t + row + col


def _write_group(root, name, spec):
    t, h, w = len(YEARS), spec["rows"], spec["cols"]
    g = root.create_group(name)
    g.attrs.update(
        {
            "proj:code": f"EPSG:{(32600 if name[-1] == 'N' else 32700) + 31}",
            "spatial:transform": [PX, 0.0, spec["west"], 0.0, -PX, spec["top"]],
            "spatial:shape": [h, w],
            "years_complete": spec["complete"],
        }
    )
    time = g.create_array("time", shape=(t,), dtype="int64", dimension_names=("time",))
    time[:] = (
        (np.array(YEARS) - 1970)
        .astype("datetime64[Y]")
        .astype("datetime64[ns]")
        .astype("int64")
    )
    time.attrs["units"] = "nanoseconds since 1970-01-01"
    band = g.create_array(
        "band", shape=(BANDS,), dtype="int64", dimension_names=("band",)
    )
    band[:] = np.arange(BANDS)
    emb = g.create_array(
        "embeddings",
        shape=(t, h, w, BANDS),
        chunks=(1, 4, 4, BANDS),
        dtype="int8",
        fill_value=0,
        dimension_names=("time", "northing", "easting", "band"),
    )
    scales = g.create_array(
        "scales",
        shape=(t, h, w),
        chunks=(1, 4, 4),
        dtype="float32",
        fill_value=np.nan,
        dimension_names=("time", "northing", "easting"),
    )
    ti, r, c = np.meshgrid(np.arange(t), np.arange(h), np.arange(w), indexing="ij")
    values = _value(name, ti, r, c).astype(np.int8)
    emb[:] = np.repeat(values[..., None], BANDS, axis=3) + np.arange(
        BANDS, dtype=np.int8
    )
    scales[:] = 1.0


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("ice") / "test.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    root = zarr.group(session.store)
    root.attrs.update(
        {"geoemb:model": "https://geotessera.org/model/1.1", "geoemb:dimensions": BANDS}
    )
    for name, spec in GROUPS.items():
        _write_group(root, name, spec)
    session.commit("fixture")
    return str(path)


def test_zone_merges_hemispheres(store_path):
    store = IcechunkStore(store_path)
    ds = store.open_zone(31)
    arrays = store.zone_arrays(31)
    assert ds.tessera.crs == "EPSG:32631"
    assert ds.tessera.years == YEARS
    # North keeps all 8 rows (centres 75..5); south drops its 2 rows
    # above the equator and keeps 6 (centres -5..-55).
    assert ds.sizes == {"time": 2, "band": BANDS, "y": 14, "x": 6}
    assert ds["y"].values[7] == 5.0 and ds["y"].values[8] == -5.0
    assert ds.attrs["spatial:transform"] == [PX, 0.0, WEST, 0.0, -PX, 80.0]
    assert ds["embeddings"].dims == ("time", "band", "y", "x")

    emb = arrays["embeddings"][1, :, :, :]
    assert emb.shape == (BANDS, 14, 6)
    # North row 7, column 0; south local row 2 (first kept), local column 0
    # at merged column 2.
    assert emb[0, 7, 0] == _value("31N", 1, 7, 0)
    assert emb[0, 8, 2] == _value("31S", 1, 2, 0)
    assert emb[3, 8, 2] == _value("31S", 1, 2, 0) + 3
    # South is narrower: columns outside it read as fill.
    assert emb[0, 8, 0] == 0 and np.isnan(arrays["scales"][1, 8, 0])

    # 2024 is incomplete in the south, so it reads as fill there.
    assert np.isnan(arrays["scales"][0, 8:, :]).all()
    assert np.isfinite(arrays["scales"][0, :8, :]).all()
    assert ds.attrs["geotessera:years_incomplete"] == {"31S": [2024]}

    # oindex, vindex and xarray windows agree with basic indexing.
    picked = arrays["embeddings"].oindex[1, [2, 0], 6:10, 3]
    assert np.array_equal(picked, emb[[2, 0], 6:10, 3])
    points = arrays["embeddings"].vindex[
        np.array([1, 1]), np.array([1, 1]), np.array([7, 9]), np.array([0, 3])
    ]
    assert points.tolist() == [emb[1, 7, 0], emb[1, 9, 3]]
    window = ds.isel(time=1, y=slice(6, 10), x=slice(1, 4))["embeddings"].values
    assert np.array_equal(window, emb[:, 6:10, 1:4])


def test_geotessera_zarr_reads_icechunk(store_path, tmp_path: Path):
    gt = GeoTesseraZarr(store_path)
    assert gt.years == YEARS and gt.n_bands == BANDS
    assert gt.model_version.endswith("/1.1")

    ds = gt.open_zone(zone=31)
    vec, status = ds.tessera.probe(WEST + 25.0, -15.0, 2025)
    assert status == VALID
    assert vec.tolist() == [_value("31S", 1, 3, 0) + b for b in range(BANDS)]
    # NaN scales are unembedded pixels, never water.
    assert ds.tessera.probe(WEST + 25.0, -15.0, 2024, search_px=0)[1] == NODATA

    lon, lat = 3.0, 0.0003  # on the central meridian, just north of the equator
    X = gt.sample_points([(lon, lat)], 2025, cross_zone=False)
    assert np.isfinite(X).all()

    files = gt.export_geotiffs(
        (2.9999, -0.0004, 3.0005, 0.0006), 2025, tmp_path, bands=[1]
    )
    with rasterio.open(files[0]) as src:
        data = src.read(1)
        assert src.crs.to_epsg() == 32631
    assert np.isfinite(data).any()


def test_exports_are_stamped_and_not_mixed(store_path, tmp_path: Path):
    from geotessera.raster import merge_geotiffs
    from geotessera.registry import find_dataset, recorded_dataset

    bbox = (2.9999, -0.0004, 3.0005, 0.0006)
    gt = GeoTesseraZarr(store_path)
    gt.dataset = find_dataset("1.1", "dclimate")
    dclimate = gt.export_geotiffs(bbox, 2025, tmp_path / "a", bands=[1])
    with rasterio.open(dclimate[0]) as src:
        assert src.tags()["TESSERA_DATASET_VARIANT"] == "dclimate"
    assert recorded_dataset(tmp_path / "a") == ("1.1", "dclimate")

    gt.dataset = find_dataset("1.1", "cambridge")
    with pytest.raises(ValueError, match="cannot be interchanged"):
        gt.export_geotiffs(bbox, 2025, tmp_path / "a", bands=[1])
    cambridge = gt.export_geotiffs(bbox, 2025, tmp_path / "b", bands=[1])
    with pytest.raises(ValueError, match="different datasets"):
        merge_geotiffs(dclimate + cambridge, tmp_path / "mosaic.tif")
    merge_geotiffs(dclimate, tmp_path / "mosaic.tif")


def test_dataset_resolution():
    url, registry = icechunk_dataset("v1.1")
    assert url.endswith("/v1.1/dclimate.icechunk") and registry.endswith("/parts")
    assert icechunk_dataset("v1.1", "cambridge") is None
    assert icechunk_dataset("v1") is None
    with pytest.raises(ValueError, match="not published as NPY"):
        dataset_path("1.1", "dclimate")


def test_format_default_stays_within_version():
    from geotessera.registry import default_variant, variant_note

    assert default_variant("2.0", "icechunk") == "2B-L~beta1"
    assert variant_note("2.0", "icechunk") is None
    assert default_variant("9.9", "npy") == "vultr"


def test_zarr_copy_keeps_icechunk_registry(monkeypatch):
    from geotessera import registry

    ds = registry.find_dataset("1.1", "dclimate")
    both = registry.Dataset(
        "1.1",
        "dclimate",
        zarr="v1.1-dclimate",
        icechunk=ds.icechunk,
        tile_registry=ds.tile_registry,
    )
    monkeypatch.setattr(
        registry,
        "DATASETS",
        tuple(both if d is ds else d for d in registry.DATASETS),
    )
    assert registry.zarr_store_url("v1.1").endswith("/zarr/v1.1-dclimate")
    assert registry.icechunk_dataset("v1.1") == (ds.icechunk, ds.tile_registry)


def test_tile_registry_keeps_latest_run(tmp_path: Path, monkeypatch):
    registry = TileRegistry("s3://example/parts", cache_dir=tmp_path)
    rows = {
        "a": pd.DataFrame(
            {
                "tile": ["t1", "t2"],
                "assembled_at": ["2026-01-01", "2026-01-01"],
                "embedded": [False, True],
                "bbox_west": [1.0, 179.9],
                "bbox_south": [0.0, 0.0],
                "bbox_east": [1.2, -179.9],
                "bbox_north": [0.2, 0.2],
            }
        ),
        "b": pd.DataFrame(
            {
                "tile": ["t1"],
                "assembled_at": ["2026-02-01"],
                "embedded": [True],
                "bbox_west": [1.0],
                "bbox_south": [0.0],
                "bbox_east": [1.2],
                "bbox_north": [0.2],
            }
        ),
    }
    monkeypatch.setattr(
        registry,
        "parts",
        lambda: [
            ("31", "N", 2025, "a"),
            ("31", "N", 2025, "b"),
            ("01", "N", 2024, "a"),
        ],
    )
    monkeypatch.setattr(registry, "_read", lambda path: rows[path].copy())

    tiles = registry.tiles(year=2025)
    assert tiles.set_index("tile")["embedded"].to_dict() == {"t1": True, "t2": True}
    # A row crossing the antimeridian matches from either side.
    assert registry.tiles(bbox=(-180.0, 0.0, -179.95, 0.1), year=2024)[
        "tile"
    ].tolist() == ["t2"]
    assert registry.tiles(bbox=(1.1, 0.1, 1.15, 0.15), year=2025)["tile"].tolist() == [
        "t1"
    ]


def test_xarray_point_reads_use_vindex(store_path, monkeypatch):
    from geotessera.icechunk import ZoneArray

    ds = IcechunkStore(store_path).open_zone(31)
    es = WEST + 5.0 + PX * np.array([0, 3, 5, 1])
    ns = np.array([75.0, 5.0, 15.0, 45.0])

    def outer(self, key):
        raise AssertionError("point reads must not read an outer product")

    monkeypatch.setattr(ZoneArray, "_outer", outer)
    X = ds.tessera.sample_points(list(zip(es, ns)), 2025)
    rows = ((80.0 - ns) / PX - 0.5).astype(int)
    cols = ((es - WEST) / PX - 0.5).astype(int)
    expected = [
        [_value("31N", 1, r, c) + b for b in range(BANDS)] for r, c in zip(rows, cols)
    ]
    assert X.tolist() == expected

    # Vectorized keys mixing point arrays with a slice.
    sub = ds["embeddings"].isel(
        time=1,
        y=xr.DataArray(rows, dims="p"),
        x=xr.DataArray(cols, dims="p"),
        band=slice(1, 3),
    )
    assert sub.transpose("p", "band").values.tolist() == [row[1:3] for row in expected]


def test_webmap_reuse_keys_on_resolved_store(tmp_path: Path, monkeypatch):
    from geotessera import workflows

    rendered = []

    def fake_stream_rgb(output, **kwargs):
        rendered.append(kwargs.get("variant"))
        Path(output).write_bytes(b"rgb")
        return Path(output)

    monkeypatch.setattr(workflows, "stream_rgb", fake_stream_rgb)
    output = tmp_path / "rgb.tif"
    region = {"bbox": "0.1,52.1,0.2,52.2", "year": 2024, "version": "v1.1"}
    assert workflows.cached_stream_rgb(output, **region)[1] is False
    assert workflows.cached_stream_rgb(output, **region)[1] is True
    # A default that resolves to another store must not reuse the rendering.
    from geotessera import registry

    monkeypatch.setattr(registry, "zarr_store_url", lambda *a: "s3://other/store")
    assert workflows.cached_stream_rgb(output, **region)[1] is False
    assert rendered == [None, None]
