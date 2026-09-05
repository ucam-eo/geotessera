"""Regression tests for raster metadata and mosaic reprojection."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform

from geotessera.core import GeoTessera
from geotessera.tiles import Tile
from geotessera.visualization import analyze_geotiff_coverage

UTM_CRS = "EPSG:32630"


def _write_raster(path, *, origin_x, pixel_size=10, value=1, bands=1):
    """Write a small UTM GeoTIFF with a parseable GeoTessera filename."""
    width = height = 10 if pixel_size == 10 else 5
    data = np.full((bands, height, width), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=data.dtype,
        crs=UTM_CRS,
        transform=from_origin(origin_x, 5_800_000, pixel_size, pixel_size),
    ) as dst:
        dst.write(data)


def _mosaic_runner():
    # Mosaic creation does not use instance state, so avoid Registry setup in
    # these isolated raster tests.
    return GeoTessera.__new__(GeoTessera)


def test_mosaic_converts_utm_resolution_to_target_crs_and_reuses_it(
    tmp_path, monkeypatch
):
    first = tmp_path / "grid_0.00_52.00_2024.tif"
    second = tmp_path / "grid_0.10_52.00_2024.tif"
    _write_raster(first, origin_x=500_000, pixel_size=10, value=10)
    # This tile covers the same 100 m width at a different source resolution.
    _write_raster(second, origin_x=500_100, pixel_size=20, value=20)

    with rasterio.open(first) as src:
        expected_transform, _, _ = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
    expected_resolution = (
        abs(expected_transform.a),
        abs(expected_transform.e),
    )

    runner = _mosaic_runner()
    original_reproject = runner._reproject_geotiff_file
    requested_resolutions = []

    def record_reproject(args):
        requested_resolutions.append(args[3])
        return original_reproject(args)

    monkeypatch.setattr(runner, "_reproject_geotiff_file", record_reproject)
    output = tmp_path / "mosaic.tif"
    runner.merge_geotiffs_to_mosaic([first, second], output, target_crs="EPSG:4326")

    assert requested_resolutions == [expected_resolution, expected_resolution]
    with rasterio.open(output) as mosaic:
        assert mosaic.crs.to_string() == "EPSG:4326"
        assert mosaic.res == expected_resolution
        # 10 m pixels become roughly 0.0001 degree pixels here.  Previously
        # their numeric value (10) was used as degrees, producing a 1x1 output.
        assert mosaic.width > 10
        assert mosaic.height > 5
        assert np.any(mosaic.read(1) > 0)


def test_mosaic_retains_resolution_when_target_crs_matches_source(tmp_path):
    source = tmp_path / "grid_0.00_52.00_2024.tif"
    _write_raster(source, origin_x=500_000, pixel_size=10)

    output = tmp_path / "mosaic.tif"
    _mosaic_runner().merge_geotiffs_to_mosaic([source], output, target_crs=UTM_CRS)

    with rasterio.open(output) as mosaic:
        assert mosaic.crs.to_string() == UTM_CRS
        assert mosaic.res == (10.0, 10.0)
        assert (mosaic.width, mosaic.height) == (10, 10)


def test_same_crs_mosaic_preserves_fractional_grid_and_values(tmp_path):
    source = tmp_path / "grid_0.00_52.00_2024.tif"
    transform = from_origin(500_003.25, 5_800_007.75, 10, 20)
    data = np.arange(24, dtype=np.uint8).reshape(1, 4, 6)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=UTM_CRS,
        transform=transform,
    ) as dst:
        dst.write(data)

    output = tmp_path / "mosaic.tif"
    _mosaic_runner().merge_geotiffs_to_mosaic([source], output, target_crs=UTM_CRS)

    with rasterio.open(output) as mosaic:
        assert mosaic.transform == transform
        np.testing.assert_array_equal(mosaic.read(), data)


def test_coverage_uses_geotiff_header_without_loading_embedding_data(
    tmp_path, monkeypatch
):
    source = tmp_path / "grid_0.00_52.00_2024.tif"
    _write_raster(source, origin_x=500_000, bands=3)

    def fail_if_pixels_are_loaded(self):
        raise AssertionError("coverage must not load GeoTIFF pixel data")

    monkeypatch.setattr(Tile, "load_embedding", fail_if_pixels_are_loaded)
    coverage = analyze_geotiff_coverage([source])

    assert coverage["band_counts"] == {3: 1}
    assert coverage["tiles"][0]["bands"] == 3
