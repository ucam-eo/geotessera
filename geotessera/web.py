"""Web visualization utilities for GeoTessera.

This module provides functions for generating web tiles and interactive
visualizations using Leaflet and other web technologies.
"""

from pathlib import Path
import logging

from ._typing import ProgressCallback

# Module-level logger
logger = logging.getLogger(__name__)


def prepare_mosaic_for_web(
    input_mosaic: str,
    output_path: str,
    target_crs: str = "EPSG:3857",
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Prepare an RGB mosaic for web visualization by reprojecting if needed.

    Args:
        input_mosaic: Path to input RGB mosaic (3-band GeoTIFF)
        output_path: Output path for web-ready mosaic
        target_crs: Target CRS for web visualization (default: Web Mercator)
        progress_callback: Optional progress callback

    Returns:
        Path to web-ready mosaic (may be same as input if no reprojection needed)
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.shutil import copy as copy_raster
    from .remote import atomic_output

    with rasterio.open(input_mosaic) as src:
        if src.crs == rasterio.crs.CRS.from_user_input(target_crs):
            if progress_callback:
                progress_callback(100, 100, "Mosaic already in the target CRS")
            return input_mosaic
        with WarpedVRT(
            src,
            crs=target_crs,
            add_alpha=src.count == 3,
            resampling=rasterio.enums.Resampling.bilinear,
        ) as vrt:
            with atomic_output(output_path, suffix=".tif") as staged:
                copy_raster(
                    vrt,
                    staged,
                    driver="GTiff",
                    compress="lzw",
                    tiled=True,
                    BIGTIFF="IF_SAFER",
                )
    if progress_callback:
        progress_callback(100, 100, "Mosaic prepared for web visualization")
    return str(output_path)


def geotiff_to_web_tiles(
    geotiff_path: str,
    output_dir: str,
    zoom_levels: tuple[int, int] = (8, 15),
    use_gdal_raster: bool = False,
) -> str:
    """Convert GeoTIFF to web tiles for interactive display.

    By default uses gdal2tiles.py for stability. Optionally can use the newer
    'gdal raster tile' command which may be faster but less stable.

    Args:
        geotiff_path: Path to input GeoTIFF
        output_dir: Directory for web tiles output
        zoom_levels: Min and max zoom levels
        use_gdal_raster: If True, use 'gdal raster tile' instead of gdal2tiles

    Returns:
        Path to tiles directory
    """
    try:
        import subprocess
    except ImportError:
        raise ImportError("gdal2tiles required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_zoom, max_zoom = zoom_levels

    # Use gdal raster tile if explicitly requested and available
    if use_gdal_raster:

        def _has_gdal_raster_tile() -> bool:
            """Check if 'gdal raster tile' command is available."""
            try:
                result = subprocess.run(
                    ["gdal", "raster", "tile", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                return False

        if _has_gdal_raster_tile():
            cmd = [
                "gdal",
                "raster",
                "tile",
                "--min-zoom",
                str(min_zoom),
                "--max-zoom",
                str(max_zoom),
                "--tiling-scheme",
                "WebMercatorQuad",
                "--convention",
                "tms",
                "--resampling",
                "bilinear",
                "--webviewer",
                "leaflet",
                "--num-threads",
                "1",
                geotiff_path,
                str(output_dir),
            ]

            try:
                logger.info(f"Running gdal raster tile: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    logger.debug("GDAL stdout: %s", result.stdout)
                if result.stderr:
                    logger.debug("GDAL stderr: %s", result.stderr)
                return str(output_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"gdal raster tile failed (return code {e.returncode}):")
                logger.error(f"Command: {' '.join(cmd)}")
                if e.stdout:
                    logger.error(f"Stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"Stderr: {e.stderr}")
                raise RuntimeError(f"gdal raster tile failed: {e}")
        else:
            raise RuntimeError(
                "gdal raster tile not available. Use default gdal2tiles or install gdal with raster tile support."
            )

    # Use traditional gdal2tiles.py (default)
    cmd = [
        "gdal2tiles.py",
        "-z",
        f"{min_zoom}-{max_zoom}",
        "-w",
        "leaflet",
        "-p",
        "mercator",  # Explicitly use mercator projection
        "--resampling",
        "bilinear",
        geotiff_path,
        str(output_dir),
    ]

    try:
        logger.info(f"Running gdal2tiles fallback: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.debug("gdal2tiles stdout: %s", result.stdout)
        if result.stderr:
            logger.debug("gdal2tiles stderr: %s", result.stderr)
        return str(output_dir)
    except subprocess.CalledProcessError as e:
        logger.error(f"gdal2tiles failed (return code {e.returncode}):")
        logger.error(f"Command: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(
            f"Tile generation failed with both gdal raster tile and gdal2tiles: {e}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Neither 'gdal raster tile' nor 'gdal2tiles.py' found. Install GDAL tools."
        )


def create_simple_web_viewer(
    tiles_dir: str,
    output_html: str,
    center_lon: float = 0,
    center_lat: float = 0,
    zoom: int = 10,
    title: str = "GeoTessera Visualization",
    region_file: str | None = None,
) -> str:
    """Create a simple HTML viewer for web tiles.

    Args:
        tiles_dir: Directory containing web tiles
        output_html: Output path for HTML file
        center_lon: Initial map center longitude
        center_lat: Initial map center latitude
        zoom: Initial zoom level
        title: Page title
        region_file: Optional GeoJSON/Shapefile boundary to overlay

    Returns:
        Path to created HTML file
    """
    import os
    from html import escape
    from urllib.parse import quote
    import folium
    from branca.element import Element
    from .inputs import read_region_file

    tiles_url = quote(
        Path(
            os.path.relpath(
                Path(tiles_dir).resolve(), Path(output_html).resolve().parent
            )
        ).as_posix(),
        safe="/",
    )
    map_ = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    map_.get_root().header.add_child(Element(f"<title>{escape(title)}</title>"))
    layer = folium.TileLayer(
        tiles=tiles_url + "/{z}/{x}/{y}.png",
        name="Tessera Data",
        attr="GeoTessera data",
        overlay=True,
        tms=True,
        opacity=0.8,
    ).add_to(map_)
    if region_file:
        frame = read_region_file(region_file)
        folium.GeoJson(
            frame,
            name="Region boundary",
            style_function=lambda _: {"color": "red", "fillOpacity": 0},
        ).add_to(map_)
    folium.LayerControl().add_to(map_)
    map_.get_root().html.add_child(
        Element(
            '<div style="position:absolute;bottom:20px;left:20px;z-index:1000;background:white;padding:10px">'
            '<label>GeoTessera opacity <input aria-label="GeoTessera opacity" type="range" min="0" max="1" step="0.05" value="0.8" '
            f'oninput="{layer.get_name()}.setOpacity(Number(this.value))"></label></div>'
        )
    )
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    map_.save(str(output_html))
    return str(output_html)


def create_coverage_summary_map(
    geotiff_paths: list[str], output_html: str, title: str = "GeoTessera Coverage Map"
) -> str:
    """Create an HTML map showing tile coverage.

    Args:
        geotiff_paths: List of GeoTIFF file paths
        output_html: Output HTML file path
        title: Map title

    Returns:
        Path to created HTML file
    """
    import folium
    from html import escape
    from .visualization import analyze_geotiff_coverage

    coverage = analyze_geotiff_coverage(geotiff_paths)
    if not coverage.get("tiles"):
        raise ValueError("No valid GeoTIFF files found")
    map_ = folium.Map()
    for tile in coverage["tiles"]:
        west, south, east, north = tile["bounds"]
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            fill=True,
            popup=f"{escape(Path(tile['path']).name)}<br>Year: {tile['year']}<br>Bands: {tile['bands']}",
        ).add_to(map_)
    bounds = coverage["bounds"]
    map_.fit_bounds(
        [[bounds["min_lat"], bounds["min_lon"]], [bounds["max_lat"], bounds["max_lon"]]]
    )
    from branca.element import Element

    map_.get_root().header.add_child(Element(f"<title>{escape(title)}</title>"))
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    map_.save(str(output_html))
    return str(output_html)
