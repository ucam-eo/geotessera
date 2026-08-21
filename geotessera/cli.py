"""Simplified GeoTessera command-line interface.

Focused on downloading tiles and creating visualizations from the generated GeoTIFFs.
"""

# Will configure logging after imports

import importlib.resources
import os
import webbrowser
import threading
import time
import http.server
import socketserver
import tempfile
import urllib.parse
import logging
from pathlib import Path
from typing import Optional, Callable
from typing_extensions import Annotated

import typer
from rich.logging import RichHandler
from rich.box import ROUNDED
from geotessera import __version__
from geotessera.registry import (
    EMBEDDINGS_DIR_NAME,
    LANDMASKS_DIR_NAME,
    format_bytes,
    tile_from_world,
    tile_to_landmask_filename,
    tile_to_embedding_paths,
    tile_to_geotiff_path,
)
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich import print as rprint

from .core import GeoTessera
from .country import get_country_bbox
from .visualization import (
    calculate_bbox_from_file,
    create_pca_mosaic,
)
from .web import (
    geotiff_to_web_tiles,
    create_simple_web_viewer,
    prepare_mosaic_for_web,
)
from ._terminal import console, emoji


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urllib.parse.urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_region_file(url: str) -> str:
    """Download a region file from a URL to a temporary location.

    Args:
        url: The URL to download from

    Returns:
        Path to the temporary downloaded file

    Raises:
        Exception: If download fails
    """
    temp_path = None
    try:
        # Create a temporary file with appropriate extension
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        if path.endswith(".geojson"):
            suffix = ".geojson"
        elif path.endswith(".json"):
            suffix = ".json"
        elif path.endswith(".shp"):
            suffix = ".shp"
        elif path.endswith(".gpkg"):
            suffix = ".gpkg"
        else:
            # Default to geojson for unknown extensions
            suffix = ".geojson"

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        # An existing file would send the downloader down its
        # If-Modified-Since path, so drop the empty placeholder first.
        os.unlink(temp_path)

        from geotessera.registry import download_file_to_temp

        download_file_to_temp(url, cache_path=Path(temp_path))

        return temp_path

    except Exception as e:
        # Don't leave the temp file behind if the download failed.
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Failed to download region file from {url}: {e}")


def format_bbox(bbox_coords) -> str:
    """Format bounding box coordinates for pretty display.

    Args:
        bbox_coords: Tuple of (min_lon, min_lat, max_lon, max_lat)

    Returns:
        Compact human-readable string representation of bbox with degree symbols
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords

    # Format longitude with E/W direction
    min_lon_str = f"{abs(min_lon):.6f}°{'W' if min_lon < 0 else 'E'}"
    max_lon_str = f"{abs(max_lon):.6f}°{'W' if max_lon < 0 else 'E'}"

    # Format latitude with N/S direction
    min_lat_str = f"{abs(min_lat):.6f}°{'S' if min_lat < 0 else 'N'}"
    max_lat_str = f"{abs(max_lat):.6f}°{'S' if max_lat < 0 else 'N'}"

    return f"[{min_lon_str}, {min_lat_str}] - [{max_lon_str}, {max_lat_str}]"


def point_to_tile_bbox(lon: float, lat: float) -> tuple:
    """Convert a point to the bounding box of its containing tile.

    Tiles are 0.1x0.1 degree squares centered at 0.05-degree offsets.
    Returns a point bbox at the tile center, which the registry query
    will expand by 0.05 degrees to match exactly one tile.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) for the containing tile
    """
    tile_lon, tile_lat = tile_from_world(lon, lat)
    # Return point bbox at tile center - registry expands by 0.05 degrees
    # which will match exactly this one tile
    return (tile_lon, tile_lat, tile_lon, tile_lat)


app = typer.Typer(
    name="geotessera",
    help=f"GeoTessera v{__version__}: Download satellite embedding tiles as GeoTIFFs",
    add_completion=False,
    rich_markup_mode="rich",
)


# Helper to create tables with appropriate settings for dumb terminals
def create_table(show_header=True, header_style=None, box=None, **kwargs):
    """Create a Rich Table with appropriate settings for terminal capabilities.

    Uses Rich Console's built-in detection.

    Args:
        show_header: Whether to show table header (default: True)
        header_style: Style for header (default: None)
        box: Box style override. If None, automatically determined based on terminal.
        **kwargs: Additional arguments passed to Table constructor

    Returns:
        Configured Rich Table instance
    """
    if not console.is_terminal:
        # Dumb terminal or piped output: no box, no edges, minimal padding
        # Remove padding from kwargs if present to avoid conflict
        kwargs.pop("padding", None)
        return Table(
            show_header=show_header,
            header_style=None,  # No styling in dumb terminals
            box=None,
            safe_box=True,
            show_edge=False,
            padding=(0, 1),  # Minimal padding: 0 vertical, 1 horizontal space
            collapse_padding=True,  # Collapse padding for cleaner output
            **kwargs,
        )
    else:
        # Smart terminal: use rounded box if not specified
        actual_box = box if box is not None else ROUNDED
        return Table(
            show_header=show_header, header_style=header_style, box=actual_box, **kwargs
        )


def create_panel(content, title=None, border_style=None):
    """Return content directly without panel wrapper.

    Args:
        content: Content to display (table, text, etc.)
        title: Panel title (ignored)
        border_style: Panel border style (ignored)

    Returns:
        Content without panel wrapper (tables display well on their own)
    """
    # Don't nest tables in panels - tables look good on their own
    return content


def create_progress(*args, **kwargs):
    """Create a Rich Progress instance with appropriate settings for terminal capabilities.

    Uses Rich Console's built-in detection.

    Args:
        *args: Column definitions for progress bar
        **kwargs: Additional arguments passed to Progress constructor

    Returns:
        Configured Rich Progress instance
    """
    # If console not specified, use our configured console
    if "console" not in kwargs:
        kwargs["console"] = console

    if not console.is_terminal:
        # Dumb terminal or piped output: disable progress bar, just show text updates
        # Filter out BarColumn and TimeRemainingColumn which use box characters
        filtered_args = []
        for arg in args:
            # Skip BarColumn and TimeRemainingColumn in dumb terminals
            if not isinstance(arg, (BarColumn, TimeRemainingColumn)):
                filtered_args.append(arg)
        return Progress(*filtered_args, **kwargs)
    else:
        # Smart terminal: use all columns as provided
        return Progress(*args, **kwargs)


def create_progress_callback(progress: Progress, task_id: TaskID) -> Callable:
    """Create a progress callback for core library operations."""

    def progress_callback(current: int, total: int, status: str = None):
        if status:
            progress.update(task_id, completed=current, total=total, status=status)
        else:
            progress.update(task_id, completed=current, total=total)

    return progress_callback


@app.command()
def info(
    tiles_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--tiles", help="Analyze tile files/directory (GeoTIFF or NPY format)"
        ),
    ] = None,
    geotiffs: Annotated[
        Optional[Path],
        typer.Option(
            "--geotiffs",
            help="(Deprecated: use --tiles) Analyze GeoTIFF files/directory",
        ),
    ] = None,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version",
            help="Tessera dataset version (e.g. v1, v1.1, v2; default v1)",
        ),
    ] = "v1",
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Tessera dataset variant (default: the version's default "
            "variant, e.g. vultr for v1, cambridge for v1.1, 2B-L~beta1 "
            "for v2). Run 'geotessera info' to list available datasets.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Show information about tile files or library.

    Supports both GeoTIFF and NPY format tiles (auto-detected).
    """

    # Support both --tiles and --geotiffs for backwards compatibility
    input_path = tiles_dir or geotiffs

    if input_path:
        # Analyze tiles using Tile abstraction (supports both formats)
        from geotessera.tiles import discover_tiles, discover_formats

        # Discover all available formats to show complete info
        all_formats = discover_formats(input_path)

        if not all_formats:
            # Force line break before path for deterministic output regardless of terminal width
            rprint(f"[red]No tiles found in\n{input_path}[/red]")
            rprint("[yellow]Supported formats:[/yellow]")
            rprint("  - GeoTIFF: *.tif/*.tiff files")
            rprint(
                "  - NPY: global_0.1_degree_representation/{year}/grid_{lon}_{lat}/*.npy structure"
            )
            raise typer.Exit(1)

        # Use the preferred format (NPY if available, otherwise first available)
        tiles = discover_tiles(input_path)

        # Determine format string for display
        if len(all_formats) > 1:
            # Both formats present
            format_str = ", ".join(sorted([fmt.upper() for fmt in all_formats.keys()]))
            format_str += " (using npy)"  # Lowercase to match test expectation
        else:
            # Single format
            format_str = list(all_formats.keys())[0].upper()

        # Build coverage info from tiles
        coverage = {
            "total_files": len(tiles),
            "years": sorted(list(set(str(t.year) for t in tiles))),
            "crs": sorted(list(set(str(t.crs) for t in tiles))),
            "band_counts": {},
            "bounds": {
                "min_lon": float("inf"),
                "min_lat": float("inf"),
                "max_lon": float("-inf"),
                "max_lat": float("-inf"),
            },
            "tiles": [],
            "format": format_str,
        }

        # Process each tile
        for tile in tiles:
            # Update bounds (use tile coordinates for efficiency)
            coverage["bounds"]["min_lon"] = min(
                coverage["bounds"]["min_lon"], tile.lon - 0.05
            )
            coverage["bounds"]["min_lat"] = min(
                coverage["bounds"]["min_lat"], tile.lat - 0.05
            )
            coverage["bounds"]["max_lon"] = max(
                coverage["bounds"]["max_lon"], tile.lon + 0.05
            )
            coverage["bounds"]["max_lat"] = max(
                coverage["bounds"]["max_lat"], tile.lat + 0.05
            )

            # Track band count (from metadata, not loading data)
            band_count = 128  # All tiles have 128 channels
            coverage["band_counts"][band_count] = (
                coverage["band_counts"].get(band_count, 0) + 1
            )

            # Tile info for verbose output
            if verbose:
                coverage["tiles"].append(
                    {
                        "path": tile.grid_name,
                        "year": str(tile.year),
                        "tile_lat": tile.lat,
                        "tile_lon": tile.lon,
                        "bands": band_count,
                    }
                )

        # Create analysis table
        analysis_table = create_table(show_header=False, box=None)
        analysis_table.add_row("Total tiles:", str(coverage["total_files"]))
        analysis_table.add_row("Format:", coverage["format"].upper())
        analysis_table.add_row("Years:", ", ".join(coverage["years"]))
        analysis_table.add_row("CRS:", ", ".join(coverage["crs"]))

        rprint(
            create_panel(
                analysis_table,
                title="[bold]📊 Tile Analysis[/bold]",
                border_style="blue",
            )
        )

        bounds = coverage["bounds"]

        bounds_table = create_table(show_header=False, box=None)
        bounds_table.add_row(
            "Longitude:", f"{bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}"
        )
        bounds_table.add_row(
            "Latitude:", f"{bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}"
        )

        rprint(
            create_panel(
                bounds_table, title="[bold]🗺️ Bounding Box[/bold]", border_style="green"
            )
        )

        bands_table = create_table(show_header=True, header_style="bold blue")
        bands_table.add_column("Band Count")
        bands_table.add_column("Files", justify="right")

        for bands_count, count in coverage["band_counts"].items():
            bands_table.add_row(f"{bands_count} bands", str(count))

        rprint(
            create_panel(
                bands_table,
                title="[bold]🎵 Band Information[/bold]",
                border_style="cyan",
            )
        )

        if verbose:
            tiles_table = create_table(show_header=True, header_style="bold blue")
            tiles_table.add_column("Filename")
            tiles_table.add_column("Coordinates")
            tiles_table.add_column("Bands", justify="right")

            for tile in coverage["tiles"][:10]:
                tiles_table.add_row(
                    Path(tile["path"]).name,
                    f"({tile['tile_lon']}, {tile['tile_lat']})",
                    str(tile["bands"]),
                )

            rprint(
                create_panel(
                    tiles_table,
                    title="[bold]📁 First 10 Tiles[/bold]",
                    border_style="yellow",
                )
            )

    else:
        # Show library info
        gt = GeoTessera(
            dataset_version=dataset_version, dataset_variant=dataset_variant
        )
        years = gt.registry.get_available_years()

        # Count tiles per year using fast pandas operations
        tiles_per_year = gt.registry.get_tile_counts_by_year()

        # Count total landmasks using fast pandas operations
        total_landmasks = gt.registry.get_landmask_count()

        info_table = create_table(show_header=False, box=None)
        info_table.add_row("Version:", gt.version)
        info_table.add_row("Available years:", ", ".join(map(str, years)))

        # Show tiles per year
        for year in years:
            count = tiles_per_year.get(year, 0)
            info_table.add_row(f"  {year} tiles:", f"{count:,}")

        info_table.add_row("Total landmasks:", f"{total_landmasks:,}")

        rprint(
            create_panel(
                info_table,
                title=f"[bold]🌍 GeoTessera v{__version__} Library Info[/bold]",
                border_style="blue",
            )
        )

        # List every known (version, variant) dataset so users can discover
        # valid --dataset-version/--dataset-variant combinations.
        from geotessera.registry import KNOWN_DATASETS, VERSION_DEFAULT_VARIANTS

        datasets_table = create_table(box=None)
        datasets_table.add_column("Version")
        datasets_table.add_column("Variant")
        datasets_table.add_column("Repository dir")
        datasets_table.add_column("Status")
        for ds_version, ds_variant, ds_dir in KNOWN_DATASETS:
            is_default = VERSION_DEFAULT_VARIANTS.get(ds_version) == ds_variant
            datasets_table.add_row(
                ds_version,
                ds_variant + (" (default)" if is_default else ""),
                ds_dir or "-",
                "available" if ds_dir else "coming soon",
            )
        rprint(
            create_panel(
                datasets_table,
                title="[bold]📚 Known Datasets[/bold] "
                "(select with --dataset-version/--dataset-variant)",
                border_style="cyan",
            )
        )


@app.command()
def coverage(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output PNG file path (JSON and HTML will be created in same directory)",
        ),
    ] = Path("tessera_coverage.png"),
    year: Annotated[
        Optional[int],
        typer.Option(
            "--year",
            help="Specific year to visualize (e.g., 2024). If not specified, shows multi-year analysis.",
        ),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="GeoJSON/Shapefile to focus coverage map on specific region (file path or URL)",
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country",
            help="Country name to focus coverage map on (e.g., 'United Kingdom', 'UK', 'GB')",
        ),
    ] = None,
    bbox: Annotated[
        Optional[str],
        typer.Option(
            "--bbox",
            help="Bounding box: 'lon,lat' (single tile) or 'min_lon,min_lat,max_lon,max_lat'",
        ),
    ] = None,
    tile: Annotated[
        Optional[str],
        typer.Option("--tile", help="Single tile by any point within it: 'lon,lat'"),
    ] = None,
    tile_color: Annotated[
        str,
        typer.Option(
            "--tile-color",
            help="Color for tile rectangles (when not using multi-year colors)",
        ),
    ] = "red",
    tile_alpha: Annotated[
        float, typer.Option("--tile-alpha", help="Transparency of tiles (0.0-1.0)")
    ] = 0.6,
    tile_size: Annotated[
        float,
        typer.Option(
            "--tile-size", help="Size multiplier for tiles (1.0 = actual size)"
        ),
    ] = 1.0,
    width_pixels: Annotated[
        int, typer.Option("--width", help="Output image width in pixels")
    ] = 2000,
    no_countries: Annotated[
        bool, typer.Option("--no-countries", help="Don't show country boundaries")
    ] = False,
    no_multi_year_colors: Annotated[
        bool,
        typer.Option("--no-multi-year-colors", help="Disable multi-year color coding"),
    ] = False,
    by_source: Annotated[
        bool,
        typer.Option(
            "--by-source",
            help="Render each (version, variant) source in a distinct colour. "
            "When set without an explicit --dataset-version/--dataset-variant, "
            "downloads every known version's manifest and renders all sources.",
        ),
    ] = False,
    dataset_version: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-version",
            help="Tessera dataset version (e.g. v1, v1.1, v2). Defaults to v1 "
            "for the single-source view; with --by-source, omitting this "
            "flag means 'all known versions'. Pass 'all' explicitly to "
            "force the multi-version view.",
        ),
    ] = None,
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Tessera dataset variant. Defaults to the version's "
            "default variant for the single-source view; with --by-source, "
            "omitting this means 'all variants'. Pass 'all' explicitly to "
            "force the multi-variant view. Run 'geotessera info' to list "
            "available datasets.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Cache directory")
    ] = None,
    registry_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--registry-dir",
            help="Directory containing registry.parquet and landmasks.parquet files",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Generate coverage visualizations showing Tessera embedding availability.

    This command generates multiple outputs in one pass:
    1. PNG map - Static visualization with tiles overlaid on world map
    2. coverage.json + coverage_YYYY.json - Split JSON data (metadata + per-year tile lists)
    3. coverage_texture.png - Pre-rendered coverage texture for the globe
    4. globe.html - Interactive 3D globe visualization

    The PNG map supports regional filtering, year selection, and customization.
    The HTML globe shows global multi-year coverage with interactive hover tooltips.

    For PNG maps, when no specific year is requested, uses three colors:
    - Green: All available years present for this tile
    - Blue: Only the latest year available for this tile
    - Orange: Partial years coverage (some combination of years)

    Examples:
        # Generate coverage visualizations for a region
        geotessera coverage --region-file study_area.geojson
        # Creates: tessera_coverage.png, coverage.json, coverage_YYYY.json, globe.html

        # Specify output location
        geotessera coverage --country "Colombia" -o maps/colombia.png
        # Creates: maps/colombia.png, maps/coverage.json, maps/coverage_YYYY.json, maps/globe.html

        # Customize PNG visualization
        geotessera coverage --region-file area.geojson --tile-alpha 0.3 --width 3000
    """
    from .visualization import visualize_global_coverage
    from rich.progress import BarColumn, TextColumn, TimeRemainingColumn

    # Process region file or country if provided
    region_bbox = None
    country_geojson_file = None
    region_file_temp = None  # Track if we created a temporary file

    # Check mutual exclusivity of region options
    region_sources = sum(1 for x in [bbox, tile, region_file, country] if x)
    if region_sources > 1:
        rprint(
            "[red]Error: Cannot specify multiple region options. "
            "Choose one of: --bbox, --tile, --region-file, --country[/red]"
        )
        raise typer.Exit(1)

    if tile:
        try:
            tile_coords = tuple(map(float, tile.split(",")))
            if len(tile_coords) != 2:
                rprint("[red]Error: --tile must be 'lon,lat'[/red]")
                raise typer.Exit(1)
            lon, lat = tile_coords
            tile_center = tile_from_world(lon, lat)
            region_bbox = point_to_tile_bbox(lon, lat)
            rprint(
                f"[green]Point ({lon}, {lat}) -> tile "
                f"grid_{tile_center[0]:.2f}_{tile_center[1]:.2f}[/green]"
            )
            rprint(f"[green]Region bounding box:[/green] {format_bbox(region_bbox)}")
        except ValueError as e:
            rprint(f"[red]Error: Invalid --tile format. Use 'lon,lat': {e}[/red]")
            raise typer.Exit(1)
    elif bbox:
        try:
            bbox_coords = tuple(map(float, bbox.split(",")))
            if len(bbox_coords) == 2:
                # Two coordinates = single tile
                lon, lat = bbox_coords
                tile_center = tile_from_world(lon, lat)
                region_bbox = point_to_tile_bbox(lon, lat)
                rprint(
                    f"[green]Point ({lon}, {lat}) -> tile "
                    f"grid_{tile_center[0]:.2f}_{tile_center[1]:.2f}[/green]"
                )
            elif len(bbox_coords) == 4:
                region_bbox = bbox_coords
            else:
                rprint(
                    "[red]Error: bbox must be 'lon,lat' (single tile) "
                    "or 'min_lon,min_lat,max_lon,max_lat'[/red]"
                )
                raise typer.Exit(1)
            rprint(f"[green]Region bounding box:[/green] {format_bbox(region_bbox)}")
        except ValueError:
            rprint(
                "[red]Error: Invalid bbox format. Use: 'lon,lat' or "
                "'min_lon,min_lat,max_lon,max_lat'[/red]"
            )
            raise typer.Exit(1)
    elif region_file:
        try:
            # Check if region_file is a URL
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_temp = download_region_file(region_file)
                region_file_path = region_file_temp
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(
                        f"[red]Error: Region file {region_file} does not exist[/red]"
                    )
                    raise typer.Exit(1)
                region_file_path = str(region_path)

            region_bbox = calculate_bbox_from_file(region_file_path)
            rprint(f"[green]Region bounding box: {format_bbox(region_bbox)}[/green]")
        except Exception as e:
            rprint(f"[red]Error reading region file: {e}[/red]")
            raise typer.Exit(1)
    elif country:
        # Create progress bar for country data download
        with create_progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
        ) as progress:
            country_task = progress.add_task(
                f"{emoji('🌍 ')}Loading country data...",
                total=100,
                status="Checking cache...",
            )

            def country_progress_callback(current: int, total: int, status: str = None):
                progress.update(
                    country_task,
                    completed=current,
                    total=total,
                    status=status or "Processing...",
                )

            try:
                # Get country lookup instance
                from .country import get_country_lookup

                country_lookup = get_country_lookup(
                    progress_callback=country_progress_callback
                )

                # Get both bbox and geometry
                region_bbox = country_lookup.get_bbox(country)
                country_gdf = country_lookup.get_geometry(country)

                # Create temporary GeoJSON file for the country boundary
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".geojson", delete=False
                ) as tmp:
                    tmp.close()
                    country_gdf.to_file(tmp.name, driver="GeoJSON")
                    country_geojson_file = tmp.name

                progress.update(country_task, completed=100, status="Complete")
            except ValueError as e:
                rprint(f"[red]Error: {e}[/red]")
                rprint(
                    "[blue]Check the country name spelling (uses Natural Earth admin-0 names)[/blue]"
                )
                raise typer.Exit(1)
            except Exception as e:
                rprint(f"[red]Error fetching country data: {e}[/red]")
                raise typer.Exit(1)

        rprint(f"[green]Using country '{country}': {format_bbox(region_bbox)}[/green]")

    # Initialize GeoTessera
    if verbose:
        rprint("[blue]Initializing GeoTessera...[/blue]")

    # Resolve flag defaults. The defaults differ between single-source and
    # --by-source modes so the "complete picture" rendering doesn't require
    # the user to type --dataset-version=all every time.
    if by_source:
        version_spec = dataset_version if dataset_version is not None else "all"
        variant_spec = dataset_variant if dataset_variant is not None else "all"
    else:
        from geotessera.registry import _parse_dataset_version, default_variant

        version_spec = dataset_version if dataset_version is not None else "v1"
        variant_spec = (
            dataset_variant
            if dataset_variant is not None
            else default_variant(_parse_dataset_version(version_spec)[1])
        )

    # The placeholder GeoTessera below initialises one Registry to get its
    # cache paths; additional manifests are downloaded into the same cache
    # dir when by-source spans multiple versions.
    init_version = (
        "v1" if (by_source and version_spec.lower() == "all") else version_spec
    )
    init_variant = (
        "vultr" if (by_source and variant_spec.lower() == "all") else variant_spec
    )

    gt = GeoTessera(
        dataset_version=init_version,
        dataset_variant=init_variant,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None,
    )

    # Generate coverage map
    try:
        with create_progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"{emoji('🔄 ')}Generating coverage map...",
                total=100,
                status="Starting...",
            )

            if verbose:
                rprint(
                    f"[blue]Generating coverage map for year: {year if year else 'All years'}[/blue]"
                )

            # Resolve --output into (png_path, ancillary_dir). When the user
            # passes a directory-style path (no image suffix, or trailing slash,
            # or an existing directory), treat it as a folder and put both the
            # PNG and the ancillary coverage.json / globe.html / per-year files
            # inside it. Otherwise the path is a PNG file and ancillaries go
            # next to it in the parent directory.
            image_suffixes = {".png", ".jpg", ".jpeg"}
            looks_like_dir = (
                output.suffix.lower() not in image_suffixes
                or output.is_dir()
                or str(output).endswith(("/", os.sep))
            )
            if looks_like_dir:
                ancillary_dir = output
                ancillary_dir.mkdir(parents=True, exist_ok=True)
                output = ancillary_dir / "tessera_coverage.png"
            else:
                ancillary_dir = output.parent or Path(".")
                ancillary_dir.mkdir(parents=True, exist_ok=True)

            # When using region files or countries, default to no countries for cleaner view
            show_countries_final = not no_countries and not region_file and not country

            # Determine which region file to use (original region file or country boundary)
            region_file_to_use = None
            if region_file:
                region_file_to_use = (
                    region_file_path
                    if "region_file_path" in locals()
                    else str(region_file)
                )
            elif (
                country and "country_geojson_file" in locals() and country_geojson_file
            ):
                region_file_to_use = country_geojson_file

            if by_source:
                # Multi-source render. Download one manifest per requested
                # dataset (npy/ directory) and concatenate them in the
                # renderer.
                from geotessera.visualization import visualize_sources_coverage
                from geotessera.registry import (
                    _parse_dataset_version,
                    download_file_to_temp,
                    manifest_url,
                    published_datasets,
                )

                if version_spec.lower() == "all":
                    target_dataset_dirs = [d for _v, _var, d in published_datasets()]
                else:
                    _vpath, vnorm = _parse_dataset_version(version_spec)
                    target_dataset_dirs = [
                        d for v, _var, d in published_datasets() if v == vnorm
                    ]

                manifest_paths = []
                for dataset_dir in target_dataset_dirs:
                    cache_path = (
                        gt.registry._registry_cache_dir
                        / dataset_dir
                        / "manifest.parquet"
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if dataset_dir == gt.registry._dataset_path:
                        # Already downloaded during GeoTessera init.
                        manifest_paths.append(gt.registry.manifest_path)
                        continue
                    url = manifest_url(dataset_dir)
                    try:
                        manifest_paths.append(
                            Path(download_file_to_temp(url, cache_path=cache_path))
                        )
                    except Exception as e:
                        rprint(f"[yellow]Skipping {dataset_dir}: {e}[/yellow]")

                if not manifest_paths:
                    rprint("[red]No manifests available to render.[/red]")
                    raise typer.Exit(1)

                variants_filter = None
                if variant_spec.lower() != "all":
                    variants_filter = [variant_spec]

                output_path = visualize_sources_coverage(
                    manifest_path=manifest_paths,
                    output_path=str(output),
                    year=year,
                    width_pixels=width_pixels,
                    show_countries=show_countries_final,
                    tile_alpha=tile_alpha,
                    tile_size=tile_size,
                    variants=variants_filter,
                    region_bbox=region_bbox,
                    region_file=region_file_to_use,
                    progress_callback=create_progress_callback(progress, task),
                )
            else:
                output_path = visualize_global_coverage(
                    tessera_client=gt,
                    output_path=str(output),
                    year=year,
                    width_pixels=width_pixels,
                    show_countries=show_countries_final,
                    tile_color=tile_color,
                    tile_alpha=tile_alpha,
                    tile_size=tile_size,
                    multi_year_colors=not no_multi_year_colors,
                    progress_callback=create_progress_callback(progress, task),
                    region_bbox=region_bbox,
                    region_file=region_file_to_use,
                )

        rprint(f"[green]{emoji('✅ ')}Coverage map saved to: {output_path}[/green]")

        # Show next steps hint
        if region_file:
            rprint("[blue]Next step: Download data for your region:[/blue]")
            rprint(
                f"[cyan]  geotessera download --region-file {region_file} --output tiles/[/cyan]"
            )
        elif country:
            rprint("[blue]Next step: Download data for your country:[/blue]")
            rprint(
                f'[cyan]  geotessera download --country "{country}" --output tiles/[/cyan]'
            )
        else:
            rprint("[blue]Next step: Download data for a specific region:[/blue]")
            rprint(
                "[cyan]  geotessera download --bbox 'lon1,lat1,lon2,lat2' --output tiles/[/cyan]"
            )

        # Show summary statistics
        available_embeddings = gt.registry.get_available_embeddings()
        if available_embeddings:
            if year:
                tile_count = len(
                    [(y, lon, lat) for y, lon, lat in available_embeddings if y == year]
                )
                rprint(
                    f"[cyan]{emoji('📊 ')}Tiles shown: {tile_count:,} (year {year})[/cyan]"
                )
            else:
                unique_tiles = len(
                    set((lon, lat) for _, lon, lat in available_embeddings)
                )
                years = sorted(set(y for y, _, _ in available_embeddings))
                rprint(
                    f"[cyan]{emoji('📊 ')}Unique tile locations: {unique_tiles:,}[/cyan]"
                )
                if years:
                    rprint(
                        f"[cyan]{emoji('📅 ')}Years covered: {min(years)}-{max(years)}[/cyan]"
                    )

        # Also generate JSON + HTML globe visualization. In --by-source mode
        # we emit one texture + per-year JSON set per (version, variant) so
        # globe.html can toggle layers.
        rprint("\n[blue]Generating interactive globe visualization...[/blue]")
        try:
            output_dir = ancillary_dir
            globe_html_path = output_dir / "globe.html"

            # Build the list of datasets the globe should know about.
            #   Single-source coverage: just the one (version, variant) selected.
            #   --by-source coverage: every (version, variant) the user asked for.
            from geotessera.registry import (
                _parse_dataset_version,
                published_datasets,
            )

            datasets_to_export = []  # list of dicts: {dataset_version, dataset_variant, dataset_id, color}
            # Stable tab10-ish palette so layer colours match the static PNG.
            palette = [
                (31, 119, 180),  # tab:blue
                (255, 127, 14),  # tab:orange
                (44, 160, 44),  # tab:green
                (214, 39, 40),  # tab:red
                (148, 103, 189),  # tab:purple
                (140, 86, 75),  # tab:brown
            ]
            if by_source:
                # Walk the same combos that the static PNG rendered: the
                # published (version, variant) datasets, narrowed by any
                # explicit --dataset-version/--dataset-variant.
                pubs = published_datasets()
                if version_spec.lower() != "all":
                    _vpath, vnorm = _parse_dataset_version(version_spec)
                    pubs = [p for p in pubs if p[0] == vnorm]
                if variant_spec.lower() != "all":
                    pubs = [p for p in pubs if p[1] == variant_spec]
                combos = [(v, var) for v, var, _d in pubs]
            else:
                combos = [(init_version, init_variant)]

            color_idx = 0
            for vspec, vrnt in combos:
                vpath, vnorm = _parse_dataset_version(vspec)
                dataset_id = f"{vpath}_{vrnt}"
                color = palette[color_idx % len(palette)]
                color_idx += 1
                datasets_to_export.append(
                    {
                        "dataset_version": vspec,
                        "dataset_version_norm": vnorm,
                        "dataset_version_path": vpath,
                        "dataset_variant": vrnt,
                        "dataset_id": dataset_id,
                        "color": color,
                    }
                )

            # Per-dataset GeoTessera instances. Reuse `gt` when its
            # (version, variant) matches a combo so we don't redownload.
            dataset_index = []  # entries that will land in coverage.json
            for ds in datasets_to_export:
                if (
                    ds["dataset_version_path"] == gt.registry._version_path
                    and ds["dataset_variant"] == gt.registry._variant
                ):
                    ds_gt = gt
                else:
                    try:
                        from geotessera import GeoTessera as _GT

                        ds_gt = _GT(
                            dataset_version=ds["dataset_version"],
                            dataset_variant=ds["dataset_variant"],
                            cache_dir=str(cache_dir) if cache_dir else None,
                            registry_dir=str(registry_dir) if registry_dir else None,
                        )
                    except Exception as e:
                        rprint(f"[yellow]Skipping {ds['dataset_id']}: {e}[/yellow]")
                        continue

                ds_id = ds["dataset_id"]
                json_path = output_dir / f"coverage_{ds_id}.json"
                texture_path = output_dir / f"coverage_texture_{ds_id}.png"

                rprint(f"[blue]Exporting coverage for {ds_id}...[/blue]")
                coverage_data = ds_gt.export_coverage_map(
                    output_file=str(json_path), dataset_id=ds_id
                )
                ds_gt.generate_coverage_texture(
                    coverage_data,
                    output_file=str(texture_path),
                    tint_color=ds["color"],
                )

                dataset_index.append(
                    {
                        "id": ds_id,
                        "version": ds["dataset_version_norm"],
                        "version_path": ds["dataset_version_path"],
                        "variant": ds["dataset_variant"],
                        "color": list(ds["color"]),
                        "years": coverage_data["years"],
                        "coverage_json": f"coverage_{ds_id}.json",
                        "texture_png": f"coverage_texture_{ds_id}.png",
                    }
                )

            # Top-level coverage.json indexing every dataset.
            top_index = output_dir / "coverage.json"
            import json as _json

            with open(top_index, "w", encoding="utf-8") as f:
                _json.dump(
                    {"datasets": dataset_index, "schema_version": 2},
                    f,
                    separators=(",", ":"),
                )

            # Generate globe.html
            with open(globe_html_path, "w", encoding="utf-8") as f:
                f.write(_get_globe_html_template())

            rprint(
                f"[green]{emoji('✅ ')}Coverage index: {top_index} "
                f"({len(dataset_index)} dataset(s))[/green]"
            )
            for entry in dataset_index:
                rprint(
                    f"   • {entry['id']}: {len(entry['years'])} year(s), "
                    f"texture {entry['texture_png']}"
                )
            rprint(f"[green]{emoji('✅ ')}Globe viewer: {globe_html_path}[/green]")
            rprint(
                f"[dim]   Open {globe_html_path} in a web browser for interactive visualization[/dim]"
            )

        except Exception as e:
            rprint(
                f"[yellow]Warning: Failed to generate globe visualization: {e}[/yellow]"
            )
            if verbose:
                import traceback

                traceback.print_exc()

    except ImportError:
        rprint("[red]Error: Missing required dependencies[/red]")
        rprint("[yellow]Please install: pip install matplotlib geodatasets[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error generating coverage map: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)
    finally:
        # Clean up temporary country GeoJSON file if created
        if country_geojson_file and (
            not region_file or country_geojson_file != str(region_file)
        ):
            try:
                os.unlink(country_geojson_file)
            except Exception:
                pass  # Ignore cleanup errors

        # Clean up temporary region file if downloaded from URL
        if region_file_temp:
            try:
                os.unlink(region_file_temp)
            except Exception:
                pass  # Ignore cleanup errors


@app.command()
def download(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory (required for actual downloads, optional for --dry-run)",
        ),
    ] = None,
    bbox: Annotated[
        Optional[str],
        typer.Option(
            "--bbox",
            help="Bounding box: 'lon,lat' (single tile) or 'min_lon,min_lat,max_lon,max_lat'",
        ),
    ] = None,
    tile: Annotated[
        Optional[str],
        typer.Option("--tile", help="Single tile by any point within it: 'lon,lat'"),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="GeoJSON/Shapefile to define region (file path or URL)",
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country", help="Country name (e.g., 'United Kingdom', 'UK', 'GB')"
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'tiff' (georeferenced) or 'npy' (raw arrays)",
        ),
    ] = "tiff",
    year: Annotated[int, typer.Option("--year", help="Year of embeddings")] = 2024,
    bands: Annotated[
        Optional[str],
        typer.Option("--bands", help="Comma-separated band indices (default: all 128)"),
    ] = None,
    compress: Annotated[
        str, typer.Option("--compress", help="Compression method (tiff format only)")
    ] = "lzw",
    list_files: Annotated[
        bool, typer.Option("--list-files", help="List all created files with details")
    ] = False,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version",
            help="Tessera dataset version (e.g. v1, v1.1, v2; default v1)",
        ),
    ] = "v1",
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Tessera dataset variant (default: the version's default "
            "variant, e.g. vultr for v1, cambridge for v1.1, 2B-L~beta1 "
            "for v2). Run 'geotessera info' to list available datasets.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Cache directory")
    ] = None,
    registry_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--registry-dir",
            help="Directory containing manifest.parquet and landmasks.parquet files",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Calculate total download size without downloading"
        ),
    ] = False,
):
    """Download embeddings as numpy arrays or GeoTIFF files.

    Supports two output formats:
    - tiff: Georeferenced GeoTIFF files with proper CRS metadata (default)
    - npy: Quantized numpy arrays with separate scales files and landmask TIFFs

    For GeoTIFF format, tiles are organized in the registry structure:
    - global_0.1_degree_representation/{year}/grid_{lon:.2f}_{lat:.2f}/grid_{lon:.2f}_{lat:.2f}_{year}.tiff

    For numpy format, downloads quantized embeddings in the registry structure:
    - global_0.1_degree_representation/{year}/grid_{lon:.2f}_{lat:.2f}/grid_{lon:.2f}_{lat:.2f}.npy
    - global_0.1_degree_representation/{year}/grid_{lon:.2f}_{lat:.2f}/grid_{lon:.2f}_{lat:.2f}_scales.npy
    - global_0.1_degree_tiff_all/grid_{lon:.2f}_{lat:.2f}.tiff (landmask TIFF)

    The NPY format supports resume - if a download is interrupted, running the command
    again will skip files that already exist and only download missing files.

    Note: Band selection (--bands) is only supported for TIFF format. The NPY format
    downloads the full quantized embeddings as they exist in the registry.
    """

    # Validate output parameter
    if not dry_run and output is None:
        rprint("[red]Error: --output/-o is required for actual downloads[/red]")
        rprint(
            "[dim]Use --dry-run to calculate download size without specifying output directory[/dim]"
        )
        raise typer.Exit(1)

    # For dry-run, use a dummy path if not provided (won't be used for actual downloads)
    if output is None:
        output = Path(".")

    # Initialize GeoTessera with embeddings_dir set to output directory
    gt = GeoTessera(
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None,
        embeddings_dir=str(output)
        if not dry_run
        else None,  # Only set for actual downloads
    )

    # Check mutual exclusivity of region options
    region_sources = sum(1 for x in [bbox, tile, region_file, country] if x)
    if region_sources > 1:
        rprint(
            "[red]Error: Cannot specify multiple region options. "
            "Choose one of: --bbox, --tile, --region-file, --country[/red]"
        )
        raise typer.Exit(1)

    # Parse bounding box or tile
    if tile:
        try:
            tile_coords = tuple(map(float, tile.split(",")))
            if len(tile_coords) != 2:
                rprint("[red]Error: --tile must be 'lon,lat'[/red]")
                raise typer.Exit(1)
            lon, lat = tile_coords
            tile_center = tile_from_world(lon, lat)
            bbox_coords = point_to_tile_bbox(lon, lat)
            rprint(
                f"[green]Point ({lon}, {lat}) -> tile "
                f"grid_{tile_center[0]:.2f}_{tile_center[1]:.2f}[/green]"
            )
            rprint(f"[green]Using bounding box:[/green] {format_bbox(bbox_coords)}")
        except ValueError as e:
            rprint(f"[red]Error: Invalid --tile format. Use 'lon,lat': {e}[/red]")
            raise typer.Exit(1)
    elif bbox:
        try:
            bbox_coords = tuple(map(float, bbox.split(",")))
            if len(bbox_coords) == 2:
                # Two coordinates = single tile
                lon, lat = bbox_coords
                tile_center = tile_from_world(lon, lat)
                bbox_coords = point_to_tile_bbox(lon, lat)
                rprint(
                    f"[green]Point ({lon}, {lat}) -> tile "
                    f"grid_{tile_center[0]:.2f}_{tile_center[1]:.2f}[/green]"
                )
                rprint(f"[green]Using bounding box:[/green] {format_bbox(bbox_coords)}")
            elif len(bbox_coords) == 4:
                rprint(f"[green]Using bounding box:[/green] {format_bbox(bbox_coords)}")
            else:
                rprint(
                    "[red]Error: bbox must be 'lon,lat' (single tile) "
                    "or 'min_lon,min_lat,max_lon,max_lat'[/red]"
                )
                raise typer.Exit(1)
        except ValueError:
            rprint(
                "[red]Error: Invalid bbox format. Use: 'lon,lat' or "
                "'min_lon,min_lat,max_lon,max_lat'[/red]"
            )
            raise typer.Exit(1)
    elif region_file:
        try:
            # Check if region_file is a URL
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_path = download_region_file(region_file)
                region_file_temp = region_file_path  # Track for cleanup
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(
                        f"[red]Error: Region file {region_file} does not exist[/red]"
                    )
                    raise typer.Exit(1)
                region_file_path = str(region_path)
                region_file_temp = None

            bbox_coords = calculate_bbox_from_file(region_file_path)
            rprint(
                f"[green]Calculated bbox from {region_file}:[/green] {format_bbox(bbox_coords)}"
            )
        except Exception as e:
            rprint(f"[red]Error reading region file: {e}[/red]")
            rprint("Supported formats: GeoJSON, Shapefile, etc.")
            # Clean up temp file if we created one
            if "region_file_temp" in locals() and region_file_temp:
                try:
                    import os

                    os.unlink(region_file_temp)
                except Exception:
                    pass
            raise typer.Exit(1)
    elif country:
        # Create progress bar for country data download
        with create_progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
        ) as progress:
            country_task = progress.add_task(
                f"{emoji('🌍 ')}Loading country data...",
                total=100,
                status="Checking cache...",
            )

            def country_progress_callback(current: int, total: int, status: str = None):
                progress.update(
                    country_task,
                    completed=current,
                    total=total,
                    status=status or "Processing...",
                )

            try:
                bbox_coords = get_country_bbox(
                    country, progress_callback=country_progress_callback
                )
                progress.update(country_task, completed=100, status="Complete")
            except ValueError as e:
                rprint(f"[red]Error: {e}[/red]")
                rprint(
                    "[blue]Check the country name spelling (uses Natural Earth admin-0 names)[/blue]"
                )
                raise typer.Exit(1)
            except Exception as e:
                rprint(f"[red]Error fetching country data: {e}[/red]")
                raise typer.Exit(1)

        # Print country info after progress bar completes
        rprint(f"[green]Using country '{country}':[/green] {format_bbox(bbox_coords)}")
    else:
        rprint(
            "[red]Error: Must specify either --bbox, --tile, --region-file, or --country[/red]"
        )
        rprint("Examples:")
        rprint("  --tile '0.17,52.23'          # Single tile containing point")
        rprint(
            "  --bbox '0.17,52.23'          # Same as above (2 coords = single tile)"
        )
        rprint("  --bbox '-0.2,51.4,0.1,51.6'  # London area (4 coords = region)")
        rprint("  --region-file london.geojson  # From GeoJSON file")
        rprint("  --country 'United Kingdom'    # Country by name")
        raise typer.Exit(1)

    # Parse bands
    bands_list = None
    if bands:
        try:
            bands_list = list(map(int, bands.split(",")))
            rprint(
                f"[blue]Exporting {len(bands_list)} selected bands:[/blue] {bands_list}"
            )
        except ValueError:
            rprint("[red]Error: bands must be comma-separated integers (0-127)[/red]")
            rprint("Example: --bands '0,1,2' for first 3 bands")
            raise typer.Exit(1)
    else:
        rprint("[blue]Exporting all 128 bands[/blue]")

    # Validate format
    if format == "zarr":
        rprint(
            "[red]Error: Zarr format is no longer supported in the download command. "
            "Use geotessera-registry zarr-init/zarr-fill to build zarr stores, "
            "and GeoTesseraZarr to read them.[/red]"
        )
        raise typer.Exit(1)
    if format not in ["tiff", "npy"]:
        rprint(f"[red]Error: Invalid format '{format}'. Must be 'tiff' or 'npy'[/red]")
        raise typer.Exit(1)

    # Display export info
    info_table = create_table(show_header=False, box=None)
    info_table.add_row("Format:", format.upper())
    info_table.add_row("Year:", str(year))
    # Only show output directory when not doing dry-run
    if not dry_run:
        info_table.add_row("Output directory:", str(output))
    if format == "tiff":
        info_table.add_row("Compression:", compress)
    info_table.add_row("Dataset version:", dataset_version)

    rprint(
        create_panel(
            info_table,
            title=f"[bold]GeoTessera v{__version__} - Region of Interest Download[/bold]",
            border_style="blue",
        )
    )

    try:
        # Load tiles for the region first (before Progress context)
        tiles_to_fetch = gt.registry.load_blocks_for_region(
            bounds=bbox_coords, year=year
        )

        if not tiles_to_fetch:
            rprint(
                f"[yellow]{emoji('⚠️  ')}No tiles found in the specified region.[/yellow]"
            )
            rprint("Try expanding your bounding box or checking data availability.")
            return

        # Handle dry-run mode: calculate and display size information (no progress bar)
        if dry_run:
            try:
                total_bytes, total_files, _ = (
                    gt.registry.calculate_download_requirements(
                        tiles_to_fetch, output, format, check_existing=False
                    )
                )
            except ValueError as e:
                rprint(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)

            # Display results
            result_table = create_table(show_header=False, box=None, padding=(0, 2))
            result_table.add_row("Files to download:", f"[cyan]{total_files:,}[/cyan]")
            result_table.add_row(
                "Total download size:", f"[cyan]{format_bytes(total_bytes)}[/cyan]"
            )
            result_table.add_row(
                "Tiles in region:", f"[cyan]{len(tiles_to_fetch):,}[/cyan]"
            )
            result_table.add_row("Year:", f"[cyan]{year}[/cyan]")
            result_table.add_row("Format:", f"[cyan]{format.upper()}[/cyan]")

            rprint(
                create_panel(
                    result_table,
                    title="[bold]Dry Run Results[/bold]",
                    border_style="green",
                )
            )

            if format == "tiff":
                rprint("[dim]Note: TIFF sizes are estimates (4x quantized size)[/dim]")

            rprint("\n[dim]Run without --dry-run to download these files[/dim]")
            return

        # Export tiles with progress tracking
        with create_progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"{emoji('📥 ')}Downloading tiles...", total=100, status="Starting..."
            )

            match format:
                case "tiff":
                    skipped_files = 0
                    filtered_tiles = []
                    for tile_year, tile_lon, tile_lat in tiles_to_fetch:
                        geotiff_path = (
                            output
                            / EMBEDDINGS_DIR_NAME
                            / tile_to_geotiff_path(tile_lon, tile_lat, tile_year)
                        )
                        if geotiff_path.exists():
                            skipped_files += 1
                        else:
                            filtered_tiles.append((tile_year, tile_lon, tile_lat))

                    total_tiles = len(filtered_tiles)
                    progress.update(task, total=max(total_tiles, 1), completed=0)

                    if not filtered_tiles:
                        progress.update(task, completed=1, status="Complete")
                        files = []
                    else:
                        files = gt.export_embedding_geotiffs(
                            filtered_tiles,
                            output_dir=output,
                            bands=bands_list,
                            compress=compress,
                            progress_callback=create_progress_callback(progress, task),
                        )

                    if not filtered_tiles:
                        rprint(
                            f"\n[green]{emoji('✅ ')}SUCCESS: All {len(tiles_to_fetch)} GeoTIFF files already exist[/green]"
                        )
                    else:
                        rprint(
                            f"\n[green]{emoji('✅ ')}SUCCESS: Exported {len(files)} GeoTIFF files"
                            + (
                                f" ({skipped_files} skipped)"
                                if skipped_files > 0
                                else ""
                            )
                            + "[/green]"
                        )
                    rprint(
                        "   Each file preserves its native UTM projection from landmask tiles"
                    )
                    rprint("   Files can be individually inspected and processed")

                case "npy":
                    # Export as quantized numpy arrays with scales

                    # Create output directory structure
                    output.mkdir(parents=True, exist_ok=True)

                    files = []
                    downloaded_files = 0
                    skipped_files = 0

                    # Calculate total download size from registry using Registry method
                    progress.update(
                        task,
                        completed=0,
                        total=100,
                        status="Calculating download size...",
                    )

                    try:
                        total_bytes, _, file_sizes = (
                            gt.registry.calculate_download_requirements(
                                tiles_to_fetch, output, format
                            )
                        )
                    except ValueError as e:
                        rprint(f"[red]Error: {e}[/red]")
                        raise typer.Exit(1)

                    # Track cumulative bytes downloaded
                    bytes_downloaded = 0
                    total_size_str = format_bytes(total_bytes)

                    # Create a progress callback factory that updates overall byte progress
                    def create_download_callback(file_key):
                        """Create a callback for download progress updates with overall byte tracking."""

                        def callback(current, total, status):
                            nonlocal bytes_downloaded
                            # Update total bytes progress
                            file_bytes_so_far = current
                            progress.update(
                                task,
                                completed=bytes_downloaded + file_bytes_so_far,
                                total=total_bytes,
                                status=status,
                            )

                        return callback

                    def mark_file_complete(file_key):
                        """Mark a file as complete and update bytes_downloaded."""
                        nonlocal bytes_downloaded
                        if file_key in file_sizes:
                            bytes_downloaded += file_sizes[file_key]

                    # Reset progress bar for download
                    progress.update(
                        task,
                        completed=0,
                        total=total_bytes,
                        status=f"Downloading {total_size_str}...",
                    )

                    # Process each tile
                    for idx, (tile_year, tile_lon, tile_lat) in enumerate(
                        tiles_to_fetch
                    ):
                        # Set up final paths with structure mirroring remote
                        embedding_rel, scales_rel = tile_to_embedding_paths(
                            tile_lon, tile_lat, tile_year
                        )
                        embedding_final = output / EMBEDDINGS_DIR_NAME / embedding_rel
                        scales_final = output / EMBEDDINGS_DIR_NAME / scales_rel
                        landmask_final = (
                            output
                            / LANDMASKS_DIR_NAME
                            / tile_to_landmask_filename(tile_lon, tile_lat)
                        )

                        # Create cache keys for tracking
                        embedding_key = f"embedding_{tile_year}_{tile_lon}_{tile_lat}"
                        scales_key = f"scales_{tile_year}_{tile_lon}_{tile_lat}"
                        landmask_key = f"landmask_{tile_lon}_{tile_lat}"

                        # Download embedding file (fetch() saves directly to embeddings_dir)
                        if embedding_final.exists():
                            skipped_files += 1
                        else:
                            try:
                                gt.registry.fetch(
                                    year=tile_year,
                                    lon=tile_lon,
                                    lat=tile_lat,
                                    is_scales=False,
                                    progress_callback=create_download_callback(
                                        embedding_key
                                    ),
                                    refresh=True,
                                )
                                mark_file_complete(embedding_key)
                                files.append(str(embedding_final))
                                downloaded_files += 1
                            except Exception as e:
                                rprint(
                                    f"[yellow]Warning: Failed to download embedding for ({tile_lon}, {tile_lat}, {tile_year}): {e}[/yellow]"
                                )
                                continue

                        # Download scales file (fetch() saves directly to embeddings_dir)
                        if scales_final.exists():
                            skipped_files += 1
                        else:
                            try:
                                gt.registry.fetch(
                                    year=tile_year,
                                    lon=tile_lon,
                                    lat=tile_lat,
                                    is_scales=True,
                                    progress_callback=create_download_callback(
                                        scales_key
                                    ),
                                    refresh=True,
                                )
                                mark_file_complete(scales_key)
                                files.append(str(scales_final))
                                downloaded_files += 1
                            except Exception as e:
                                rprint(
                                    f"[yellow]Warning: Failed to download scales for ({tile_lon}, {tile_lat}, {tile_year}): {e}[/yellow]"
                                )

                        # Download landmask file (fetch_landmask() saves directly to embeddings_dir)
                        if landmask_final.exists():
                            skipped_files += 1
                        else:
                            try:
                                gt.registry.fetch_landmask(
                                    lon=tile_lon,
                                    lat=tile_lat,
                                    progress_callback=create_download_callback(
                                        landmask_key
                                    ),
                                    refresh=True,
                                )
                                mark_file_complete(landmask_key)
                                files.append(str(landmask_final))
                                downloaded_files += 1
                            except Exception as e:
                                rprint(
                                    f"[yellow]Warning: Failed to download landmask for ({tile_lon}, {tile_lat}): {e}[/yellow]"
                                )

                    # Final progress update (after all tiles processed)
                    progress.update(task, completed=total_bytes, status="Complete")

                    downloaded_size_str = (
                        format_bytes(bytes_downloaded) if bytes_downloaded > 0 else "0B"
                    )
                    rprint(
                        f"\n[green]{emoji('✅ ')}SUCCESS: Downloaded {len(tiles_to_fetch)} tiles ({downloaded_files} files, {downloaded_size_str})[/green]"
                    )
                    if skipped_files > 0:
                        rprint(
                            f"   Skipped {skipped_files} existing files (resume capability)"
                        )
                    rprint("   Format: Quantized embeddings with separate scales files")
                    rprint(
                        "   Structure: global_0.1_degree_representation/{year}/grid_{lon}_{lat}/grid_{lon}_{lat}.npy"
                    )
                    rprint(
                        "             global_0.1_degree_tiff_all/grid_{lon}_{lat}.tiff"
                    )
                    if bands_list:
                        rprint(
                            "   [yellow]Note: Band selection not supported in NPY format (use TIFF format instead)[/yellow]"
                        )

        # Record provenance in a sidecar JSON. Local layout uses the bare
        # global_0.1_degree_representation/ regardless of variant, so this
        # file is the source of truth for which (version, variant) produced
        # the tiles in this directory.
        if not dry_run:
            from geotessera.registry import write_tessera_metadata

            try:
                sidecar = write_tessera_metadata(
                    output,
                    dataset_version=dataset_version,
                    dataset_variant=gt.dataset_variant,
                    extra={
                        "format": format,
                        "year": year,
                        "tile_count": len(tiles_to_fetch),
                    },
                )
                rprint(f"   Metadata written to: {sidecar}")
            except Exception as e:
                rprint(
                    f"[yellow]   Warning: could not write sidecar metadata: {e}[/yellow]"
                )

        if verbose or list_files:
            rprint(f"\n[blue]{emoji('📁 ')}Created files:[/blue]")
            file_table = create_table(show_header=True, header_style="bold blue")
            file_table.add_column("#", style="dim", width=3)
            file_table.add_column("Filename")
            file_table.add_column("Size", justify="right")

            for i, f in enumerate(files, 1):
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_table.add_row(str(i), file_path.name, f"{file_size:,} bytes")

            console.print(file_table)
        elif len(files) > 0:
            rprint(
                f"\n[blue]{emoji('📁 ')}Sample files (use --verbose or --list-files to see all):[/blue]"
            )
            for f in files[:3]:
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                rprint(f"     {file_path.name} ({file_size:,} bytes)")
            if len(files) > 3:
                rprint(f"     ... and {len(files) - 3} more files")

        # Show spatial information
        rprint(f"\n[blue]{emoji('🗺️  ')}Spatial Information:[/blue]")
        if verbose:
            try:
                match format:
                    case "tiff":
                        import rasterio

                        with rasterio.open(files[0]) as src:
                            rprint(f"   CRS: {src.crs}")
                            rprint(f"   Transform: {src.transform}")
                            rprint(f"   Dimensions: {src.width} x {src.height} pixels")
                            rprint(f"   Data type: {src.dtypes[0]}")
            except Exception:
                pass

        rprint(f"   Output directory: {Path(output).resolve()}")

        tips_table = create_table(show_header=False, box=None)
        match format:
            case "tiff":
                tips_table.add_row(
                    "Inspect individual tiles with QGIS, GDAL, or rasterio"
                )
                tips_table.add_row(
                    "Use 'gdalinfo <filename>' to see projection details"
                )
                tips_table.add_row("Process tiles individually or in groups as needed")
                tips_table.add_row("Create PCA visualization:")
                tips_table.add_row(
                    f"  [cyan]geotessera visualize {output} pca_mosaic.tif[/cyan]"
                )

        rprint(
            create_panel(
                tips_table, title="[bold] Next steps[/bold]", border_style="green"
            )
        )

    except Exception as e:
        rprint(f"\n[red]{emoji('❌ ')}Error: {e}[/red]")
        if verbose:
            rprint("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)
    finally:
        # Clean up temporary region file if downloaded from URL
        if "region_file_temp" in locals() and region_file_temp:
            try:
                import os

                os.unlink(region_file_temp)
            except Exception:
                pass  # Ignore cleanup errors


@app.command()
def visualize(
    input_path: Annotated[
        Path, typer.Argument(help="Input GeoTIFF or NPY file directory")
    ],
    output_file: Annotated[Path, typer.Argument(help="Output PCA mosaic file (.tif)")],
    target_crs: Annotated[
        str, typer.Option("--crs", help="Target CRS for reprojection")
    ] = "EPSG:3857",
    n_components: Annotated[
        int,
        typer.Option(
            "--n-components",
            help="Number of PCA components. Only first 3 used for RGB visualization - increase for analysis/research.",
        ),
    ] = 3,
    balance_method: Annotated[
        str,
        typer.Option(
            "--balance",
            help="RGB balance method: histogram (default), percentile, or adaptive",
        ),
    ] = "histogram",
    percentile_low: Annotated[
        float,
        typer.Option(
            "--percentile-low", help="Lower percentile for percentile balance method"
        ),
    ] = 2.0,
    percentile_high: Annotated[
        float,
        typer.Option(
            "--percentile-high", help="Upper percentile for percentile balance method"
        ),
    ] = 98.0,
):
    """Create PCA visualization from GeoTIFF or NPY format embeddings.

    This command combines all embedding data across tiles, applies a single PCA
    transformation to the combined dataset, then creates a unified RGB mosaic.
    This ensures consistent principal components across the entire region,
    eliminating tiling artifacts.

    Supports two input formats:
    - GeoTIFF format: Directory containing *.tif/*.tiff files
    - NPY format: Directory with global_0.1_degree_representation/{year}/grid_{lon}_{lat}/*.npy structure

    The first 3 principal components are mapped to RGB channels for visualization.
    Additional components can be computed for research/analysis purposes.

    Examples:
        # Create PCA visualization from GeoTIFF tiles
        geotessera visualize tiles/ pca_mosaic.tif

        # Create PCA visualization from NPY format tiles
        geotessera visualize npy_tiles/ pca_mosaic.tif

        # Use histogram equalization for maximum contrast
        geotessera visualize tiles/ pca_balanced.tif --balance histogram

        # Use adaptive scaling based on variance
        geotessera visualize tiles/ pca_adaptive.tif --balance adaptive

        # Custom percentile range for outlier-robust scaling
        geotessera visualize tiles/ pca_custom.tif --percentile-low 5 --percentile-high 95

        # Use custom projection
        geotessera visualize tiles/ pca_mosaic.tif --crs EPSG:4326

        # PCA for research - compute more components for analysis
        # (still only uses first 3 for RGB, but saves variance info)
        geotessera visualize tiles/ pca_research.tif --n-components 10

        # Then create web visualization
        geotessera webmap pca_mosaic.tif --serve
    """

    # Validate output file extension
    if output_file.suffix.lower() not in [".tif", ".tiff"]:
        rprint("[red]Error: Output file must have .tif or .tiff extension[/red]")
        raise typer.Exit(1)

    # Validate n_components
    if n_components < 1:
        rprint("[red]Error: Number of components must be at least 1[/red]")
        raise typer.Exit(1)
    if n_components < 3:
        rprint(
            f"[yellow]Warning: Using {n_components} component(s). RGB visualization works best with 3+ components[/yellow]"
        )

    # Validate balance_method
    if balance_method not in ["percentile", "histogram", "adaptive"]:
        rprint(
            f"[red]Error: Invalid balance method '{balance_method}'. Must be 'percentile', 'histogram', or 'adaptive'[/red]"
        )
        raise typer.Exit(1)

    # Validate percentile ranges
    if balance_method == "percentile":
        if not (0 <= percentile_low < percentile_high <= 100):
            rprint(
                f"[red]Error: Invalid percentile range [{percentile_low}, {percentile_high}]. Must be 0 <= low < high <= 100[/red]"
            )
            raise typer.Exit(1)

    # Discover tiles (handles both GeoTIFF and NPY formats automatically)
    from geotessera.tiles import discover_tiles

    tiles = discover_tiles(input_path)

    if not tiles:
        # Force line break before path for deterministic output regardless of terminal width
        rprint(f"[red]No tiles found in\n{input_path}[/red]")
        rprint("[yellow]Expected either:[/yellow]")
        rprint("  - GeoTIFF files: *.tif/*.tiff in the directory")
        rprint(
            "  - NPY format: global_0.1_degree_representation/{year}/grid_{lon}_{lat}/*.npy structure"
        )
        raise typer.Exit(1)

    rprint(f"[blue]Found {len(tiles)} tiles ({tiles[0]._format} format)[/blue]")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with create_progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Creating PCA mosaic ({n_components} components)...",
            total=5,
            status="Starting...",
        )

        try:
            # Create a progress callback that maps to our 5-step progress
            def visualization_progress_callback(
                current: float, total: float, status: str = None
            ):
                progress.update(
                    task,
                    completed=current,
                    total=total,
                    status=status or "Processing...",
                )

            # Convert tiles to dict format for create_pca_mosaic
            tiles_data = [tile.to_dict() for tile in tiles]

            # PCA MODE: Use clean visualization function
            create_pca_mosaic(
                tiles_data=tiles_data,
                output_path=output_file,
                n_components=n_components,
                target_crs=target_crs,
                progress_callback=visualization_progress_callback,
                balance_method=balance_method,
                percentile_range=(percentile_low, percentile_high),
            )

            progress.update(task, completed=5, total=5, status="Complete")

        except Exception as e:
            rprint(f"[red]Error creating PCA visualization: {e}[/red]")
            raise typer.Exit(1)

    # Success output after progress bar completes
    # Force line break before filename to avoid wrapping issues in tests
    rprint(f"[green]Created PCA mosaic:\n{output_file}[/green]")
    rprint(f"[blue]Components: {n_components} | CRS: {target_crs}[/blue]")
    rprint("[blue]Next step: Create web visualization with:[/blue]")
    rprint(f"[cyan]  geotessera webmap {output_file} --serve[/cyan]")


@app.command()
def webmap(
    rgb_mosaic: Annotated[Path, typer.Argument(help="3-band RGB mosaic GeoTIFF file")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory")
    ] = None,
    min_zoom: Annotated[
        int, typer.Option("--min-zoom", help="Min zoom for web tiles")
    ] = 8,
    max_zoom: Annotated[
        int, typer.Option("--max-zoom", help="Max zoom for web tiles")
    ] = 15,
    initial_zoom: Annotated[
        int, typer.Option("--initial-zoom", help="Initial zoom level")
    ] = 10,
    force_regenerate: Annotated[
        bool,
        typer.Option(
            "--force/--no-force", help="Force regeneration of tiles even if they exist"
        ),
    ] = False,
    serve_immediately: Annotated[
        bool, typer.Option("--serve/--no-serve", help="Start web server immediately")
    ] = False,
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port for web server")
    ] = 8000,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="GeoJSON/Shapefile boundary to overlay (file path or URL)",
        ),
    ] = None,
    use_gdal_raster: Annotated[
        bool,
        typer.Option(
            "--use-gdal-raster/--use-gdal2tiles",
            help="Use newer gdal raster tile (faster but less stable) vs gdal2tiles (default, stable)",
        ),
    ] = False,
):
    """Create web tiles and viewer from a 3-band RGB mosaic.

    This command takes an RGB GeoTIFF mosaic, reprojects it if needed for web viewing,
    generates web tiles, creates an HTML viewer, and optionally starts a web server.

    Example workflow:
        1. geotessera download --bbox lon1,lat1,lon2,lat2 tiles/
        2. geotessera visualize tiles/ --type rgb --output mosaics/
        3. geotessera webmap mosaics/rgb_mosaic.tif --output webmap/ --serve
    """
    if not rgb_mosaic.exists():
        rprint(f"[red]Error: Mosaic file {rgb_mosaic} does not exist[/red]")
        raise typer.Exit(1)

    if rgb_mosaic.suffix.lower() not in [".tif", ".tiff"]:
        rprint("[red]Error: Input must be a GeoTIFF file (.tif/.tiff)[/red]")
        raise typer.Exit(1)

    # Handle region file URL download
    region_file_temp = None
    region_file_path = None
    if region_file:
        try:
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_temp = download_region_file(region_file)
                region_file_path = region_file_temp
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(
                        f"[red]Error: Region file {region_file} does not exist[/red]"
                    )
                    raise typer.Exit(1)
                region_file_path = str(region_path)
        except Exception as e:
            rprint(f"[red]Error processing region file: {e}[/red]")
            # Clean up temp file if we created one
            if region_file_temp:
                try:
                    import os

                    os.unlink(region_file_temp)
                except Exception:
                    pass
            raise typer.Exit(1)

    # Default output directory
    if output is None:
        output = Path(f"{rgb_mosaic.stem}_webmap")

    output.mkdir(parents=True, exist_ok=True)

    with create_progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        TimeRemainingColumn(),
    ) as progress:
        # Step 1: Prepare mosaic for web (reproject if needed)
        web_mosaic_path = output / "web_ready_mosaic.tif"

        task1 = progress.add_task(
            "Preparing mosaic for web...", total=100, status="Starting..."
        )

        try:
            actual_mosaic_path = prepare_mosaic_for_web(
                input_mosaic=str(rgb_mosaic),
                output_path=str(web_mosaic_path),
                target_crs="EPSG:3857",
                progress_callback=create_progress_callback(progress, task1),
            )

            # If no reprojection was needed, use original file
            if actual_mosaic_path == str(rgb_mosaic):
                mosaic_status = "Using original mosaic (already in correct CRS)"
            else:
                # Force line break before filename to avoid wrapping issues
                mosaic_status = f"Created web-ready mosaic:\n{web_mosaic_path}"

        except Exception as e:
            rprint(f"[red]Error preparing mosaic: {e}[/red]")
            raise typer.Exit(1)

        # Step 2: Generate web tiles
        tiles_dir = output / "tiles"

        # Check if we should regenerate tiles
        if force_regenerate and tiles_dir.exists():
            import shutil

            shutil.rmtree(tiles_dir)
            tiles_regenerated = True
        else:
            tiles_regenerated = False

        tiles_force_hint = None

        if not tiles_dir.exists() or not any(tiles_dir.iterdir()):
            task2 = progress.add_task(
                "Generating web tiles...", total=100, status="Starting..."
            )

            try:
                result_dir = geotiff_to_web_tiles(
                    geotiff_path=actual_mosaic_path,
                    output_dir=str(tiles_dir),
                    zoom_levels=(min_zoom, max_zoom),
                    use_gdal_raster=use_gdal_raster,
                )
                progress.update(task2, completed=100)
                # Force line break before filename to avoid wrapping issues
                tiles_status = f"Created web tiles in:\n{result_dir}"

            except Exception as e:
                rprint(f"[red]Error generating web tiles: {e}[/red]")
                raise typer.Exit(1)
        else:
            tiles_status = f"Using existing tiles in: {tiles_dir}"
            tiles_force_hint = "Use --force to regenerate tiles"

        # Step 3: Create HTML viewer
        html_path = output / "viewer.html"

        task3 = progress.add_task(
            "Creating web viewer...", total=100, status="Starting..."
        )

        try:
            # Get mosaic bounds for centering
            import rasterio

            with rasterio.open(actual_mosaic_path) as src:
                bounds = src.bounds
                # Transform bounds to lat/lon if needed
                if src.crs != "EPSG:4326":
                    from rasterio.warp import transform_bounds

                    lon_min, lat_min, lon_max, lat_max = transform_bounds(
                        src.crs,
                        "EPSG:4326",
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )
                else:
                    lon_min, lat_min, lon_max, lat_max = (
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )

                center_lat = (lat_min + lat_max) / 2
                center_lon = (lon_min + lon_max) / 2

            create_simple_web_viewer(
                tiles_dir=str(tiles_dir),
                output_html=str(html_path),
                center_lon=center_lon,
                center_lat=center_lat,
                zoom=initial_zoom,
                title=f"GeoTessera v{__version__} - {rgb_mosaic.name}",
                region_file=region_file_path if region_file_path else None,
            )

            progress.update(task3, completed=100)
            # Force line break before filename to avoid wrapping issues
            viewer_status = f"Created web viewer:\n{html_path}"

        except Exception as e:
            rprint(f"[red]Error creating web viewer: {e}[/red]")
            raise typer.Exit(1)

    # Summary
    # Force line break before filename to avoid wrapping issues
    rprint(f"\n[green]{emoji('✅ ')}Web visualization ready in:\n{output}[/green]")

    # Print status messages from the progress context
    rprint(f"[green]{mosaic_status}[/green]")

    if tiles_regenerated:
        rprint("[yellow]Removed existing tiles directory for regeneration[/yellow]")

    rprint(f"[green]{tiles_status}[/green]")
    if tiles_force_hint:
        rprint(f"[blue]{tiles_force_hint}[/blue]")

    rprint(f"[green]{viewer_status}[/green]")

    if serve_immediately:
        rprint("[blue]Starting web server...[/blue]")
        # Call the serve function directly
        try:
            serve(
                directory=output, port=port, open_browser=True, html_file="viewer.html"
            )
        except KeyboardInterrupt:
            rprint("\n[green]Web server stopped.[/green]")
        except Exception as e:
            rprint(f"[yellow]Could not start server automatically: {e}[/yellow]")
            rprint("[blue]To view the map, start a web server manually:[/blue]")
            rprint(f"[cyan]  geotessera serve {output} --port {port}[/cyan]")
    else:
        rprint("[blue]To view the map, start a web server:[/blue]")
        rprint(f"[cyan]  geotessera serve {output} --port {port}[/cyan]")

    # Clean up temporary region file if downloaded from URL
    if region_file_temp:
        try:
            import os

            os.unlink(region_file_temp)
        except Exception:
            pass  # Ignore cleanup errors


@app.command()
def serve(
    directory: Annotated[
        Path, typer.Argument(help="Directory containing web visualization files")
    ],
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port number for web server")
    ] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Automatically open browser")
    ] = True,
    html_file: Annotated[
        Optional[str],
        typer.Option(
            "--html", help="Specific HTML file to serve (relative to directory)"
        ),
    ] = None,
):
    """Start a web server to serve visualization files.

    This is needed for leaflet-based web visualizations to work properly
    since they require HTTP access to load tiles and other resources.
    """
    if not directory.exists():
        rprint(f"[red]Error: Directory {directory} does not exist[/red]")
        raise typer.Exit(1)

    if not directory.is_dir():
        rprint(f"[red]Error: {directory} is not a directory[/red]")
        raise typer.Exit(1)

    # Change to the directory to serve
    original_dir = Path.cwd()

    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Only log errors, not every request. Some SimpleHTTPRequestHandler
            # log paths call this with fewer args, so guard the index access.
            if len(args) > 1 and args[1] != "200":
                super().log_message(format, *args)

    try:
        # Find available port
        while True:
            try:
                with socketserver.TCPServer(("", port), QuietHTTPRequestHandler):
                    break
            except OSError:
                port += 1
                if port > 9000:
                    rprint("[red]Error: Could not find available port[/red]")
                    raise typer.Exit(1)

        rprint(f"[green]Starting web server on port {port}[/green]")
        rprint(f"[blue]Serving directory: {directory.absolute()}[/blue]")

        # Debug: Show directory contents
        try:
            contents = list(directory.iterdir())
            rprint(
                f"[yellow]Directory contains: {[p.name for p in contents[:10]]}{'...' if len(contents) > 10 else ''}[/yellow]"
            )
        except Exception as e:
            rprint(f"[yellow]Could not list directory contents: {e}[/yellow]")

        # Determine what to open in browser
        if html_file:
            html_path = directory / html_file
            if not html_path.exists():
                rprint(f"[yellow]Warning: HTML file {html_file} not found[/yellow]")
                browser_url = f"http://localhost:{port}/"
            else:
                browser_url = f"http://localhost:{port}/{html_file}"
        else:
            # Look for common HTML files
            common_names = ["index.html", "viewer.html", "map.html", "coverage.html"]
            found_html = None
            for name in common_names:
                if (directory / name).exists():
                    found_html = name
                    break

            if found_html:
                browser_url = f"http://localhost:{port}/{found_html}"
                rprint(f"[blue]Found HTML file: {found_html}[/blue]")
            else:
                browser_url = f"http://localhost:{port}/"

        # Start server in background thread
        def start_server():
            import os

            os.chdir(directory)
            try:
                with socketserver.TCPServer(
                    ("", port), QuietHTTPRequestHandler
                ) as httpd:
                    httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                os.chdir(original_dir)

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Give server a moment to start
        time.sleep(0.5)

        rprint(
            f"[green]{emoji('✅ ')}Web server running at: http://localhost:{port}/[/green]"
        )

        if open_browser:
            rprint(f"[blue]Opening browser: {browser_url}[/blue]")
            webbrowser.open(browser_url)
        else:
            rprint(f"[blue]Open in browser: {browser_url}[/blue]")

        rprint("\n[yellow]Press Ctrl+C to stop the server[/yellow]")

        try:
            # Keep main thread alive
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            rprint("\n[green]Stopping web server...[/green]")
            raise typer.Exit(0)

    except Exception as e:
        rprint(f"[red]Error starting web server: {e}[/red]")
        raise typer.Exit(1)


def _get_globe_html_template() -> str:
    """Return the bundled globe.html template."""
    return (
        importlib.resources.files("geotessera")
        .joinpath("templates/globe.html")
        .read_text(encoding="utf-8")
    )


@app.command()
def version():
    """Print the geotessera library version."""
    from geotessera import __version__

    print(__version__)


def main():
    """Main CLI entry point."""
    # Configure logging with rich handler
    # Disable rich formatting in dumb terminals (use Rich Console's built-in detection)
    use_rich = console.is_terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True, show_time=False, show_path=False, console=console
            )
        ]
        if use_rich
        else [logging.StreamHandler()],
    )

    # Optionally reduce logging level for specific noisy libraries
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)

    app()


if __name__ == "__main__":
    main()
