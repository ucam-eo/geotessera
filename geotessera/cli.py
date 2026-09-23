"""Simplified GeoTessera command-line interface.

Focused on downloading tiles and creating visualizations from the generated GeoTIFFs.
"""

# Will configure logging after imports

import importlib.resources
import os
import webbrowser
import http.server
import urllib.parse
import logging
from enum import StrEnum
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
from .visualization import (
    create_pca_mosaic,
)
from .web import (
    geotiff_to_web_tiles,
    create_simple_web_viewer,
    prepare_mosaic_for_web,
)
from ._terminal import console, emoji


class DownloadSource(StrEnum):
    auto = "auto"
    zarr = "zarr"
    tiles = "tiles"


class BalanceMethod(StrEnum):
    histogram = "histogram"
    percentile = "percentile"
    adaptive = "adaptive"


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
    help=f"GeoTessera v{__version__}: Read, export, and display Tessera embeddings.",
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
            "--tiles", help="Inspect a local GeoTIFF or NPY file or directory."
        ),
    ] = None,
    geotiffs: Annotated[
        Optional[Path],
        typer.Option(
            "--geotiffs",
            help="Use the deprecated alias for --tiles.",
        ),
    ] = None,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version",
            help="Select the dataset version. Run geotessera info to list datasets.",
        ),
    ] = "v1.1",
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Select the dataset variant. If omitted, use the version's default in the chosen format. Run geotessera info to list them.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print additional details.")
    ] = False,
):
    """Show library information and available datasets.

    Use --tiles to inspect local GeoTIFF or NPY files.
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

        rprint(analysis_table)

        bounds = coverage["bounds"]

        bounds_table = create_table(show_header=False, box=None)
        bounds_table.add_row(
            "Longitude:", f"{bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}"
        )
        bounds_table.add_row(
            "Latitude:", f"{bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}"
        )

        rprint(bounds_table)

        bands_table = create_table(show_header=True, header_style="bold blue")
        bands_table.add_column("Band Count")
        bands_table.add_column("Files", justify="right")

        for bands_count, count in coverage["band_counts"].items():
            bands_table.add_row(f"{bands_count} bands", str(count))

        rprint(bands_table)

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

            rprint(tiles_table)

    else:
        from geotessera.registry import icechunk_dataset

        rprint(f"geotessera {__version__}")
        stores = _datasets_table()
        icechunk = icechunk_dataset(dataset_version, dataset_variant)
        if icechunk is not None:
            _icechunk_info(icechunk[0], verbose, dataset_version, dataset_variant)
        else:
            gt = _tiles_client(
                dataset_version=dataset_version, dataset_variant=dataset_variant
            )
            tiles_per_year = gt.registry.get_tile_counts_by_year()
            years = sorted(tiles_per_year)
            table = create_table(show_header=False, box=None)
            table.add_row(
                "Selected:", f"{gt.registry.version} {gt.dataset_variant} (NPY)"
            )
            table.add_row("Years:", f"{years[0]}-{years[-1]}" if years else "-")
            table.add_row("Tiles:", f"{sum(tiles_per_year.values()):,}")
            if verbose:
                for year in years:
                    table.add_row(f"  {year}:", f"{tiles_per_year[year]:,}")
            table.add_row("Landmasks:", f"{gt.registry.get_landmask_count():,}")
            rprint(table)
        rprint(stores)


def _tiles_client(**kwargs) -> GeoTessera:
    """A tile client, exiting with the message for an unavailable dataset."""
    try:
        return GeoTessera(**kwargs)
    except ValueError as exc:
        rprint(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


def _icechunk_info(url: str, verbose: bool, version: str, variant) -> None:
    """Describe an Icechunk store from its root and group attributes."""
    from .icechunk import IcechunkStore

    store = IcechunkStore(url)
    attrs = dict(store.root.attrs)
    years = store.years
    incomplete = store.incomplete_years()
    table = create_table(show_header=False, box=None)
    from geotessera.registry import _parse_dataset_version, default_variant

    version_path, norm = _parse_dataset_version(version)
    variant = variant or default_variant(norm, "stream")
    table.add_row("Selected:", f"{version_path} {variant} (Icechunk)")
    table.add_row("Checkpoint:", str(attrs.get("checkpoint_id", "-")))
    table.add_row("Years:", f"{years[0]}-{years[-1]}" if years else "-")
    table.add_row("UTM zones:", str(len(store.zones())))
    table.add_row(
        "Incomplete zone-years:", str(sum(len(v) for v in incomplete.values()))
    )
    if verbose:
        for group, group_years in incomplete.items():
            table.add_row(f"  {group}:", ", ".join(map(str, group_years)))
    rprint(table)


def _datasets_table():
    """Print every dataset and its formats; return a table of its stores."""
    from geotessera.registry import (
        DATASETS,
        FORMATS,
        STREAM_FORMATS,
        TESSERA_MIRROR_URL,
        default_variant,
    )

    yes, no = emoji("✓") or "yes", emoji("✗") or "-"
    table = create_table(box=None)
    for column in ("Version", "Variant", "NPY", "Zarr", "Icechunk", "Description"):
        table.add_column(column)
    stores = create_table(box=None)
    for column in ("Version", "Variant", "Format", "URL"):
        stores.add_column(column)
    notes = []
    for ds in DATASETS:
        default = default_variant(ds.version)
        table.add_row(
            f"v{ds.version}",
            ds.variant + ("*" if ds.variant == default else ""),
            *(yes if ds.location(fmt) else no for fmt in FORMATS),
            ds.description,
        )
        for fmt in FORMATS:
            key = "stream" if fmt in STREAM_FORMATS else fmt
            fmt_default = default_variant(ds.version, key)
            if ds.location(fmt) and fmt_default == ds.variant != default:
                notes.append(f"v{ds.version} {fmt.upper()} defaults to {ds.variant}.")
        if ds.zarr:
            url = f"{TESSERA_MIRROR_URL}/zarr/{ds.zarr}"
            stores.add_row(f"v{ds.version}", ds.variant, "Zarr", url)
        if ds.icechunk:
            stores.add_row(f"v{ds.version}", ds.variant, "Icechunk", ds.icechunk)
    rprint(table)
    rprint(
        "[dim]* Default variant. "
        + " ".join(notes)
        + " Embeddings of different versions or variants cannot be "
        "interchanged. NPY tiles are deprecated and will be removed.[/dim]"
    )
    return stores


def _tile_registry_coverage(
    registry_url,
    output,
    *,
    year,
    region_bbox,
    region_file,
    width_pixels,
    show_countries,
    tile_alpha,
    cache_dir,
):
    """Draw coverage from an Icechunk dataset's tile registry."""
    from .icechunk import TileRegistry
    from .visualization import visualize_tile_registry_coverage

    if output.suffix.lower() not in {".png", ".jpg", ".jpeg"} or output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        output = output / "tessera_coverage.png"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    registry = TileRegistry(registry_url, cache_dir=cache_dir)
    if region_bbox is None:
        rprint("[yellow]Reading the whole tile registry (about 140 MB)[/yellow]")
    with create_progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Reading tile registry", total=None, status="")
        tiles = registry.tiles(
            bbox=region_bbox,
            year=year,
            progress_callback=lambda done, total, status: progress.update(
                task, completed=done, total=total, status=status
            ),
        )
    if tiles.empty:
        rprint("[red]No registry tiles intersect the selection[/red]")
        raise typer.Exit(1)
    path = visualize_tile_registry_coverage(
        tiles,
        str(output),
        year=year,
        width_pixels=width_pixels,
        show_countries=show_countries,
        tile_alpha=tile_alpha,
        region_bbox=region_bbox,
        region_file=region_file,
    )
    shown = tiles if year is None else tiles[tiles["year"] == year]
    rprint(f"[green]{emoji('✅ ')}Coverage map saved to: {path}[/green]")
    rprint(
        f"Tiles: {shown[['zone', 'tile']].drop_duplicates().shape[0]:,} "
        f"of 2048 x 2048 pixels, years {', '.join(map(str, sorted(shown['year'].unique())))}"
    )


@app.command()
def coverage(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Write the PNG here, with JSON and HTML files in the same directory.",
        ),
    ] = Path("tessera_coverage.png"),
    year: Annotated[
        Optional[int],
        typer.Option(
            "--year",
            help="Show one year. If omitted, show all available years.",
        ),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="Limit the PNG map to a vector region from a file or URL.",
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country",
            help="Select a country by name or code, such as United Kingdom or GB.",
        ),
    ] = None,
    bbox: Annotated[
        Optional[str],
        typer.Option(
            "--bbox",
            help="Select WGS84 bounds as west,south,east,north, or one tile as lon,lat.",
        ),
    ] = None,
    tile: Annotated[
        Optional[str],
        typer.Option("--tile", help="Select the 0.1-degree tile containing lon,lat."),
    ] = None,
    tile_color: Annotated[
        str,
        typer.Option(
            "--tile-color",
            help="Set the tile color when year-based colors are disabled.",
        ),
    ] = "red",
    tile_alpha: Annotated[
        float, typer.Option("--tile-alpha", help="Set tile opacity from 0 to 1.")
    ] = 0.6,
    tile_size: Annotated[
        float,
        typer.Option(
            "--tile-size",
            help="Set the tile size multiplier. Use 1 for the actual size.",
        ),
    ] = 1.0,
    width_pixels: Annotated[
        int, typer.Option("--width", help="Set the PNG width in pixels.")
    ] = 2000,
    no_countries: Annotated[
        bool, typer.Option("--no-countries", help="Hide country boundaries.")
    ] = False,
    no_multi_year_colors: Annotated[
        bool,
        typer.Option("--no-multi-year-colors", help="Disable year-based tile colors."),
    ] = False,
    by_source: Annotated[
        bool,
        typer.Option(
            "--by-source",
            help="Show each dataset in a separate color. Omitted version and variant options select all datasets.",
        ),
    ] = False,
    dataset_version: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-version",
            help="Select a version, or all. The default is v1.1, or all with --by-source.",
        ),
    ] = None,
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Select a variant, or all. The default is the version's default variant, or all with --by-source.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir", help="Cache downloaded manifests in this directory."
        ),
    ] = None,
    registry_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--registry-dir",
            help="Read manifest.parquet and landmasks.parquet from this directory.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print additional details.")
    ] = False,
):
    """Show embedding availability as a PNG map and HTML globe.

    Region selectors limit the PNG map; the globe shows global coverage.
    Supporting JSON files and textures are written beside the PNG.

    Icechunk datasets, including the default, draw the PNG map alone from
    their tile registry, one rectangle per 2048-pixel tile.
    """
    from .visualization import visualize_global_coverage
    from rich.progress import BarColumn, TextColumn, TimeRemainingColumn

    from .inputs import resolve_region

    region_bbox, region_geometry = None, None
    if any(value is not None for value in (bbox, tile, region_file, country)):
        try:
            region_bbox, region_geometry = resolve_region(
                bbox=bbox, tile=tile, region_file=region_file, country=country
            )
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        point_selector = tile if tile is not None else bbox
        if point_selector is not None and len(point_selector.split(",")) == 2:
            lon, lat = map(float, point_selector.split(","))
            region_bbox = point_to_tile_bbox(lon, lat)
            rprint(
                f"Point ({lon}, {lat}) -> tile grid_{region_bbox[0]:.2f}_{region_bbox[1]:.2f}"
            )
        rprint(f"Region bounding box: {format_bbox(region_bbox)}")

    # Initialize GeoTessera
    if verbose:
        rprint("[blue]Initializing GeoTessera...[/blue]")

    # Resolve flag defaults. The defaults differ between single-source and
    # --by-source modes so the "complete picture" rendering doesn't require
    # the user to type --dataset-version=all every time.
    # An explicit 'all' for either flag forces the multi-source view even
    # without --by-source, as documented in the flag help.
    if (dataset_version is not None and dataset_version.lower() == "all") or (
        dataset_variant is not None and dataset_variant.lower() == "all"
    ):
        if not by_source:
            rprint(
                "[blue]Explicit 'all' requested: enabling --by-source rendering[/blue]"
            )
        by_source = True
    if not by_source:
        from geotessera.registry import icechunk_dataset

        icechunk = icechunk_dataset(dataset_version or "v1.1", dataset_variant)
        if icechunk is not None:
            _tile_registry_coverage(
                icechunk[1],
                output,
                year=year,
                region_bbox=region_bbox,
                region_file=region_geometry,
                width_pixels=width_pixels,
                show_countries=not no_countries and not region_file and not country,
                tile_alpha=tile_alpha,
                cache_dir=cache_dir,
            )
            return
    from geotessera.registry import _parse_dataset_version, default_variant

    if by_source:
        version_spec = dataset_version if dataset_version is not None else "all"
        variant_spec = dataset_variant if dataset_variant is not None else "all"
    else:
        version_spec = dataset_version if dataset_version is not None else "v1.1"
        variant_spec = (
            dataset_variant
            if dataset_variant is not None
            else default_variant(_parse_dataset_version(version_spec)[1], "npy")
        )

    # The placeholder GeoTessera below initialises one Registry to get its
    # cache paths; additional manifests are downloaded into the same cache
    # dir when by-source spans multiple versions.
    init_version = (
        "v1" if (by_source and version_spec.lower() == "all") else version_spec
    )
    # With every variant selected, open the version's default NPY variant.
    init_variant = (
        default_variant(_parse_dataset_version(init_version)[1], "npy")
        if (by_source and variant_spec.lower() == "all")
        else variant_spec
    )

    gt = _tiles_client(
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

            region_file_to_use = region_geometry

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
            logging.info(f"Exporting coverage for {len(datasets_to_export)} dataset(s)")
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
                    tint_color=ds["color"] if len(datasets_to_export) > 1 else None,
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


@app.command()
def download(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Write files to this directory. This option is required unless --dry-run is set.",
        ),
    ] = None,
    bbox: Annotated[
        Optional[str],
        typer.Option(
            "--bbox",
            help="Select WGS84 bounds as west,south,east,north, or one tile as lon,lat.",
        ),
    ] = None,
    tile: Annotated[
        Optional[str],
        typer.Option("--tile", help="Select the 0.1-degree tile containing lon,lat."),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="Select a vector region from a file or URL. Zarr uses its bounding box.",
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country",
            help="Select a country by name or code, such as United Kingdom or GB.",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Write GeoTIFFs (tiff) or download quantized arrays with scales and landmasks (npy).",
        ),
    ] = "tiff",
    year: Annotated[
        int, typer.Option("--year", help="Select the embedding year.")
    ] = 2024,
    bands: Annotated[
        Optional[str],
        typer.Option(
            "--bands",
            help="Select comma-separated, zero-based bands for TIFF output. The default is all bands at the selected depth.",
        ),
    ] = None,
    compress: Annotated[
        str, typer.Option("--compress", help="Set the GeoTIFF compression method.")
    ] = "lzw",
    list_files: Annotated[
        bool,
        typer.Option("--list-files", help="List tile output files with their sizes."),
    ] = False,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version",
            help="Select the dataset version. Run geotessera info to list datasets.",
        ),
    ] = "v1.1",
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Select the dataset variant. If omitted, use the version's default in the chosen format. Run geotessera info to list them.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Set the metadata cache directory. Zarr byte-range reads are cached within the process.",
        ),
    ] = None,
    registry_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--registry-dir",
            help="Read local manifests from this directory. This selects tiles in auto mode.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print additional details.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Report the size without reading embeddings. Zarr reports uncompressed output; tiles use manifest estimates.",
        ),
    ] = False,
    source: Annotated[
        DownloadSource,
        typer.Option(
            "--source",
            help="Select the source. Auto uses Zarr for TIFF and tiles for NPY or --registry-dir.",
        ),
    ] = DownloadSource.auto,
    store_url: Annotated[
        Optional[str],
        typer.Option(
            "--store-url",
            help="Read Zarr from this URL or local path, overriding the dataset options.",
        ),
    ] = None,
    depth: Annotated[
        Optional[int],
        typer.Option(
            "--depth",
            help="Read a published Zarr embedding prefix. If omitted, read the full embedding.",
        ),
    ] = None,
):
    """Export a region as GeoTIFFs or download individual tiles.

    Specify one of --bbox, --tile, --region-file, or --country. Zarr exports
    write tessera_YEAR_utmNN.tif on each native UTM grid with float32 values
    and NaN nodata. Country and vector selectors use their bounding boxes.

    Rerunning replaces each completed Zarr file. Individual tile downloads
    skip existing files; use a new output directory when changing the
    dataset or bands. --store-url and --depth apply to Zarr only.
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

    stream = source == "zarr" or (
        source == "auto" and format == "tiff" and registry_dir is None
    )
    if stream:
        if format != "tiff":
            raise typer.BadParameter(
                "Zarr streaming exports TIFF; use --source tiles for NPY"
            )
        if registry_dir is not None:
            raise typer.BadParameter(
                "--registry-dir applies to --source tiles; use --store-url for Zarr"
            )
        from .workflows import stream_download

        try:
            result = stream_download(
                output,
                bbox=bbox,
                tile=tile,
                region_file=region_file,
                country=country,
                year=year,
                version=dataset_version,
                variant=dataset_variant,
                store_url=store_url,
                cache_dir=cache_dir,
                bands=bands,
                depth=depth,
                compress=compress,
                dry_run=dry_run,
            )
        except (ValueError, OSError, KeyError) as exc:
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if dry_run:
            size = sum(item["uncompressed_bytes"] for item in result)
            rprint(
                f"Zarr export: {len(result)} UTM zone file(s), {format_bytes(size)} uncompressed output"
            )
            rprint(
                "Network transfer and compressed output sizes depend on store chunks and data."
            )
        else:
            rprint(f"SUCCESS: Exported {len(result)} GeoTIFF file(s) from Zarr")
            for path in result:
                rprint(path)
        return
    if store_url is not None or depth is not None:
        raise typer.BadParameter("--store-url and --depth require --source zarr")
    rprint(
        "[yellow]NPY tiles are deprecated and will be removed; "
        "stream GeoTIFFs with --source zarr.[/yellow]"
    )

    from .inputs import resolve_region

    try:
        bbox_coords, _ = resolve_region(
            bbox=bbox, tile=tile, region_file=region_file, country=country
        )
        # Registry point selectors intentionally use a degenerate centre bbox.
        point_selector = tile if tile is not None else bbox
        if point_selector is not None and len(point_selector.split(",")) == 2:
            lon, lat = map(float, point_selector.split(","))
            bbox_coords = point_to_tile_bbox(lon, lat)
            rprint(
                f"Point ({lon}, {lat}) -> tile grid_{bbox_coords[0]:.2f}_{bbox_coords[1]:.2f}"
            )
    except (ValueError, OSError) as exc:
        rprint(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

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

    # Initialize GeoTessera with embeddings_dir set to output directory
    gt = _tiles_client(
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None,
        embeddings_dir=str(output)
        if not dry_run
        else None,  # Only set for actual downloads
    )

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

    rprint(info_table)

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

            rprint(result_table)

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
                    failed_tiles = set()
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
                                failed_tiles.add((tile_year, tile_lon, tile_lat))
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
                                failed_tiles.add((tile_year, tile_lon, tile_lat))
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
                                failed_tiles.add((tile_year, tile_lon, tile_lat))
                                rprint(
                                    f"[yellow]Warning: Failed to download landmask for ({tile_lon}, {tile_lat}): {e}[/yellow]"
                                )

                    if failed_tiles:
                        raise RuntimeError(
                            f"{len(failed_tiles)} of {len(tiles_to_fetch)} tiles are incomplete; rerun to resume"
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

        rprint(tips_table)

    except Exception as e:
        rprint(f"\n[red]{emoji('❌ ')}Error: {e}[/red]")
        if verbose:
            rprint("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def visualize(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Read a GeoTIFF file or a directory of GeoTIFF or NPY tiles."
        ),
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Write the PCA mosaic to this GeoTIFF file.")
    ],
    target_crs: Annotated[
        str, typer.Option("--crs", help="Set the output coordinate reference system.")
    ] = "EPSG:3857",
    n_components: Annotated[
        int,
        typer.Option(
            "--n-components",
            min=1,
            help="Set the number of PCA components to fit. Only the first three are written.",
        ),
    ] = 3,
    balance_method: Annotated[
        BalanceMethod,
        typer.Option(
            "--balance",
            help="Set the color scaling method.",
        ),
    ] = BalanceMethod.histogram,
    percentile_low: Annotated[
        float,
        typer.Option(
            "--percentile-low",
            help="Set the lower clipping percentile with --balance percentile.",
        ),
    ] = 2.0,
    percentile_high: Annotated[
        float,
        typer.Option(
            "--percentile-high",
            help="Set the upper clipping percentile with --balance percentile.",
        ),
    ] = 98.0,
):
    """Create a PCA mosaic from GeoTIFF or NPY embeddings.

    Read one GeoTIFF or a directory of GeoTIFF or NPY tiles. Fit one PCA
    model to a reproducible sample of up to 100,000 valid pixels and apply
    it across all inputs. Missing pixels remain masked.

    The output contains the first three components as a display-scaled uint8
    image, or fewer bands if fewer components are requested. Use the
    original embeddings for analysis.
    """

    # Validate output file extension
    if output_file.suffix.lower() not in [".tif", ".tiff"]:
        rprint("[red]Error: Output file must have .tif or .tiff extension[/red]")
        raise typer.Exit(1)

    if n_components < 3:
        rprint(
            f"[yellow]Warning: Using {n_components} component(s). RGB visualization works best with 3+ components[/yellow]"
        )

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
            tiles_data = tiles

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
    rgb_mosaic: Annotated[
        Optional[Path],
        typer.Argument(help="Read this RGB GeoTIFF. Omit it to stream a Zarr region."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Write map files here. The default is tessera_webmap for a region or <image_stem>_webmap for an image.",
        ),
    ] = None,
    min_zoom: Annotated[
        int,
        typer.Option(
            "--min-zoom", help="Set the lowest tile zoom level, from 0 to 24."
        ),
    ] = 8,
    max_zoom: Annotated[
        int,
        typer.Option(
            "--max-zoom", help="Set the highest tile zoom level, from 0 to 24."
        ),
    ] = 15,
    initial_zoom: Annotated[
        int, typer.Option("--initial-zoom", help="Set the initial viewer zoom level.")
    ] = 10,
    force_regenerate: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Rebuild the streamed RGB mosaic and tiles. Use --force when a store changes at the same URL.",
        ),
    ] = False,
    serve_immediately: Annotated[
        bool,
        typer.Option(
            "--serve/--no-serve",
            help="Start the server and open a browser after processing. Reserve the port before processing.",
        ),
    ] = False,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Set the web server port. An occupied port causes an error.",
        ),
    ] = 8000,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="Overlay a vector boundary from a file or URL. In Zarr mode, also select its bounding box.",
        ),
    ] = None,
    use_gdal_raster: Annotated[
        bool,
        typer.Option(
            "--use-gdal-raster/--use-gdal2tiles",
            help="Generate tiles with gdal raster tile or gdal2tiles. The selected tool must be on PATH.",
        ),
    ] = False,
    bbox: Annotated[
        Optional[str],
        typer.Option(
            "--bbox",
            help="Select WGS84 bounds as west,south,east,north, or one tile as lon,lat.",
        ),
    ] = None,
    tile: Annotated[
        Optional[str],
        typer.Option("--tile", help="Select the 0.1-degree tile containing lon,lat."),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country",
            help="Select a country by name or code, such as United Kingdom or GB.",
        ),
    ] = None,
    year: Annotated[
        int, typer.Option("--year", help="Select the embedding year.")
    ] = 2024,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version",
            help="Select the dataset version. Run geotessera info to list datasets.",
        ),
    ] = "v1.1",
    dataset_variant: Annotated[
        Optional[str],
        typer.Option(
            "--dataset-variant",
            help="Select the dataset variant. If omitted, use the version's default in the chosen format. Run geotessera info to list them.",
        ),
    ] = None,
    store_url: Annotated[
        Optional[str],
        typer.Option(
            "--store-url",
            help="Read Zarr from this URL or local path, overriding the dataset options.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Set the metadata cache directory. Zarr byte-range reads are cached within the process.",
        ),
    ] = None,
    bands: Annotated[
        str,
        typer.Option(
            "--bands",
            help="Select three comma-separated, zero-based embedding bands for a Zarr region.",
        ),
    ] = "0,1,2",
    depth: Annotated[
        Optional[int],
        typer.Option(
            "--depth",
            help="Read a published Zarr embedding prefix. If omitted, read the full embedding.",
        ),
    ] = None,
):
    """Create web tiles and a viewer from an RGB image or Zarr region.

    Supply a three-band RGB GeoTIFF, or omit it and select one region with
    --bbox, --tile, --region-file, or --country. Region mode scales three
    embedding bands to RGB. Use visualize first for PCA maps.

    Matching completed mosaics and tiles are reused. Changing the zoom
    range rebuilds only tiles. Keep the output directory and its JSON
    completion files to reuse completed work. GDAL command-line tools are
    required to generate tiles.
    """
    if not 0 <= min_zoom <= max_zoom <= 24:
        raise typer.BadParameter("Zoom levels must satisfy 0 <= min <= max <= 24")
    from .workflows import cached_stream_rgb, file_identity, stage_matches, save_stage

    # Reserve the port before any expensive processing, and release it even on failure.
    server = None
    if serve_immediately:
        import click

        serving_directory = output or (
            Path("tessera_webmap")
            if rgb_mosaic is None
            else Path(f"{rgb_mosaic.stem}_webmap")
        )
        server = _bind_web_server(serving_directory, port)
        click.get_current_context().call_on_close(server.server_close)
    if rgb_mosaic is None:
        output = output or Path("tessera_webmap")
        output.mkdir(parents=True, exist_ok=True)
        try:
            rgb_mosaic, reused = cached_stream_rgb(
                output / "rgb_mosaic.tif",
                force=force_regenerate,
                bbox=bbox,
                tile=tile,
                region_file=region_file,
                country=country,
                year=year,
                version=dataset_version,
                variant=dataset_variant,
                store_url=store_url,
                cache_dir=cache_dir,
                bands=bands,
                depth=depth,
            )
        except (ValueError, OSError, KeyError) as exc:
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if reused:
            rprint("[green]Using existing streamed RGB mosaic[/green]")
    elif any(value is not None for value in (bbox, tile, country, store_url, depth)):
        raise typer.BadParameter("Choose an existing RGB mosaic or a Zarr region")

    if not rgb_mosaic.exists():
        rprint(f"[red]Error: Mosaic file {rgb_mosaic} does not exist[/red]")
        raise typer.Exit(1)

    if rgb_mosaic.suffix.lower() not in [".tif", ".tiff"]:
        rprint("[red]Error: Input must be a GeoTIFF file (.tif/.tiff)[/red]")
        raise typer.Exit(1)

    # GeoPandas handles supported paths and URLs directly; no temporary copies.
    region_file_path = region_file
    region_file_temp = None

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

        tile_marker = output / "tiles.json"
        tile_state = {
            "schema": 1,
            "source": file_identity(rgb_mosaic),
            "zoom": [min_zoom, max_zoom],
            "gdal_raster": use_gdal_raster,
        }
        regenerate_tiles = force_regenerate or not stage_matches(
            tile_marker, tile_state
        )
        if regenerate_tiles:
            tile_marker.unlink(missing_ok=True)
        if regenerate_tiles and tiles_dir.exists():
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
                save_stage(tile_marker, tile_state)
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
        _run_web_server(server, output, True, "viewer.html")
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
        Path, typer.Argument(help="Serve all files in this directory.")
    ],
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Set the web server port. An occupied port causes an error.",
        ),
    ] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the viewer in a browser.")
    ] = True,
    html_file: Annotated[
        Optional[str],
        typer.Option(
            "--html", help="Open this HTML file relative to the served directory."
        ),
    ] = None,
):
    """Serve an existing map directory over HTTP.

    Serve all files in DIRECTORY without regenerating the map. An occupied
    port causes an error. Press Ctrl+C to stop the server.
    """
    if not directory.exists():
        rprint(f"[red]Error: Directory {directory} does not exist[/red]")
        raise typer.Exit(1)

    if not directory.is_dir():
        rprint(f"[red]Error: {directory} is not a directory[/red]")
        raise typer.Exit(1)

    with _bind_web_server(directory, port) as server:
        _run_web_server(server, directory, open_browser, html_file)


def _bind_web_server(directory, port):
    from functools import partial
    import socket

    dual_stack = socket.has_dualstack_ipv6()

    class ExclusiveHTTPServer(http.server.ThreadingHTTPServer):
        # Python versions may enable SO_REUSEPORT: that can share a busy port.
        allow_reuse_address = False
        allow_reuse_port = False
        address_family = socket.AF_INET6 if dual_stack else socket.AF_INET

        def server_bind(self):
            # Windows otherwise lets a wildcard listener share a port already
            # bound on loopback, even with SO_REUSEADDR disabled.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            if dual_stack:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory.resolve())
    )
    try:
        return ExclusiveHTTPServer(("::" if dual_stack else "", port), handler)
    except OSError as exc:
        raise typer.BadParameter(
            f"Cannot bind web server on port {port}: {exc}. Choose another --port."
        ) from exc


def _run_web_server(server, directory, open_browser, html_file):
    port = server.server_port
    if html_file is None:
        html_file = next(
            (
                name
                for name in ("index.html", "viewer.html", "map.html", "coverage.html")
                if (directory / name).exists()
            ),
            "",
        )
    url = f"http://localhost:{port}/" + urllib.parse.quote(html_file, safe="/")
    rprint(f"Web server running at: {url}")
    rprint(f"Serving directory: {directory.resolve()}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        rprint("Stopping web server...")


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
