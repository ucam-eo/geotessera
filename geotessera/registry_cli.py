#!/usr/bin/env python3
"""
Command-line interface for managing GeoTessera registry files.

This module provides tools for generating and maintaining Pooch registry files
used by the GeoTessera package. It supports parallel processing, incremental
updates, and generation of a master registry index.
"""

import os
import re
import hashlib
import argparse
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, Dict, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

from .registry import (
    TESSERA_MIRROR_ENDPOINT,
    _DATASET_DIR_RE,
    block_from_world,
    block_to_embeddings_registry_filename,
    block_to_landmasks_registry_filename,
    dataset_from_path,
    dataset_path,
    default_variant,
    format_bytes,
    parse_grid_name,
    zarr_store_url,
)
from ._terminal import console, emoji

# Module-level logger
logger = logging.getLogger(__name__)


def _atomic_write_parquet(df, dest, *, compression=None):
    """Write ``df`` to ``dest`` atomically via a temp file + rename (cron-safe).

    Works for both DataFrames (regular parquet) and GeoDataFrames
    (GeoParquet, chosen automatically by ``to_parquet``).
    """
    from .remote import atomic_output

    with atomic_output(dest, suffix=".parquet") as temp_path:
        # NamedTemporaryFile is 0o600 and these files get published.
        os.chmod(temp_path, 0o644)
        if compression:
            df.to_parquet(temp_path, compression=compression, index=False)
        else:
            df.to_parquet(temp_path, index=False)


@dataclass
class TileInfo:
    """Complete information about a single tessera tile."""

    year: int
    lat: float
    lon: float
    grid_name: str
    embedding_path: str
    scales_path: str
    directory_path: str
    embedding_hash: Optional[str]
    scales_hash: Optional[str]
    embedding_mtime: float
    scales_mtime: float
    embedding_size: int
    scales_size: int


def process_grid_directory(args):
    """Process a single grid directory and return TileInfo.

    Args:
        args: Tuple of (year, year_path, grid_item, base_dir)

    Returns:
        TileInfo object or None if directory should be skipped
    """
    year, year_path, grid_item, _base_dir = args
    grid_path = os.path.join(year_path, grid_item)

    try:
        # Check if directory has any files at all
        dir_contents = os.listdir(grid_path)
        if not dir_contents:
            return None  # Empty directory - skip silently

        # Check if this looks like a tile directory (has SHA256 or .npy files)
        has_sha256 = "SHA256" in dir_contents
        has_npy_files = any(f.endswith(".npy") for f in dir_contents)

        if not has_sha256 and not has_npy_files:
            return None  # Directory has files but doesn't look like a tile directory - skip

        # Parse coordinates from grid name
        lon, lat = parse_grid_name(grid_item)
        if lon is None or lat is None:
            raise ValueError(f"Could not parse coordinates from grid name: {grid_item}")

        # Read SHA256 file
        sha256_file = os.path.join(grid_path, "SHA256")
        if not os.path.exists(sha256_file):
            # If there are .npy files but no SHA256, return warning and skip
            if has_npy_files:
                return ("WARNING", "missing SHA256 file", grid_path)
            return None  # Skip this directory

        # Parse hashes from SHA256 file
        embedding_hash = None
        scales_hash = None

        with open(sha256_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    hash_value = parts[0]
                    filename = parts[-1]  # Handle spaces in paths

                    if filename == f"{grid_item}.npy":
                        embedding_hash = hash_value
                    elif filename == f"{grid_item}_scales.npy":
                        scales_hash = hash_value

        # Validate required files and hashes
        embedding_path = os.path.join(grid_path, f"{grid_item}.npy")
        scales_path = os.path.join(grid_path, f"{grid_item}_scales.npy")

        embedding_exists = os.path.exists(embedding_path)
        scales_exists = os.path.exists(scales_path)

        # Skip directories with incomplete npy/scales files (one but not the other)
        if embedding_exists and not scales_exists:
            return ("WARNING", "missing scales", grid_path)
        if scales_exists and not embedding_exists:
            return ("WARNING", "missing embedding", grid_path)
        if not embedding_exists and not scales_exists:
            # Both missing - skip silently (probably not a tile directory)
            return None

        # Check for hashes in SHA256 file
        if embedding_hash is None:
            return ("WARNING", "no hash for embedding in SHA256 file", grid_path)
        if scales_hash is None:
            return ("WARNING", "no hash for scales in SHA256 file", grid_path)

        # Get file stats
        embedding_stat = os.stat(embedding_path)
        scales_stat = os.stat(scales_path)

        # Create TileInfo object
        tile_info = TileInfo(
            year=year,
            lat=lat,
            lon=lon,
            grid_name=grid_item,
            embedding_path=embedding_path,
            scales_path=scales_path,
            directory_path=grid_path,
            embedding_hash=embedding_hash,
            scales_hash=scales_hash,
            embedding_mtime=embedding_stat.st_mtime,
            scales_mtime=scales_stat.st_mtime,
            embedding_size=embedding_stat.st_size,
            scales_size=scales_stat.st_size,
        )

        return tile_info

    except Exception as e:
        # Return error info as a special tuple
        return ("ERROR", grid_item, str(e))


def iterate_tessera_tiles(
    base_dir: str,
    callback: Callable[[TileInfo], Any],
    progress_callback: Optional[Callable] = None,
) -> tuple[List[Any], List[tuple[str, str]]]:
    """
    Single-pass iterator through Tessera embedding filesystem structure.

    For each tile:
    1. Reads hashes from existing SHA256 file in grid directory
    2. Gets file stats (mtime, size) from filesystem
    3. Calls callback with complete TileInfo object
    4. Validates that both embedding and scales files exist

    Uses multiprocessing for parallel directory scanning.

    Args:
        base_dir: Base directory containing global_0.1_degree_representation
        callback: Function called for each tile with TileInfo object
        progress_callback: Optional progress reporting function(current, total, status)

    Returns:
        Tuple of (results, warnings) where:
        - results: List of results from callback calls (None results are filtered out)
        - warnings: List of (reason, path) tuples for skipped directories

    Raises:
        FileNotFoundError: Missing SHA256 files or embedding/scales files
        ValueError: Corrupted directory structure or hash format
        OSError: Filesystem access errors
    """
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    if not os.path.exists(repr_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {repr_dir}")

    results = []
    warnings = []
    processed_dirs = 0
    total_dirs = 0

    # First count total directories for progress (all grid directories, even empty ones)
    for year_item in os.listdir(repr_dir):
        year_path = os.path.join(repr_dir, year_item)
        if os.path.isdir(year_path) and year_item.isdigit() and len(year_item) == 4:
            for grid_item in os.listdir(year_path):
                grid_path = os.path.join(year_path, grid_item)
                if os.path.isdir(grid_path) and grid_item.startswith("grid_"):
                    total_dirs += 1

    if total_dirs == 0:
        # No grid directories at all
        return results, warnings  # Return empty lists instead of raising error

    # Get number of CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()

    # Report parallelization to user if progress callback is available
    if progress_callback:
        progress_callback(
            0, total_dirs, f"Using {num_cores} CPU cores for parallel processing"
        )

    # Collect all grid directory tasks
    grid_tasks = []
    for year_item in os.listdir(repr_dir):
        year_path = os.path.join(repr_dir, year_item)
        if not (
            os.path.isdir(year_path) and year_item.isdigit() and len(year_item) == 4
        ):
            continue

        year = int(year_item)

        for grid_item in os.listdir(year_path):
            grid_path = os.path.join(year_path, grid_item)
            if os.path.isdir(grid_path) and grid_item.startswith("grid_"):
                grid_tasks.append((year, year_path, grid_item, base_dir))

    # Process grid directories in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_grid_directory, task): task for task in grid_tasks
        }

        # Process results as they complete
        for future in as_completed(futures):
            task = futures[future]
            year, year_path, grid_item, _ = task

            processed_dirs += 1

            # Update progress
            if progress_callback:
                progress_callback(processed_dirs, total_dirs, f"Processing {grid_item}")

            try:
                tile_info_or_error = future.result()

                # Check if this is an error result
                if (
                    isinstance(tile_info_or_error, tuple)
                    and len(tile_info_or_error) == 3
                ):
                    if tile_info_or_error[0] == "ERROR":
                        _, grid_name, error_msg = tile_info_or_error
                        grid_path = os.path.join(year_path, grid_name)
                        raise RuntimeError(f"Error processing {grid_path}: {error_msg}")
                    elif tile_info_or_error[0] == "WARNING":
                        _, reason, path = tile_info_or_error
                        warnings.append((reason, path))
                        continue

                # Skip None results (empty/skipped directories)
                if tile_info_or_error is None:
                    continue

                # Call callback and collect result
                result = callback(tile_info_or_error)
                if result is not None:
                    results.append(result)

            except Exception as e:
                # Stop on first error as requested
                if isinstance(e, RuntimeError):
                    raise
                else:
                    raise RuntimeError(
                        f"Unexpected error in parallel processing: {e}"
                    ) from e

    return results, warnings


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _check_tiff_has_land(file_path):
    """Check if a landmask TIFF contains any land pixels (non-zero data).

    Args:
        file_path: Path to the TIFF file

    Returns:
        True if the TIFF has at least one non-zero pixel, False otherwise
    """
    import rasterio

    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            return data.max() > 0
    except Exception:
        return False


def create_landmasks_parquet_database(base_dir, output_path, console):
    """Create a Parquet database for landmasks by reading from SHA256SUM file.

    Validates each TIFF in parallel, skipping ocean-only tiles (all-zero data).

    Args:
        base_dir: Base directory containing global_0.1_degree_tiff_all
        output_path: Output path for the Parquet file
        console: Rich console for output

    Returns:
        True on success, False on failure
    """
    console.print(
        Panel.fit(
            f"[bold blue]🗺️ Creating Landmasks Parquet Database[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"📄 {output_path}",
            style="blue",
        )
    )

    sha256sum_file = os.path.join(base_dir, "SHA256SUM")
    if not os.path.exists(sha256sum_file):
        console.print(f"[red]SHA256SUM file not found:[/red] {sha256sum_file}")
        return False

    # Phase 1: Parse SHA256SUM file to collect candidate tiles
    candidates = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        read_task = progress.add_task(
            "Reading SHA256SUM file...", total=100, status="Starting..."
        )

        try:
            with open(sha256sum_file, "r") as f:
                lines = f.readlines()

            progress.update(read_task, completed=50, status="Parsing entries...")

            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        checksum = parts[0]
                        filename = parts[-1]

                        if filename.endswith(".tiff") or filename.endswith(".tif"):
                            if filename.startswith("grid_"):
                                try:
                                    # Remove 'grid_' prefix and '.tiff' suffix
                                    coords_str = (
                                        filename[5:]
                                        .replace(".tiff", "")
                                        .replace(".tif", "")
                                    )
                                    lon_str, lat_str = coords_str.split("_")
                                    lon = float(lon_str)
                                    lat = float(lat_str)

                                    file_path = os.path.join(base_dir, filename)
                                    if not os.path.exists(file_path):
                                        continue

                                    file_size = os.path.getsize(file_path)

                                    candidates.append(
                                        {
                                            "lat": lat,
                                            "lon": lon,
                                            "hash": checksum,
                                            "file_size": file_size,
                                            "file_path": file_path,
                                        }
                                    )
                                except (ValueError, IndexError):
                                    continue

            progress.update(read_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]Error reading SHA256SUM file: {e}[/red]")
            return False

    if not candidates:
        console.print("[red]No landmask tiles found in SHA256SUM file[/red]")
        return False

    # Phase 2: Validate TIFFs in parallel, skip all-zero (ocean-only) tiles
    records = []
    skipped_ocean = 0
    num_workers = min(multiprocessing.cpu_count(), 16)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        validate_task = progress.add_task(
            "Validating landmask TIFFs...",
            total=len(candidates),
            status=f"0 land / 0 ocean ({num_workers} workers)",
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_candidate = {
                executor.submit(_check_tiff_has_land, c["file_path"]): c
                for c in candidates
            }

            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    has_land = future.result()
                except Exception:
                    has_land = False

                if has_land:
                    records.append(
                        {
                            "lat": candidate["lat"],
                            "lon": candidate["lon"],
                            "hash": candidate["hash"],
                            "file_size": candidate["file_size"],
                        }
                    )
                else:
                    skipped_ocean += 1

                progress.update(
                    validate_task,
                    advance=1,
                    status=f"{len(records):,} land / {skipped_ocean:,} ocean",
                )

    if skipped_ocean > 0:
        console.print(
            f"[yellow]Skipped {skipped_ocean:,} ocean-only TIFFs (all-zero data)[/yellow]"
        )

    if not records:
        console.print("[red]No land tiles found after validation[/red]")
        return False

    # Phase 3: Convert to GeoParquet
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        parquet_task = progress.add_task(
            "Creating GeoParquet database...",
            total=100,
            status="Converting to DataFrame...",
        )

        progress.update(parquet_task, completed=25, status="Sorting records...")
        df = pd.DataFrame(records)
        df = df.sort_values(["lat", "lon"])

        # Add integer grid indices for robust cross-platform lookups
        df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
        df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

        progress.update(parquet_task, completed=50, status="Creating geometries...")
        # Convert to GeoDataFrame with Point geometries
        geometry = gpd.points_from_xy(df["lon"], df["lat"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        progress.update(parquet_task, completed=75, status="Writing GeoParquet file...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        _atomic_write_parquet(gdf, output_path, compression="zstd")

        progress.update(parquet_task, completed=100, status="Complete")

    # Get file size and show results
    file_size = Path(output_path).stat().st_size

    # Summary table
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("📊 Records:", f"{len(records):,}")
    summary_table.add_row("🚫 Skipped ocean:", f"{skipped_ocean:,}")
    summary_table.add_row("💾 File size:", f"{file_size:,} bytes")
    summary_table.add_row(
        "🌍 Coordinates:",
        f"{len(gdf[['lat', 'lon']].drop_duplicates()):,} unique tiles",
    )
    summary_table.add_row("🗺️ Format:", "GeoParquet with Point geometries")

    console.print(
        Panel(
            summary_table,
            title="[bold green]✅ Landmasks GeoParquet Database Created[/bold green]",
            border_style="green",
        )
    )

    return True


def create_parquet_database_from_filesystem(base_dir, output_path, console):
    """Create a Parquet database by reading from existing SHA256 files.

    Fast implementation that reads hashes from SHA256 files instead of
    recalculating them, making database creation much faster.
    Uses temporary file for atomic writing to ensure cron-safe operation.

    Args:
        base_dir: Base directory containing global_0.1_degree_representation
        output_path: Output path for the Parquet file
        console: Rich console for output
    """
    # Show initial header
    console.print(
        Panel.fit(
            f"[bold blue]🗄️ Creating Embeddings Parquet Database[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"📄 {output_path}",
            style="blue",
        )
    )

    # Show CPU core count for parallel processing
    num_cores = multiprocessing.cpu_count()
    console.print(
        f"[cyan]Using {num_cores} CPU cores for parallel tile scanning[/cyan]"
    )

    records = []

    def collect_tile_data(tile_info: TileInfo):
        """Callback to collect data for each tile."""
        return {
            "lat": tile_info.lat,
            "lon": tile_info.lon,
            "year": tile_info.year,
            "hash": tile_info.embedding_hash,  # From SHA256 file, no recalculation!
            "scales_hash": tile_info.scales_hash,  # Scales file hash
            "mtime": tile_info.embedding_mtime,
            "file_size": tile_info.embedding_size,
            "scales_size": tile_info.scales_size,  # Scales file size
        }

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        # Single progress task for the iterator
        process_task = progress.add_task(
            "Reading tile metadata...", total=100, status="Starting..."
        )

        def progress_callback(current, total, status):
            progress.update(process_task, completed=current, total=total, status=status)

        try:
            # Use the fast iterator - reads SHA256 files, no hash calculation
            records, warnings = iterate_tessera_tiles(
                base_dir, collect_tile_data, progress_callback=progress_callback
            )

            progress.update(process_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]Error iterating tiles: {e}[/red]")
            return False

        if not records:
            console.print(
                "[red]No tiles found. Make sure 'geotessera-registry hash' has been run first.[/red]"
            )
            return False

        # Convert to Parquet
        parquet_task = progress.add_task(
            "Creating Parquet database...",
            total=100,
            status="Converting to DataFrame...",
        )

        progress.update(parquet_task, completed=25, status="Sorting records...")
        df = pd.DataFrame(records)
        df = df.sort_values(["year", "lat", "lon"])

        progress.update(parquet_task, completed=40, status="Converting timestamps...")
        df["mtime"] = pd.to_datetime(df["mtime"], unit="s")

        # Add integer grid indices for robust cross-platform lookups
        df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
        df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

        progress.update(parquet_task, completed=55, status="Creating geometries...")
        # Convert to GeoDataFrame with Point geometries
        geometry = gpd.points_from_xy(df["lon"], df["lat"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        progress.update(parquet_task, completed=75, status="Writing GeoParquet file...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        _atomic_write_parquet(gdf, output_path, compression="zstd")

        progress.update(parquet_task, completed=100, status="Complete")

    # Get file size and show results
    file_size = Path(output_path).stat().st_size

    # Summary table
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("📊 Records:", f"{len(records):,}")
    summary_table.add_row("💾 File size:", f"{file_size:,} bytes")
    summary_table.add_row("🗓️ Years:", ", ".join(map(str, sorted(gdf["year"].unique()))))
    summary_table.add_row(
        "🌍 Coordinates:",
        f"{len(gdf[['lat', 'lon']].drop_duplicates()):,} unique tiles",
    )
    summary_table.add_row("🗺️ Format:", "GeoParquet with Point geometries")

    console.print(
        Panel(
            summary_table,
            title="[bold green]✅ GeoParquet Database Created[/bold green]",
            border_style="green",
        )
    )

    # Display warning summary if there are any skipped directories
    if warnings:
        console.print()
        console.print(
            f"[yellow]⚠ Skipped {len(warnings)} director{'y' if len(warnings) == 1 else 'ies'} with incomplete files:[/yellow]"
        )

        # Group warnings by reason for better readability
        from collections import defaultdict

        warnings_by_reason = defaultdict(list)
        for reason, path in warnings:
            warnings_by_reason[reason].append(path)

        for reason, paths in sorted(warnings_by_reason.items()):
            console.print(f"\n[dim]  {reason.capitalize()}:[/dim]")
            for path in sorted(paths):
                console.print(f"    {path}")

        # Write missing files to separate text files
        output_dir = Path(output_path).parent

        # Write missing embeddings
        missing_embeddings = warnings_by_reason.get("missing embedding", [])
        if missing_embeddings:
            missing_embeddings_file = output_dir / "missing_embeddings.txt"
            with open(missing_embeddings_file, "w") as f:
                for path in sorted(missing_embeddings):
                    f.write(f"{path}\n")
            console.print(
                f"\n[dim]Written {len(missing_embeddings)} paths to {missing_embeddings_file}[/dim]"
            )

        # Write missing scales
        missing_scales = warnings_by_reason.get("missing scales", [])
        if missing_scales:
            missing_scales_file = output_dir / "missing_scales.txt"
            with open(missing_scales_file, "w") as f:
                for path in sorted(missing_scales):
                    f.write(f"{path}\n")
            console.print(
                f"[dim]Written {len(missing_scales)} paths to {missing_scales_file}[/dim]"
            )

    return True


def check_command(args):
    """Check the integrity of the tessera filesystem structure."""
    console = Console()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Error: Directory {base_dir} does not exist[/red]")
        return 1

    verify_hashes = getattr(args, "verify_hashes", False)

    # Show header
    console.print(
        Panel.fit(
            f"[bold blue]🔍 Checking Tessera Structure[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"🔐 Hash verification: {'enabled' if verify_hashes else 'disabled'}",
            style="blue",
        )
    )

    checked_tiles = 0

    truncated_files: list = []

    def _check_npy_integrity(path: str) -> Optional[str]:
        """Verify a .npy file header matches its file size.

        Returns an error message if truncated, None if OK.
        """
        try:
            file_size = os.path.getsize(path)
            # Read numpy header to get dtype and shape without mmap
            with open(path, "rb") as f:
                version = np.lib.format.read_magic(f)
                shape, _fortran, dtype = np.lib.format._read_array_header(f, version)
                header_size = f.tell()
            expected_data = int(np.prod(shape)) * dtype.itemsize
            expected_total = header_size + expected_data
            if file_size < expected_total:
                return (
                    f"truncated: {file_size:,} bytes on disk, "
                    f"expected {expected_total:,} "
                    f"(shape={shape}, dtype={dtype})"
                )
        except Exception as e:
            return f"unreadable: {e}"
        return None

    def validate_tile(tile_info: TileInfo):
        """Callback to validate each tile."""
        nonlocal checked_tiles
        checked_tiles += 1

        # Check if files exist (already done by iterator, but good to be explicit)
        if not os.path.exists(tile_info.embedding_path):
            raise FileNotFoundError(f"Missing embedding: {tile_info.embedding_path}")
        if not os.path.exists(tile_info.scales_path):
            raise FileNotFoundError(f"Missing scales: {tile_info.scales_path}")

        # Check .npy file integrity (detects truncated downloads)
        for path in [tile_info.embedding_path, tile_info.scales_path]:
            if path.endswith(".npy"):
                err = _check_npy_integrity(path)
                if err is not None:
                    truncated_files.append((path, err))

        # Verify hashes if requested
        if verify_hashes:
            actual_embedding_hash = calculate_sha256(tile_info.embedding_path)
            if actual_embedding_hash != tile_info.embedding_hash:
                raise ValueError(
                    f"Hash mismatch in {tile_info.embedding_path}: "
                    f"expected {tile_info.embedding_hash}, got {actual_embedding_hash}"
                )

            actual_scales_hash = calculate_sha256(tile_info.scales_path)
            if actual_scales_hash != tile_info.scales_hash:
                raise ValueError(
                    f"Hash mismatch in {tile_info.scales_path}: "
                    f"expected {tile_info.scales_hash}, got {actual_scales_hash}"
                )

        return "OK"

    # Run validation with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        check_task = progress.add_task(
            "Validating tiles...", total=100, status="Starting..."
        )

        def progress_callback(current, total, status):
            progress.update(check_task, completed=current, total=total, status=status)

        try:
            # Run validation
            _, warnings = iterate_tessera_tiles(
                base_dir, validate_tile, progress_callback=progress_callback
            )

            progress.update(check_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]{emoji('❌ ')}Validation failed: {e}[/red]")
            return 1

    # Show results
    has_errors = len(truncated_files) > 0
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("Tiles checked:", f"{checked_tiles:,}")
    summary_table.add_row("Truncated files:", f"{len(truncated_files):,}")
    if verify_hashes:
        summary_table.add_row("Hash verification:", "All hashes verified")
    else:
        summary_table.add_row("Hash verification:", "Skipped (use --verify-hashes)")

    if has_errors:
        console.print(
            Panel(
                summary_table,
                title="[bold red]Tessera Structure Check: errors found[/bold red]",
                border_style="red",
            )
        )
        console.print(f"\n[red]Truncated .npy files ({len(truncated_files)}):[/red]")
        for path, err in sorted(truncated_files):
            console.print(f"  {path}")
            console.print(f"    [dim]{err}[/dim]")

        # Write truncated files list
        output_dir = Path(base_dir)
        truncated_file = output_dir / "truncated_files.txt"
        with open(truncated_file, "w") as f:
            for path, err in sorted(truncated_files):
                f.write(f"{path}\t{err}\n")
        console.print(f"\n[dim]Truncated file list written to {truncated_file}[/dim]")
    else:
        summary_table.add_row("Status:", "All checks passed")
        console.print(
            Panel(
                summary_table,
                title="[bold green]Tessera Structure Check Complete[/bold green]",
                border_style="green",
            )
        )

    # Display warning summary if there are any skipped directories
    if warnings:
        console.print()
        console.print(
            f"[yellow]⚠ Skipped {len(warnings)} director{'y' if len(warnings) == 1 else 'ies'} with incomplete files:[/yellow]"
        )

        # Group warnings by reason for better readability
        from collections import defaultdict

        warnings_by_reason = defaultdict(list)
        for reason, path in warnings:
            warnings_by_reason[reason].append(path)

        for reason, paths in sorted(warnings_by_reason.items()):
            console.print(f"\n[dim]  {reason.capitalize()}:[/dim]")
            for path in sorted(paths):
                console.print(f"    {path}")

        # Write missing files to separate text files
        output_dir = Path(base_dir)

        # Write missing embeddings
        missing_embeddings = warnings_by_reason.get("missing embedding", [])
        if missing_embeddings:
            missing_embeddings_file = output_dir / "missing_embeddings.txt"
            with open(missing_embeddings_file, "w") as f:
                for path in sorted(missing_embeddings):
                    f.write(f"{path}\n")
            console.print(
                f"\n[dim]Written {len(missing_embeddings)} paths to {missing_embeddings_file}[/dim]"
            )

        # Write missing scales
        missing_scales = warnings_by_reason.get("missing scales", [])
        if missing_scales:
            missing_scales_file = output_dir / "missing_scales.txt"
            with open(missing_scales_file, "w") as f:
                for path in sorted(missing_scales):
                    f.write(f"{path}\n")
            console.print(
                f"[dim]Written {len(missing_scales)} paths to {missing_scales_file}[/dim]"
            )

    return 1 if has_errors else 0


def list_command(args):
    """List existing registry files in the specified directory."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        return

    logger.info(f"Scanning for registry files in: {base_dir}")

    # Find all embeddings_*.txt and landmasks_*.txt files
    registry_files = []
    for file in os.listdir(base_dir):
        if (
            file.startswith("embeddings_") or file.startswith("landmasks_")
        ) and file.endswith(".txt"):
            registry_path = os.path.join(base_dir, file)
            # Count entries in the registry
            try:
                with open(registry_path, "r") as f:
                    entry_count = sum(
                        1 for line in f if line.strip() and not line.startswith("#")
                    )
                registry_files.append((file, entry_count))
            except Exception:
                registry_files.append((file, -1))

    if not registry_files:
        logger.warning("No registry files found")
        return

    # Sort by filename
    registry_files.sort()

    logger.info(f"\nFound {len(registry_files)} registry files:")
    for filename, count in registry_files:
        if count >= 0:
            logger.info(f"  - {filename}: {count:,} entries")
        else:
            logger.warning(f"  - {filename}: (error reading file)")

    # Check for master registry
    master_registry = os.path.join(base_dir, "registry.txt")
    if os.path.exists(master_registry):
        logger.info("\nMaster registry found: registry.txt")


def process_grid_checksum(args):
    """Process a single grid directory to generate SHA256 checksums.

    Only recalculates checksums if:
    - SHA256 file doesn't exist, OR
    - force=True, OR
    - Any .npy file has mtime newer than the SHA256 file
    """
    year_dir, grid_name, force, dry_run = args
    grid_dir = os.path.join(year_dir, grid_name)
    sha256_file = os.path.join(grid_dir, "SHA256")

    # Find all .npy files in this grid directory
    npy_files = [f for f in os.listdir(grid_dir) if f.endswith(".npy")]

    # If SHA256 file exists and force is not enabled, check mtimes
    if not force and os.path.exists(sha256_file):
        sha256_mtime = os.path.getmtime(sha256_file)

        # Check if any .npy file is newer than the SHA256 file
        needs_update = False
        newer_files = []
        for npy_file in npy_files:
            npy_path = os.path.join(grid_dir, npy_file)
            npy_mtime = os.path.getmtime(npy_path)
            if npy_mtime > sha256_mtime:
                needs_update = True
                newer_files.append((npy_file, npy_mtime - sha256_mtime))

        if not needs_update:
            # All files are older than SHA256 file, skip
            return (grid_name, len(npy_files), True, "skipped", None)
        elif dry_run:
            # In dry-run mode, report what would be updated
            return (grid_name, len(npy_files), True, "would_update", newer_files)

    elif not os.path.exists(sha256_file) and dry_run:
        # No SHA256 file exists - would need to create
        return (grid_name, len(npy_files), True, "would_create", None)

    if dry_run:
        # In dry-run mode, don't actually do anything
        return (grid_name, len(npy_files), True, "dry_run", None)

    if npy_files:
        try:
            # Change to grid directory and run sha256sum
            result = subprocess.run(
                ["sha256sum"] + sorted(npy_files),
                cwd=grid_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            # Write output to SHA256 file
            with open(sha256_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            return (grid_name, len(npy_files), True, None, None)
        except subprocess.CalledProcessError as e:
            return (grid_name, len(npy_files), False, f"CalledProcessError: {e}", None)
        except Exception as e:
            return (grid_name, len(npy_files), False, f"Exception: {e}", None)

    return (grid_name, 0, True, None, None)


def generate_embeddings_checksums(
    base_dir, force=False, dry_run=False, year_filter=None
):
    """Generate SHA256 checksums for .npy files in each embeddings subdirectory.

    Only recalculates checksums for grid directories where:
    - SHA256 file doesn't exist, OR
    - force=True, OR
    - Any .npy file has mtime newer than the SHA256 file

    This optimization makes subsequent runs much faster by only updating
    checksums when files have actually changed.

    Args:
        base_dir: Base directory containing year subdirectories
        force: Force regeneration even if SHA256 files are up to date
        dry_run: Don't actually update files, just report what would be done
        year_filter: Optional year to process (e.g. 2024), or None for all years
    """
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    # Print header info before starting progress display
    if dry_run:
        console.print("[yellow]DRY RUN MODE - no files will be modified[/yellow]")
    console.print("Generating SHA256 checksums for embeddings...")
    if force:
        console.print(
            "[yellow]Force mode enabled - regenerating all checksums[/yellow]"
        )
    else:
        console.print("Using smart update: only recalculating for modified files")

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    console.print(f"Using [cyan]{num_cores}[/cyan] CPU cores for parallel processing")

    # Process each year directory
    year_dirs = []
    for item in os.listdir(base_dir):
        if item.isdigit() and len(item) == 4:  # Year directories
            # Apply year filter if specified
            if year_filter is not None and int(item) != year_filter:
                continue
            year_path = os.path.join(base_dir, item)
            if os.path.isdir(year_path):
                year_dirs.append(item)

    if not year_dirs:
        if year_filter:
            console.print(f"[red]No year directory found for {year_filter}[/red]")
        else:
            console.print("[red]No year directories found[/red]")
        return 1

    total_grids = 0
    processed_grids = 0
    skipped_total = 0
    would_update_total = 0
    would_create_total = 0
    errors = []

    # For dry-run, collect details
    update_details = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for year in sorted(year_dirs):
            year_dir = os.path.join(base_dir, year)

            # Find all grid directories
            grid_dirs = []
            for item in os.listdir(year_dir):
                if item.startswith("grid_"):
                    grid_path = os.path.join(year_dir, item)
                    if os.path.isdir(grid_path):
                        grid_dirs.append(item)

            total_grids += len(grid_dirs)

            # Prepare arguments for parallel processing
            grid_args = [
                (year_dir, grid_name, force, dry_run) for grid_name in sorted(grid_dirs)
            ]

            # Process grid directories in parallel
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_grid_checksum, args): args
                    for args in grid_args
                }

                # Process results with progress bar
                skipped_grids = 0
                would_update_grids = 0
                would_create_grids = 0
                task = progress.add_task(f"Year {year}", total=len(grid_dirs))
                for future in as_completed(futures):
                    grid_name, num_files, success, error_msg, details = future.result()

                    if success:
                        if error_msg == "skipped":
                            skipped_grids += 1
                        elif error_msg == "would_update":
                            would_update_grids += 1
                            if (
                                details and len(update_details) < 20
                            ):  # Limit to first 20 examples
                                update_details.append((year, grid_name, details))
                        elif error_msg == "would_create":
                            would_create_grids += 1
                            if len(update_details) < 20:
                                update_details.append((year, grid_name, "no_sha256"))
                        elif num_files > 0:
                            processed_grids += 1
                    else:
                        errors.append(f"{grid_name}: {error_msg}")

                    progress.update(task, advance=1)

            # Store counters
            skipped_total += skipped_grids
            would_update_total += would_update_grids
            would_create_total += would_create_grids

    # Report any errors (after progress bars are done)
    if errors:
        console.print("\n[red]Errors encountered:[/red]")
        for error in errors[:10]:  # Show first 10 errors
            console.print(f"  [red]- {error}[/red]")
        if len(errors) > 10:
            console.print(f"  [dim]... and {len(errors) - 10} more errors[/dim]")

    # Print summary after progress bars
    console.print()  # Blank line
    if dry_run:
        console.print("=" * 60)
        console.print("[bold cyan]DRY RUN SUMMARY[/bold cyan]")
        console.print("=" * 60)
        console.print(f"Total directories scanned: [cyan]{total_grids:,}[/cyan]")
        console.print(f"  Would skip (up to date): [green]{skipped_total:,}[/green]")
        console.print(
            f"  Would update (files modified): [yellow]{would_update_total:,}[/yellow]"
        )
        console.print(
            f"  Would create (no SHA256): [yellow]{would_create_total:,}[/yellow]"
        )

        if update_details:
            console.print("\n[bold]Example directories that would be updated:[/bold]")
            for year, grid_name, details in update_details[:10]:
                if details == "no_sha256":
                    console.print(
                        f"  [yellow]{year}/{grid_name}[/yellow] - No SHA256 file exists"
                    )
                else:
                    console.print(
                        f"  [yellow]{year}/{grid_name}[/yellow] - {len(details)} files newer than SHA256:"
                    )
                    for npy_file, delta in details[:3]:  # Show first 3 files
                        console.print(f"    - {npy_file} [dim](+{delta:.1f}s)[/dim]")
        console.print("\n[green]No files were modified (dry-run mode)[/green]")
        return 0
    else:
        console.print(
            f"Processed [cyan]{processed_grids:,}[/cyan] / [cyan]{total_grids:,}[/cyan] grid directories"
        )
        if skipped_total > 0:
            console.print(
                f"Skipped [green]{skipped_total:,}[/green] directories (SHA256 files up to date)"
            )
        return 0 if processed_grids > 0 else 1


def process_tiff_chunk(args):
    """Process a chunk of TIFF files to generate SHA256 checksums."""
    base_dir, chunk, chunk_num = args
    temp_file = os.path.join(base_dir, f".SHA256SUM.tmp{chunk_num}")

    try:
        # Run sha256sum on this chunk
        result = subprocess.run(
            ["sha256sum"] + chunk,
            cwd=base_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Write to temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        return (chunk_num, len(chunk), True, None, temp_file)
    except subprocess.CalledProcessError as e:
        return (chunk_num, len(chunk), False, f"CalledProcessError: {e}", temp_file)
    except Exception as e:
        return (chunk_num, len(chunk), False, f"Exception: {e}", temp_file)


def generate_tiff_checksums(base_dir, force=False):
    """Generate SHA256 checksums for TIFF files using chunked parallel processing.

    Only recalculates checksums if:
    - SHA256SUM file doesn't exist, OR
    - force=True, OR
    - Any .tiff file has mtime newer than the SHA256SUM file
    """
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    logger.info("Generating SHA256 checksums for TIFF files...")
    if force:
        logger.info("Force mode enabled - regenerating all checksums")
    else:
        logger.info(
            "Using smart update: only recalculating if files have been modified"
        )

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Using {num_cores} CPU cores for parallel processing")

    # Find all .tiff files
    tiff_files = []
    for file in os.listdir(base_dir):
        if file.endswith(".tiff") or file.endswith(".tif"):
            tiff_files.append(file)

    if not tiff_files:
        logger.warning("No TIFF files found")
        return 1

    # Check if SHA256SUM already exists and force is not enabled
    sha256sum_file = os.path.join(base_dir, "SHA256SUM")
    if not force and os.path.exists(sha256sum_file):
        sha256sum_mtime = os.path.getmtime(sha256sum_file)

        # Check if any TIFF file is newer than the SHA256SUM file
        needs_update = False
        for tiff_file in tiff_files:
            tiff_path = os.path.join(base_dir, tiff_file)
            if os.path.getmtime(tiff_path) > sha256sum_mtime:
                needs_update = True
                logger.debug(f"Found updated file: {tiff_file}")
                break

        if not needs_update:
            logger.info(
                "SHA256SUM file is up to date (all TIFF files are older). Skipping."
            )
            logger.info("Use --force to regenerate anyway.")
            return 0

    # Sort files for consistent ordering
    tiff_files.sort()
    total_files = len(tiff_files)
    logger.info(f"Found {total_files} TIFF files")

    # Process in chunks to avoid command line length limits
    chunk_size = 1000  # Process 1000 files at a time

    # Prepare chunks for parallel processing
    chunks = []
    for i in range(0, total_files, chunk_size):
        chunk = tiff_files[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        chunks.append((base_dir, chunk, chunk_num))

    temp_files = []
    errors = []

    try:
        # Process chunks in parallel
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Computing checksums", total=total_files)

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_tiff_chunk, args): args for args in chunks
                }

                # Process results with progress bar
                results = []
                for future in as_completed(futures):
                    chunk_num, chunk_len, success, error_msg, temp_file = (
                        future.result()
                    )

                    if success:
                        results.append((chunk_num, temp_file))
                    else:
                        errors.append(f"Chunk {chunk_num}: {error_msg}")

                    progress.update(task, advance=chunk_len)

            # Sort results by chunk number to maintain order
            results.sort(key=lambda x: x[0])
            temp_files = [temp_file for _, temp_file in results]

        if errors:
            logger.error("\nErrors encountered during processing:")
            for error in errors:
                logger.error(f"  - {error}")
            # Clean up any temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return 1

        # Concatenate all temporary files into final SHA256SUM
        logger.info("Concatenating results...")
        with open(sha256sum_file, "w", encoding="utf-8") as outfile:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        logger.info(f"Successfully generated checksums for {total_files} files")
        logger.info(f"Checksums written to: {sha256sum_file}")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Clean up any temporary files created so far (chunks holds 3-tuples, not
        # paths; temp_files is the list of actual temp paths, as used above).
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return 1


def scan_command(args):
    """Scan SHA256 checksum files and generate parquet databases."""
    console = Console()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Error: Directory {base_dir} does not exist[/red]")
        return 1

    console.print(
        Panel.fit(f"🔍 Scanning Registry Data\n📁 {base_dir}", style="bold blue")
    )

    # Determine output directory
    if hasattr(args, "registry_dir") and args.registry_dir:
        output_dir = os.path.abspath(args.registry_dir)
    else:
        output_dir = base_dir

    # Look for expected directories
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    only = getattr(args, "only", None)

    # Create embeddings Parquet database
    embeddings_parquet_path = os.path.join(output_dir, "registry.parquet")
    if only in (None, "embeddings"):
        if os.path.exists(repr_dir):
            if not create_parquet_database_from_filesystem(
                base_dir, embeddings_parquet_path, console
            ):
                console.print("[red]Failed to create embeddings parquet database[/red]")
                return 1
        else:
            console.print(
                f"[red]Error: Embeddings directory not found: {repr_dir}[/red]"
            )
            return 1

    # Create landmasks Parquet database
    landmasks_parquet_path = os.path.join(output_dir, "landmasks.parquet")
    if only in (None, "landmasks"):
        if os.path.exists(tiles_dir):
            if not create_landmasks_parquet_database(
                tiles_dir, landmasks_parquet_path, console
            ):
                console.print(
                    "[yellow]Warning: Failed to create landmasks parquet database[/yellow]"
                )
                # Don't return error, landmasks are optional
        else:
            console.print(
                f"[yellow]Warning: Landmasks directory not found: {tiles_dir}[/yellow]"
            )

    # Show final summary
    summary_lines = ["[green]✅ Registry Generation Complete[/green]\n"]
    summary_lines.append("📊 Generated outputs:")
    summary_lines.append("• Parquet databases:")
    if only in (None, "embeddings"):
        summary_lines.append(f"  → {embeddings_parquet_path} (embeddings)")
    if only in (None, "landmasks") and os.path.exists(landmasks_parquet_path):
        summary_lines.append(f"  → {landmasks_parquet_path} (landmasks)")
    summary_lines.append(f"📁 Output directory: {output_dir}")

    console.print(Panel.fit("\n".join(summary_lines), style="green"))

    return 0


def hash_command(args):
    """Generate SHA256 checksums for embeddings and TIFF files."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Directory {base_dir} does not exist[/red]")
        return 1

    force = getattr(args, "force", False)
    dry_run = getattr(args, "dry_run", False)
    year_filter = getattr(args, "year", None)

    # Check if this is an embeddings directory structure
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    processed_any = False

    # Process embeddings if directory exists
    if os.path.exists(repr_dir):
        console.print(f"\n[bold]Processing embeddings directory:[/bold] {repr_dir}")
        if year_filter:
            console.print(f"[cyan]Filtering to year:[/cyan] {year_filter}")
        if (
            generate_embeddings_checksums(
                repr_dir, force=force, dry_run=dry_run, year_filter=year_filter
            )
            == 0
        ):
            processed_any = True

    # Process TIFF files if directory exists (no year filtering for TIFF files)
    if os.path.exists(tiles_dir) and year_filter is None:
        console.print(f"\n[bold]Processing TIFF directory:[/bold] {tiles_dir}")
        if generate_tiff_checksums(tiles_dir, force=force) == 0:
            processed_any = True
    elif os.path.exists(tiles_dir) and year_filter is not None:
        console.print(
            "[dim]Skipping TIFF processing (year filter only applies to embeddings)[/dim]"
        )

    if not processed_any:
        console.print("[red]No data directories found. Expected:[/red]")
        console.print(f"[red]  - {repr_dir}[/red]")
        console.print(f"[red]  - {tiles_dir}[/red]")
        return 1

    return 0


def analyze_registry_changes():
    """Analyze git changes in registry files and summarize by year."""
    try:
        # Get all changed registry files (staged, unstaged, and untracked)
        registry_files_changed = set()

        # Single git status call to get all changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Parse git status output: XY filename (where XY is 2-char status)
            if len(line) >= 3:
                # Git status porcelain format: first 2 chars are status, then filename
                # Handle both "XY filename" and "XY  filename" formats
                filename = line[2:].lstrip()  # Skip status and any spaces

                if is_registry_file(filename):
                    registry_files_changed.add(filename)

        if not registry_files_changed:
            return {}, []

        # Analyze each changed file
        changes_by_year = defaultdict(lambda: {"added": 0, "modified": 0})
        registry_files_list = []

        for file_path in registry_files_changed:
            if not os.path.exists(file_path):
                continue

            year = extract_year_from_filename(file_path)
            is_new_file = is_untracked_file(file_path)

            if is_new_file:
                entries = count_entries_in_registry_file(file_path)
                changes_by_year[year]["added"] += entries
                registry_files_list.append(("A", file_path))
            else:
                added, removed = count_file_diff_entries(file_path)
                net_change = added - removed
                if net_change > 0:
                    changes_by_year[year]["added"] += net_change
                elif net_change < 0:
                    changes_by_year[year]["modified"] += abs(net_change)
                registry_files_list.append(("M", file_path))

        return changes_by_year, registry_files_list

    except subprocess.CalledProcessError as e:
        return None, f"Git command failed: {e}"
    except Exception as e:
        return None, f"Error analyzing changes: {e}"


def extract_year_from_filename(file_path):
    """Extract year from registry filename.

    Registry filenames follow these exact patterns:
    - embeddings_YYYY_lonX_latY.txt -> returns YYYY
    - landmasks_lonX_latY.txt -> returns None (no year)
    - registry_YYYY.txt -> returns YYYY
    - registry.txt -> returns None (master registry)
    """
    filename = os.path.basename(file_path)

    # embeddings_YYYY_lonX_latY.txt
    if filename.startswith("embeddings_"):
        parts = filename.split("_")
        if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 4:
            return int(parts[1])

    # registry_YYYY.txt
    elif filename.startswith("registry_") and filename.endswith(".txt"):
        year_str = filename[9:-4]  # Extract between 'registry_' and '.txt'
        if year_str.isdigit() and len(year_str) == 4:
            return int(year_str)

    # landmasks and master registry have no year
    return None


def is_registry_file(file_path):
    """Check if a file is a registry file we should analyze."""
    if not file_path.endswith(".txt"):
        return False

    filename = os.path.basename(file_path)
    return any(
        keyword in filename for keyword in ["registry", "embeddings", "landmasks"]
    )


def count_entries_in_registry_file(file_path):
    """Count the number of entries in a registry file."""
    if not os.path.exists(file_path):
        return 0

    try:
        with open(file_path, "r") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
            return count
    except Exception:
        return 0


def is_untracked_file(file_path):
    """Check if a file is untracked (new)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", file_path],
            capture_output=True,
            text=True,
        )
        return result.returncode != 0  # Non-zero means untracked
    except Exception:
        return False


def count_file_diff_entries(file_path):
    """Count added and removed entries in a modified registry file."""
    try:
        # Get the diff for this file
        result = subprocess.run(
            ["git", "diff", "HEAD", file_path],
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            return 0, 0  # No changes

        added = 0
        removed = 0

        for line in result.stdout.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                # Count non-comment, non-empty lines
                content = line[1:].strip()  # Remove the '+' prefix
                if content and not content.startswith("#"):
                    added += 1
            elif line.startswith("-") and not line.startswith("---"):
                # Count non-comment, non-empty lines
                content = line[1:].strip()  # Remove the '-' prefix
                if content and not content.startswith("#"):
                    removed += 1

        return added, removed

    except Exception:
        return 0, 0


def create_commit_message(changes_by_year):
    """Create a concise commit message from the changes analysis."""

    # Calculate totals
    total_added = sum(year_data["added"] for year_data in changes_by_year.values())
    total_modified = sum(
        year_data["modified"] for year_data in changes_by_year.values()
    )

    # Create summary line
    summary_parts = []
    if total_added > 0:
        summary_parts.append(f"{total_added} tiles added")
    if total_modified > 0:
        summary_parts.append(f"{total_modified} tiles modified")

    summary = (
        f"Update registry: {', '.join(summary_parts)}"
        if summary_parts
        else "Update registry files"
    )

    # Create concise message with year breakdown
    message_parts = [summary]

    if (
        changes_by_year and len(changes_by_year) > 1
    ):  # Only show breakdown if multiple years
        message_parts.append("")
        for year in sorted(y for y in changes_by_year.keys() if y is not None):
            year_data = changes_by_year[year]
            changes = []
            if year_data["added"] > 0:
                changes.append(f"+{year_data['added']}")
            if year_data["modified"] > 0:
                changes.append(f"~{year_data['modified']}")
            if changes:
                message_parts.append(f"{year}: {', '.join(changes)}")

    return "\n".join(message_parts)


def commit_command(args):
    """Analyze registry changes and create a commit with summary."""
    console = Console()

    # Check if we're in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, check=True
        )
    except subprocess.CalledProcessError:
        console.print("[red]Error: Not in a git repository[/red]")
        console.print("[dim]Run this command from within a git repository[/dim]")
        return 1

    console.print(Panel.fit("📊 Analyzing Registry Changes", style="cyan"))

    # Check if git is properly configured
    try:
        subprocess.run(["git", "config", "user.name"], capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        console.print(
            "[red]Error: Git user.name and user.email must be configured[/red]"
        )
        console.print(
            "[dim]Run: git config user.name 'Your Name' && git config user.email 'your@email.com'[/dim]"
        )
        return 1

    # Analyze changes
    changes_by_year, registry_files_changed = analyze_registry_changes()

    if changes_by_year is None:
        console.print(f"[red]Error analyzing changes: {registry_files_changed}[/red]")
        return 1

    if not registry_files_changed:
        console.print("[yellow]No registry file changes detected[/yellow]")
        console.print(
            "[dim]Only .txt files with 'registry', 'embeddings', or 'landmasks' in the name are analyzed[/dim]"
        )
        return 0

    # Display summary
    console.print(
        f"[green]Found {len(registry_files_changed)} registry files with changes[/green]"
    )

    # Display file-by-file breakdown
    console.print("\n[blue]Changed files:[/blue]")
    for status, file_path in registry_files_changed[:10]:  # Show first 10
        status_str = {
            "A": "[green]Added[/green]",
            "M": "[yellow]Modified[/yellow]",
            "D": "[red]Deleted[/red]",
        }.get(status, status)
        console.print(f"  {status_str}: {file_path}")

    if len(registry_files_changed) > 10:
        console.print(
            f"  [dim]... and {len(registry_files_changed) - 10} more files[/dim]"
        )

    if changes_by_year:
        console.print("\n[blue]Summary by year:[/blue]")
        total_added = 0
        total_modified = 0

        for year in sorted(changes_by_year.keys(), key=lambda x: (x is None, x)):
            year_str = str(year) if year else "unknown"
            year_data = changes_by_year[year]

            change_parts = []
            if year_data["added"] > 0:
                change_parts.append(f"[green]+{year_data['added']} tiles[/green]")
                total_added += year_data["added"]
            if year_data["modified"] > 0:
                change_parts.append(
                    f"[yellow]~{year_data['modified']} modified[/yellow]"
                )
                total_modified += year_data["modified"]

            if change_parts:
                console.print(f"  {year_str}: {', '.join(change_parts)}")
        console.print(
            f"\n[bold]Total: [green]+{total_added} added[/green]"
            + (
                f" / [yellow]~{total_modified} modified[/yellow]"
                if total_modified > 0
                else ""
            )
            + " tiles[/bold]"
        )

    # Validate we have something to commit
    if all(
        year_data["added"] == 0 and year_data["modified"] == 0
        for year_data in changes_by_year.values()
    ):
        console.print(
            "[yellow]Warning: No tile changes detected in registry files[/yellow]"
        )
        console.print(
            "[dim]Files may have been reformatted without content changes[/dim]"
        )

    # Stage registry files
    console.print("\n[blue]Staging registry files...[/blue]")
    staged_files = []
    failed_files = []

    for status, file_path in registry_files_changed:
        if os.path.exists(file_path):  # Only add files that exist
            try:
                subprocess.run(
                    ["git", "add", file_path], check=True, capture_output=True
                )
                staged_files.append(file_path)
            except subprocess.CalledProcessError as e:
                failed_files.append((file_path, str(e)))

    if staged_files:
        console.print(
            f"[green]{emoji('✓ ')}Staged {len(staged_files)} files successfully[/green]"
        )

    if failed_files:
        console.print(
            f"[yellow]{emoji('⚠ ')}Failed to stage {len(failed_files)} files:[/yellow]"
        )
        for file_path, error in failed_files[:3]:  # Show first 3 failures
            console.print(f"  {file_path}: {error}")
        if len(failed_files) > 3:
            console.print(f"  [dim]... and {len(failed_files) - 3} more failures[/dim]")

    # Check if there's anything staged
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        if not result.stdout.strip():
            console.print("[yellow]No files staged for commit[/yellow]")
            return 1
    except subprocess.CalledProcessError:
        console.print("[red]Error checking staged files[/red]")
        return 1

    # Create commit message
    commit_message = create_commit_message(changes_by_year)

    console.print("\n[blue]Commit message:[/blue]")
    console.print(Panel(commit_message, style="dim"))

    # Create the commit
    console.print("[blue]Creating commit...[/blue]")
    try:
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=True,
        )

        console.print(f"[green]{emoji('✓ ')}Commit created successfully[/green]")

        # Show the commit hash and stats
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit_hash = commit_result.stdout.strip()[:8]
        console.print(f"[cyan]Commit: {commit_hash}[/cyan]")

        # Show commit stats
        if result.stdout.strip():
            console.print(f"[dim]{result.stdout.strip()}[/dim]")

        return 0

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error creating commit: {e}[/red]")
        if e.stderr:
            console.print(
                f"[dim]Git error: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}[/dim]"
            )
        return 1


def export_manifests_command(args):
    """Convert Parquet registry files to Pooch-format text manifests.

    This command reads registry.parquet and landmasks.parquet files and exports
    them to the old block-based text registry format for backwards compatibility.
    Useful for maintaining the tessera-manifests repository.
    """
    console = Console()

    # Resolve input directory
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        return 1

    # Find Parquet files
    registry_parquet = input_dir / "registry.parquet"
    landmasks_parquet = input_dir / "landmasks.parquet"

    if not registry_parquet.exists():
        console.print(f"[red]Error: registry.parquet not found in {input_dir}[/red]")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir / "registry"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold blue]📦 Converting Parquet to Text Manifests[/bold blue]\n"
            f"📁 Input: {input_dir}\n"
            f"📁 Output: {output_dir}",
            style="blue",
        )
    )

    total_files_written = 0

    # ========== Process Embeddings ==========
    console.print("\n[cyan]Processing embeddings registry...[/cyan]")

    try:
        df = pd.read_parquet(registry_parquet)
        console.print(f"  Loaded [green]{len(df):,}[/green] embedding records")

        # Verify required columns
        required_cols = ["lat", "lon", "year", "hash", "scales_hash"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            console.print(
                f"[red]Error: Missing required columns in registry.parquet: {missing}[/red]"
            )
            console.print(f"[yellow]Available columns: {df.columns.tolist()}[/yellow]")
            console.print(
                "[yellow]Hint: Regenerate registry.parquet with latest geotessera-registry scan[/yellow]"
            )
            return 1

        # Group by year and block
        files_by_year_and_block = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            lon, lat, year = row["lon"], row["lat"], row["year"]
            embedding_hash = row["hash"]
            scales_hash = row["scales_hash"]

            # Calculate block coordinates
            block_lon, block_lat = block_from_world(lon, lat)

            # Construct file paths (matching the structure in data directories)
            grid_name = f"grid_{lon:.2f}_{lat:.2f}"
            embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
            scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"

            # Add both files to the block with their respective hashes
            files_by_year_and_block[year][(block_lon, block_lat)].append(
                (embedding_path, embedding_hash)
            )
            files_by_year_and_block[year][(block_lon, block_lat)].append(
                (scales_path, scales_hash)
            )

        # Write embeddings registry files
        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        for year in sorted(files_by_year_and_block.keys()):
            blocks = files_by_year_and_block[year]
            console.print(f"  Year {year}: {len(blocks)} blocks")

            for (block_lon, block_lat), entries in sorted(blocks.items()):
                registry_filename = block_to_embeddings_registry_filename(
                    str(year), block_lon, block_lat
                )
                registry_file = embeddings_dir / registry_filename

                # Write file atomically
                import tempfile

                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        dir=embeddings_dir,
                        prefix=f".{registry_filename}_tmp_",
                        suffix=".txt",
                        delete=False,
                    ) as temp_file:
                        temp_path = temp_file.name
                        for rel_path, checksum in sorted(entries):
                            temp_file.write(f"{rel_path} {checksum}\n")

                    os.rename(temp_path, registry_file)
                    temp_path = None
                    total_files_written += 1

                except Exception:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise

                console.print(
                    f"    Block ({block_lon:4d}, {block_lat:4d}): {len(entries):4d} files → {registry_filename}"
                )

        console.print(
            f"[green]{emoji('✓ ')}Wrote {total_files_written} embeddings registry files[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error processing embeddings: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

    # ========== Process Landmasks ==========
    if landmasks_parquet.exists():
        console.print("\n[cyan]Processing landmasks registry...[/cyan]")

        try:
            df = pd.read_parquet(landmasks_parquet)
            console.print(f"  Loaded [green]{len(df):,}[/green] landmask records")

            # Verify required columns
            required_cols = ["lat", "lon", "hash"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                console.print(
                    f"[yellow]Warning: Missing columns in landmasks.parquet: {missing}[/yellow]"
                )
            else:
                # Group by block
                files_by_block = defaultdict(list)

                for _, row in df.iterrows():
                    lon, lat = row["lon"], row["lat"]
                    sha256 = row["hash"]

                    # Calculate block coordinates
                    block_lon, block_lat = block_from_world(lon, lat)

                    # Construct file path
                    filename = f"grid_{lon:.2f}_{lat:.2f}.tiff"

                    files_by_block[(block_lon, block_lat)].append((filename, sha256))

                # Write landmask registry files
                landmasks_dir = output_dir / "landmasks"
                landmasks_dir.mkdir(parents=True, exist_ok=True)

                landmask_files_written = 0
                console.print(f"  Writing {len(files_by_block)} landmask blocks")

                for (block_lon, block_lat), entries in sorted(files_by_block.items()):
                    registry_filename = block_to_landmasks_registry_filename(
                        block_lon, block_lat
                    )
                    registry_file = landmasks_dir / registry_filename

                    # Write file atomically
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            dir=landmasks_dir,
                            prefix=f".{registry_filename}_tmp_",
                            suffix=".txt",
                            delete=False,
                        ) as temp_file:
                            temp_path = temp_file.name
                            for rel_path, checksum in sorted(entries):
                                temp_file.write(f"{rel_path} {checksum}\n")

                        os.rename(temp_path, registry_file)
                        temp_path = None
                        landmask_files_written += 1

                    except Exception:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise

                    console.print(
                        f"    Block ({block_lon:4d}, {block_lat:4d}): {len(entries):4d} files → {registry_filename}"
                    )

                console.print(
                    f"[green]{emoji('✓ ')}Wrote {landmask_files_written} landmask registry files[/green]"
                )
                total_files_written += landmask_files_written

        except Exception as e:
            console.print(f"[yellow]Warning: Error processing landmasks: {e}[/yellow]")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    else:
        console.print(
            f"[yellow]Landmasks parquet not found, skipping: {landmasks_parquet}[/yellow]"
        )

    # Summary
    console.print(
        Panel.fit(
            f"[green]✅ Export Complete[/green]\n"
            f"📊 Total registry files written: {total_files_written}\n"
            f"📁 Output directory: {output_dir}",
            style="green",
        )
    )

    return 0


def file_scan_command(args):
    """Recursively scan year directories for embedding tiles and generate a parquet inventory.

    This command scans an input directory expecting year subdirectories (e.g., 2024/, 2023/).
    Within each year directory, it recursively searches for embedding tiles (identified by
    grid*.npy and grid*_scales.npy files). It extracts coordinates from filenames and records
    modification times for both files, along with the year.

    Useful for finding potential duplicate embeddings across machines.
    """
    import re
    from datetime import datetime

    console = Console()

    # Resolve input directory
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output).resolve()
    else:
        output_file = input_dir / "embedding_inventory.parquet"

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Scanning for Embedding Tiles[/bold blue]\n"
            f"📁 Input: {input_dir}\n"
            f"📄 Output: {output_file}",
            style="blue",
        )
    )

    # Pattern to match grid files
    grid_pattern = re.compile(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)\.npy$")

    # Collect data
    records = []
    processed_dirs = set()
    years_found = set()

    console.print("\n[cyan]Scanning for year directories...[/cyan]")

    # First, identify year directories (subdirectories that are numeric years)
    year_dirs = []
    try:
        for item in input_dir.iterdir():
            if item.is_dir():
                # Check if directory name is a valid year (4 digits)
                if re.match(r"^\d{4}$", item.name):
                    year = int(item.name)
                    year_dirs.append((year, item))
                    console.print(f"  Found year directory: [green]{item.name}[/green]")
    except Exception as e:
        console.print(f"[red]Error scanning input directory: {e}[/red]")
        return 1

    if not year_dirs:
        console.print(
            "[yellow]No year directories found! Expected directories like 2024/, 2023/, etc.[/yellow]"
        )
        return 1

    console.print(
        f"\n[cyan]Scanning {len(year_dirs)} year directories for embedding tiles...[/cyan]"
    )

    # Walk each year directory
    for year, year_dir in sorted(year_dirs):
        console.print(f"\n[dim]Processing year {year}...[/dim]")

        for root, dirs, files in os.walk(year_dir):
            # Look for grid*.npy files in this directory
            grid_files = [f for f in files if grid_pattern.match(f)]

            if not grid_files:
                continue

            # Process each grid file found
            for grid_file in grid_files:
                match = grid_pattern.match(grid_file)
                if not match:
                    continue

                lon = float(match.group(1))
                lat = float(match.group(2))

                # Construct paths
                grid_name = f"grid_{lon:.2f}_{lat:.2f}"
                grid_path = os.path.join(root, f"{grid_name}.npy")
                scales_path = os.path.join(root, f"{grid_name}_scales.npy")

                # Check if both files exist
                if not os.path.exists(grid_path):
                    console.print(f"[yellow]Warning: Missing {grid_path}[/yellow]")
                    continue

                if not os.path.exists(scales_path):
                    console.print(
                        f"[yellow]Warning: Missing scales file for {grid_path}[/yellow]"
                    )
                    continue

                # Get modification times and full real paths
                try:
                    grid_stat = os.stat(grid_path)
                    scales_stat = os.stat(scales_path)

                    grid_mtime = datetime.fromtimestamp(grid_stat.st_mtime)
                    scales_mtime = datetime.fromtimestamp(scales_stat.st_mtime)
                    grid_size = grid_stat.st_size
                    scales_size = scales_stat.st_size

                    # Get full real paths
                    grid_realpath = os.path.realpath(grid_path)
                    scales_realpath = os.path.realpath(scales_path)

                    # Record the information
                    records.append(
                        {
                            "year": year,
                            "lon": lon,
                            "lat": lat,
                            "directory": root,
                            "grid_path": grid_realpath,
                            "scales_path": scales_realpath,
                            "grid_mtime": grid_mtime,
                            "scales_mtime": scales_mtime,
                            "grid_size": grid_size,
                            "scales_size": scales_size,
                        }
                    )

                    # Track unique directories processed
                    processed_dirs.add(root)
                    years_found.add(year)

                except Exception as e:
                    console.print(f"[red]Error processing {grid_path}: {e}[/red]")
                    continue

    if not records:
        console.print("[yellow]No embedding tiles found![/yellow]")
        return 1

    # Create DataFrame
    console.print(
        f"\n[cyan]Creating parquet file with {len(records):,} tiles...[/cyan]"
    )
    df = pd.DataFrame(records)

    # Sort by year, lon, lat for easier analysis
    df = df.sort_values(["year", "lon", "lat"])

    # Add integer grid indices for robust cross-platform lookups
    df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
    df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

    # Save to parquet (atomic write via temp file + rename)
    import tempfile

    temp_path = None
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_file.parent,
            prefix=f".{output_file.name}_tmp_",
            suffix=".parquet",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        os.chmod(temp_path, 0o644)
        df.to_parquet(temp_path, index=False)
        os.rename(temp_path, str(output_file))

        console.print(
            Panel.fit(
                f"[green]✅ Scan Complete[/green]\n"
                f"📊 Tiles found: {len(records):,}\n"
                f"📅 Years: {', '.join(str(y) for y in sorted(years_found))}\n"
                f"📁 Unique directories: {len(processed_dirs):,}\n"
                f"📄 Output: {output_file}",
                style="green",
            )
        )

        # Show sample of data
        console.print("\n[cyan]Sample of collected data:[/cyan]")
        table = Table(show_header=True)
        table.add_column("Year", style="magenta")
        table.add_column("Lon", style="cyan")
        table.add_column("Lat", style="cyan")
        table.add_column("Grid mtime", style="yellow")
        table.add_column("Scales mtime", style="yellow")
        table.add_column("Grid path", style="dim")

        for _, row in df.head(5).iterrows():
            table.add_row(
                str(row["year"]),
                f"{row['lon']:.2f}",
                f"{row['lat']:.2f}",
                row["grid_mtime"].strftime("%Y-%m-%d %H:%M:%S"),
                row["scales_mtime"].strftime("%Y-%m-%d %H:%M:%S"),
                str(row["grid_path"])[:50] + "..."
                if len(str(row["grid_path"])) > 50
                else str(row["grid_path"]),
            )

        console.print(table)

        return 0

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        console.print(f"[red]Error writing parquet file: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse an s3://bucket/prefix URI into (bucket, prefix).

    The prefix is normalised to have no leading slash and a single trailing
    slash so it can be used directly as a ListObjectsV2 prefix.
    """
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix URI, got: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _s3_list(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    delimiter: Optional[str] = None,
) -> Iterator[Tuple[str, int, str]]:
    """Yield (key, size, last_modified) for every object under prefix.

    Issues anonymous path-style HTTPS calls to the S3-compatible
    ListObjectsV2 endpoint at *endpoint_url* (e.g.
    ``https://data.source.coop``). Follows continuation tokens until the
    listing is exhausted. If delimiter is set, yields CommonPrefixes as
    (prefix, 0, "") tuples instead of object contents.

    Requests go through the shared pool (``registry._http``), which retries
    each page whole, before any of its entries are yielded, so the stream can
    never contain duplicates. A page that fails for good raises.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import urlencode

    from .registry import _http

    base = f"{endpoint_url.rstrip('/')}/{bucket}/"
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter is not None:
            params["delimiter"] = delimiter
        if token is not None:
            params["continuation-token"] = token
        url = base + "?" + urlencode(params)

        resp = _http().request("GET", url)
        if resp.status != 200:
            raise RuntimeError(f"S3 listing failed: HTTP {resp.status} for {url}")
        root = ET.fromstring(resp.data)

        if delimiter is not None:
            for cp in root.findall(f"{S3_NS}CommonPrefixes"):
                p = cp.findtext(f"{S3_NS}Prefix") or ""
                if p:
                    yield p, 0, ""
        else:
            for c in root.findall(f"{S3_NS}Contents"):
                key = c.findtext(f"{S3_NS}Key") or ""
                size = int(c.findtext(f"{S3_NS}Size") or "0")
                lm = c.findtext(f"{S3_NS}LastModified") or ""
                if key:
                    yield key, size, lm

        if (root.findtext(f"{S3_NS}IsTruncated") or "false") != "true":
            return
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            return


# Bucket-layout regexes shared by the s3scan discovery walker.
#   Datasets live as top-level dirs named after the version, optionally
#   with a ``-<variant>`` suffix: ``v1/`` (all 1.0 variants), ``v1.1-cam/``
#   (1.1/cambridge), ``v2-2B-L~beta1/`` (matched by _DATASET_DIR_RE from
#   .registry; a bare ``v1.1/`` legacy dir is also accepted).
#   In the legacy layout, variants live under a version dir as
#   ``global_0.1_degree_representation`` (the default ``vultr`` variant) or
#   ``global_0.1_degree_representation.<name>`` (e.g. ``.cambridge``).
#   Years are 4-digit directories under a dataset (or legacy variant) dir.
_VERSION_RE = re.compile(r"^v(\d+)(?:\.(\d+))?$")
_VARIANT_RE = re.compile(r"^global_0\.1_degree_representation(?:\.([\w-]+))?$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_DEFAULT_VARIANT = "vultr"


def _normalize_version(version_dir: str) -> str:
    """``v1`` → ``1.0``; ``v1.1`` → ``1.1``."""
    m = _VERSION_RE.match(version_dir)
    if not m:
        return version_dir
    return f"{m.group(1)}.{m.group(2) or '0'}"


def _discover_scan_units(
    bucket: str,
    root_prefix: str,
    endpoint_url: str,
    console: "Console",
    flat_variant: Optional[str] = None,
) -> List[Tuple[str, str, int, str]]:
    """Walk an S3 prefix and return a list of (version, variant, year, year_prefix).

    Auto-detects the level of ``root_prefix``: tree root, dataset dir, or
    legacy variant dir. Version/variant components already present in the
    supplied prefix are inferred from the path and don't need to be
    discovered again.

    Supports three layouts:

    * the dataset-dir layout (``{version}-{variant_suffix}/{year}/``, e.g.
      ``v1.1-cam/2024/``) where the directory name encodes the variant;
    * the flat layout (``{version}/{year}/``, e.g. ``v1/2024/``) which
      carries no variant in the path — rows are attributed to
      *flat_variant*, or to the version's default variant when that is
      None;
    * the legacy variant-subdir layout
      (``{version}/global_0.1_degree_representation[.<variant>]/{year}/``).
    """
    # Infer any version/variant already encoded in the path.
    pre_version = None
    pre_variant = None
    for part in root_prefix.rstrip("/").split("/"):
        m = _DATASET_DIR_RE.match(part)
        if m:
            parsed = dataset_from_path(part)
            if parsed is not None:
                pre_version, ds_variant = parsed
                # A -variant suffix pins the variant; a bare version dir
                # leaves it open for the legacy variant-subdir layout.
                if m.group(3) is not None:
                    pre_variant = ds_variant
            continue
        m = _VARIANT_RE.match(part)
        if m:
            pre_variant = m.group(1) or _DEFAULT_VARIANT

    units: List[Tuple[str, str, int, str]] = []

    def walk(prefix: str, version: Optional[str], variant: Optional[str], indent: int):
        pad = "  " * indent
        for sp, _, _ in _s3_list(bucket, prefix, endpoint_url, delimiter="/"):
            tail = sp[len(prefix) :].rstrip("/")
            if version is None:
                m = _DATASET_DIR_RE.match(tail)
                if m and dataset_from_path(tail) is not None:
                    v, ds_variant = dataset_from_path(tail)
                    if m.group(3) is not None:
                        # Suffixed dataset dir: variant comes from the name.
                        console.print(
                            f"{pad}Dataset: [green]{tail}[/green] (= {v}/{ds_variant})"
                        )
                        walk(sp, v, ds_variant, indent + 1)
                    else:
                        console.print(f"{pad}Version: [green]{tail}[/green] (= {v})")
                        walk(sp, v, variant, indent + 1)
                    continue
            if variant is None:
                m = _VARIANT_RE.match(tail)
                if m:
                    var = m.group(1) or _DEFAULT_VARIANT
                    console.print(f"{pad}Variant: [green]{tail}[/green] (= {var})")
                    walk(sp, version, var, indent + 1)
                    continue
            m = _YEAR_RE.match(tail)
            if m and version is not None:
                # A year directly under a bare version dir is the flat
                # layout; under a suffixed dataset dir the variant is
                # already bound.
                var = variant or flat_variant or default_variant(version)
                units.append((version, var, int(m.group(1)), sp))

    walk(root_prefix, pre_version, pre_variant, 1)
    return units


_LANDMASK_DIR = "global_0.1_degree_tiff_all"
_LANDMASK_RE = re.compile(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)\.tiff$")


def _discover_landmask_prefixes(
    bucket: str,
    root_prefix: str,
    endpoint_url: str,
    console: "Console",
) -> List[Tuple[str, str, str]]:
    """Return ``(version_norm, version_path, landmask_prefix)`` per version.

    Supports both layouts: TIFFs under ``{version}/global_0.1_degree_tiff_all/``
    and TIFFs directly under the version directory (the Source Cooperative
    ``landmasks/{version}/`` tree). Versions with no TIFFs at either location
    are skipped. When ``root_prefix`` already points at or inside a version
    dir, only that version is probed.
    """
    candidates: List[Tuple[str, str]] = []  # (version_norm, version_prefix)

    pre_version = None
    pre_version_path = None
    for part in root_prefix.rstrip("/").split("/"):
        m = _VERSION_RE.match(part)
        if m:
            pre_version = _normalize_version(part)
            pre_version_path = part

    if pre_version is not None:
        # Find the version-level prefix inside root_prefix.
        idx = root_prefix.find(f"/{pre_version_path}/")
        if idx >= 0:
            version_prefix = root_prefix[: idx + 1] + pre_version_path + "/"
        else:
            # root_prefix IS the version dir (no extra components after it).
            version_prefix = root_prefix
        candidates.append((pre_version, version_prefix))
    else:
        for sp, _, _ in _s3_list(bucket, root_prefix, endpoint_url, delimiter="/"):
            tail = sp[len(root_prefix) :].rstrip("/")
            m = _VERSION_RE.match(tail)
            if m:
                candidates.append((_normalize_version(tail), sp))

    from .registry import _version_path_from_norm

    from itertools import islice

    out: List[Tuple[str, str, str]] = []
    for version_norm, version_prefix in candidates:
        version_path = _version_path_from_norm(version_norm)
        # Probe a few keys at each candidate location. TIFF keys sort before
        # any parquet sidecars, so the first page suffices.
        for lm_prefix in (version_prefix + _LANDMASK_DIR + "/", version_prefix):
            probe = islice(_s3_list(bucket, lm_prefix, endpoint_url), 20)
            if any(key.endswith(".tiff") for key, _, _ in probe):
                console.print(f"  Landmasks: [green]{lm_prefix}[/green]")
                out.append((version_norm, version_path, lm_prefix))
                break
    return out


def s3scan_command(args):
    """Spider an S3 bucket for embedding tiles and write a manifest parquet.

    The input URI may point at any level of the bucket layout:

    * ``s3://bucket/tessera/npy/`` — discover all datasets and years
    * ``s3://bucket/tessera/npy/v1.1-cam/`` — one dataset (the directory
      name encodes the (version, variant) pair)
    * ``s3://bucket/v1.1/global_0.1_degree_representation.cambridge/`` —
      one (version, variant) in the legacy variant-subdir layout

    Each discovered (version, variant, year) is then listed in parallel
    using integer-longitude shards. One ``manifest.parquet`` is written per
    dataset directory under ``{output_dir}/{dataset_dir}/manifest.parquet``
    (e.g. ``v1/``, ``v1.1-cam/``, ``v2-2B-L~beta1/``), mirroring the npy/
    tree so the output can be uploaded with ``aws s3 cp --recursive``.
    Each manifest carries ``version`` and ``variant`` columns.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime

    console = Console()

    try:
        bucket, prefix = _parse_s3_uri(args.s3_uri)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    endpoint = args.endpoint_url
    if not endpoint:
        console.print("[red]Error: --endpoint-url must not be empty[/red]")
        return 1

    # Output is a *directory* under which per-version manifests are written
    # as ``{output_dir}/{version_path}/manifest.parquet``. This mirrors the S3
    # layout so the whole tree can be uploaded with ``aws s3 cp --recursive``.
    output_dir = Path(args.output).resolve() if args.output else Path.cwd()

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Spidering S3 for Embedding Tiles[/bold blue]\n"
            f"📦 Bucket: {bucket}\n"
            f"🔑 Prefix: {prefix or '(root)'}\n"
            f"🌐 Endpoint: {endpoint}\n"
            f"📂 Output dir: {output_dir}",
            style="blue",
        )
    )

    # With an explicit --landmasks-uri the landmask scan can proceed even
    # when the embeddings prefix yields nothing (a landmasks-only run).
    landmasks_only_ok = not args.no_landmasks and args.landmasks_uri is not None

    scan_units: List[Tuple[str, str, int, str]] = []
    if landmasks_only_ok and args.landmasks_uri == args.s3_uri:
        # A pure landmasks run. Skip embedding discovery: walking a flat
        # landmask tree pages through every TIFF key to find no year dirs.
        console.print("\n[cyan]Landmasks-only run; skipping embedding scan.[/cyan]")
    else:
        console.print("\n[cyan]Discovering versions, variants, and years...[/cyan]")
        try:
            scan_units = _discover_scan_units(
                bucket, prefix, endpoint, console, flat_variant=args.variant
            )
        except Exception as e:
            console.print(f"[red]Error listing S3 prefix: {e}[/red]")
            return 1

        if not scan_units:
            console.print(
                "[yellow]No (version, variant, year) units found! "
                "Expected layout: <root>/v<N>(.<M>)/"
                "[global_0.1_degree_representation(.<variant>)/]<YYYY>/grid_*[/yellow]"
            )
            if not landmasks_only_ok:
                return 1
            console.print("[yellow]Continuing with the landmask scan only.[/yellow]")

    grid_re = re.compile(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)(_scales)?\.npy$")

    # Shard the listing within each (version, variant, year) by integer-
    # longitude prefix. S3's ListObjectsV2 caps at 1000 keys per response, so a
    # year with ~5M objects needs ~5000 sequential continuation calls. Splitting
    # on ``grid_{int_lon}.`` gives 360 independent prefixes per year (~5k objs
    # each) that can be listed in parallel. The trailing dot disambiguates
    # ``grid_1.`` from ``grid_10.``, ``grid_100.``, etc.
    shard_prefixes: List[str] = (
        [f"grid_-{n}." for n in range(1, 180)]
        + ["grid_-0."]
        + [f"grid_{n}." for n in range(0, 180)]
    )

    def _scan_shard(
        version: str,
        variant: str,
        year: int,
        year_prefix: str,
        shard: str,
        progress: Progress,
        task_id,
    ):
        """List one (version, variant, year, lon-shard) and return tile records."""
        tiles: Dict[Tuple[float, float], Dict[str, Tuple[str, int, str]]] = {}
        listed = 0
        for key, size, lm in _s3_list(bucket, year_prefix + shard, endpoint):
            listed += 1
            # Batch UI updates: every 50 objects is plenty smooth and avoids
            # contention on the Progress lock during big listings.
            if listed % 50 == 0:
                progress.update(task_id, advance=50)
            name = key.rsplit("/", 1)[-1]
            m = grid_re.match(name)
            if not m:
                continue
            lon = float(m.group(1))
            lat = float(m.group(2))
            kind = "scales" if m.group(3) else "grid"
            tiles.setdefault((lon, lat), {})[kind] = (key, size, lm)
        progress.update(task_id, advance=listed % 50)

        out = []
        base_url = f"s3://{bucket}/"
        for (lon, lat), parts in tiles.items():
            if "grid" not in parts or "scales" not in parts:
                continue
            gkey, gsize, glm = parts["grid"]
            skey, ssize, slm = parts["scales"]
            directory = gkey.rsplit("/", 1)[0]
            out.append(
                {
                    "version": version,
                    "variant": variant,
                    "year": year,
                    "lon": lon,
                    "lat": lat,
                    "directory": base_url + directory,
                    "grid_path": base_url + gkey,
                    "scales_path": base_url + skey,
                    "grid_mtime": datetime.fromisoformat(glm.replace("Z", "+00:00")),
                    "scales_mtime": datetime.fromisoformat(slm.replace("Z", "+00:00")),
                    "grid_size": gsize,
                    "scales_size": ssize,
                }
            )
        return (version, variant, year), out

    n_shards_per_year = len(shard_prefixes)
    total_shards = len(scan_units) * n_shards_per_year
    console.print(
        f"\n[cyan]Scanning {len(scan_units)} (version,variant,year) units × "
        f"{n_shards_per_year} lon-shards = {total_shards:,} prefixes "
        f"({args.workers} workers)...[/cyan]"
    )

    records: List[Dict] = []
    units_found = set()
    # (version, variant) datasets that lost one or more shards even after
    # _s3_list's built-in retries. Their manifests would be silently
    # incomplete, so they are not written.
    failed_datasets: set = set()
    tile_counts: Dict[Tuple[str, str, int], int] = {
        (v, var, y): 0 for v, var, y, _ in scan_units
    }
    shards_left: Dict[Tuple[str, str, int], int] = {
        (v, var, y): n_shards_per_year for v, var, y, _ in scan_units
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("objs"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        unit_tasks: Dict[Tuple[str, str, int], int] = {
            (v, var, y): progress.add_task(f"[cyan]{v}/{var}/{y}[/cyan]", total=None)
            for v, var, y, _ in sorted(scan_units)
        }
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _scan_shard,
                    v,
                    var,
                    y,
                    p,
                    shard,
                    progress,
                    unit_tasks[(v, var, y)],
                ): (v, var, y, shard)
                for v, var, y, p in sorted(scan_units)
                for shard in shard_prefixes
            }
            for fut in as_completed(futures):
                v, var, y, shard = futures[fut]
                key = (v, var, y)
                try:
                    _, out = fut.result()
                except Exception as e:
                    console.print(
                        f"[red]Error scanning {v}/{var}/{y}/{shard}: {e}[/red]"
                    )
                    failed_datasets.add((v, var))
                    shards_left[key] -= 1
                    continue
                if out:
                    units_found.add(key)
                    records.extend(out)
                    tile_counts[key] += len(out)
                shards_left[key] -= 1
                count = tile_counts[key]
                if shards_left[key] == 0:
                    progress.update(
                        unit_tasks[key],
                        description=f"[green]{v}/{var}/{y}[/green] · {count:,} tiles",
                    )
                else:
                    done = n_shards_per_year - shards_left[key]
                    progress.update(
                        unit_tasks[key],
                        description=(
                            f"[cyan]{v}/{var}/{y}[/cyan] · {count:,} tiles "
                            f"({done}/{n_shards_per_year} shards)"
                        ),
                    )

    if not records:
        console.print("[yellow]No embedding tiles found![/yellow]")
        if not landmasks_only_ok:
            return 1

    from .registry import _version_path_from_norm

    # One parquet per dataset version. Layout mirrors the remote tree so the
    # whole output can be uploaded with
    # ``aws s3 cp --recursive <output_dir>/ s3://<bucket>/``.
    written_files: List[Path] = []
    # Set when a landmask scan loses shards; embedding-side losses are
    # tracked in failed_datasets. Either makes the run exit non-zero.
    scan_had_failures = False
    try:
        if records:
            df = pd.DataFrame(records)
            df = df.sort_values(["version", "variant", "year", "lon", "lat"])
            df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
            df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)
            # One parquet per dataset directory. dataset_path collapses all
            # 1.0 variants into v1/ and maps later versions to their
            # suffixed dirs (v1.1-cam, v2-2B-L~beta1, ...), so the output
            # lands exactly where the npy/ tree expects the manifest.
            df["_dataset_dir"] = [
                dataset_path(str(v), str(var))
                for v, var in zip(df["version"], df["variant"])
            ]
            failed_dataset_dirs = {
                dataset_path(str(v), str(var)) for v, var in failed_datasets
            }
            for dataset_dir, group_df in df.groupby("_dataset_dir", sort=True):
                if dataset_dir in failed_dataset_dirs:
                    console.print(
                        f"[red]Not writing {dataset_dir}/manifest.parquet: "
                        f"one or more lon-shards failed to list even after "
                        f"retries, so the manifest would be incomplete. "
                        f"Re-run the scan for this dataset.[/red]"
                    )
                    continue
                group_df = group_df.drop(columns=["_dataset_dir"])
                # Defensive dedupe: bucket-side misfiling (e.g. a tile filed
                # under the wrong grid directory) can surface the same key
                # twice via different lon-shards.
                before_dedupe = len(group_df)
                group_df = group_df.drop_duplicates(
                    subset=["version", "variant", "year", "lon", "lat"],
                    keep="first",
                )
                if len(group_df) != before_dedupe:
                    console.print(
                        f"[yellow]  Dropped {before_dedupe - len(group_df):,} "
                        f"duplicate (variant, year, lon, lat) rows for "
                        f"{dataset_dir}[/yellow]"
                    )
                out_file = output_dir / str(dataset_dir) / "manifest.parquet"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"[cyan]Writing {len(group_df):,} tiles to {out_file}...[/cyan]"
                )
                _atomic_write_parquet(group_df, out_file)
                written_files.append(out_file)

        # Landmask scan: one parquet per version that has landmask TIFFs.
        landmask_files_by_version: Dict[str, Path] = {}
        if not args.no_landmasks:
            # The landmask tree may live outside the embeddings prefix (on
            # data.source.coop it is the sibling landmasks/ tree);
            # --landmasks-uri points the scan there.
            if args.landmasks_uri:
                lm_bucket, lm_root = _parse_s3_uri(args.landmasks_uri)
            else:
                lm_bucket, lm_root = bucket, prefix
            console.print("\n[cyan]Discovering landmask directories...[/cyan]")
            try:
                lm_units = _discover_landmask_prefixes(
                    lm_bucket, lm_root, endpoint, console
                )
            except Exception as e:
                console.print(f"[yellow]Could not list landmasks: {e}[/yellow]")
                lm_units = []

            for _version_norm, version_path, lm_prefix in lm_units:
                console.print(
                    f"\n[cyan]Scanning landmasks for {version_path} "
                    f"({len(shard_prefixes)} shards)...[/cyan]"
                )
                lm_records: List[Dict] = []

                def _scan_lm_shard(shard: str):
                    out_local = []
                    for key, size, lm in _s3_list(
                        lm_bucket, lm_prefix + shard, endpoint
                    ):
                        name = key.rsplit("/", 1)[-1]
                        m = _LANDMASK_RE.match(name)
                        if not m:
                            continue
                        lon = float(m.group(1))
                        lat = float(m.group(2))
                        # No version column — landmasks are a property of the
                        # 0.1° lat/lon grid, not of an embedding version.
                        out_local.append(
                            {
                                "lon": lon,
                                "lat": lat,
                                "file_size": size,
                                "mtime": datetime.fromisoformat(
                                    lm.replace("Z", "+00:00")
                                ),
                                "key": f"s3://{lm_bucket}/{key}",
                            }
                        )
                    return out_local

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=20),
                    MofNCompleteColumn(),
                    TextColumn("shards"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as lm_progress:
                    lm_task = lm_progress.add_task(
                        f"[cyan]{version_path} landmasks[/cyan]",
                        total=len(shard_prefixes),
                    )
                    lm_shards_failed = 0
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures = {
                            pool.submit(_scan_lm_shard, s): s for s in shard_prefixes
                        }
                        for fut in as_completed(futures):
                            try:
                                lm_records.extend(fut.result())
                            except Exception as e:
                                console.print(
                                    f"[red]  Landmask shard {futures[fut]} failed: {e}[/red]"
                                )
                                lm_shards_failed += 1
                            lm_progress.update(lm_task, advance=1)

                if lm_shards_failed:
                    # An incomplete landmask registry is worse than none:
                    # skip the write so a partial file can't be uploaded.
                    console.print(
                        f"[red]Not writing {version_path}/landmasks.parquet: "
                        f"{lm_shards_failed} shard(s) failed to list even "
                        f"after retries. Re-run the scan.[/red]"
                    )
                    scan_had_failures = True
                    continue

                if not lm_records:
                    console.print(
                        f"[yellow]  No landmasks parsed for {version_path}[/yellow]"
                    )
                    continue

                lm_df = pd.DataFrame(lm_records)
                before_dedupe = len(lm_df)
                lm_df = lm_df.drop_duplicates(subset=["lon", "lat"], keep="first")
                if len(lm_df) != before_dedupe:
                    console.print(
                        f"[yellow]  Dropped {before_dedupe - len(lm_df):,} "
                        f"duplicate (lon, lat) landmask rows[/yellow]"
                    )
                lm_df = lm_df.sort_values(["lon", "lat"])
                lm_df["lon_i"] = (lm_df["lon"] * 100).round().astype(np.int32)
                lm_df["lat_i"] = (lm_df["lat"] * 100).round().astype(np.int32)

                lm_out_file = output_dir / version_path / "landmasks.parquet"
                lm_out_file.parent.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"[cyan]Writing {len(lm_df):,} landmasks to {lm_out_file}...[/cyan]"
                )
                _atomic_write_parquet(lm_df, lm_out_file)
                written_files.append(lm_out_file)
                landmask_files_by_version[version_path] = lm_out_file

            # Each dataset version has its own landmasks now — no cross-version
            # copying. If a version has embeddings but no landmasks dir on S3,
            # warn rather than silently substituting another version's masks.
            embedding_version_paths = {
                _version_path_from_norm(str(v)) for v, _, _, _ in scan_units
            }
            missing = sorted(embedding_version_paths - landmask_files_by_version.keys())
            for vpath in missing:
                console.print(
                    f"[yellow]Warning: {vpath} has embeddings but no landmask "
                    f"TIFFs at the scanned location. No landmasks.parquet "
                    f"will be written for {vpath}.[/yellow]"
                )

        # Summary grouped by (version, variant)
        from collections import defaultdict

        summary: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for v, var, y in sorted(units_found):
            summary[(v, var)].append(y)
        if scan_had_failures or failed_datasets:
            summary_lines = [
                "[red]⚠ S3 Scan finished with failures[/red]",
                f"📊 Tiles found: {len(records):,}",
            ]
        else:
            summary_lines = [
                "[green]✅ S3 Scan Complete[/green]",
                f"📊 Tiles found: {len(records):,}",
            ]
        for (v, var), years in sorted(summary.items()):
            summary_lines.append(
                f"  • {v}/{var}: {', '.join(str(y) for y in sorted(years))}"
            )
        for v, var in sorted(failed_datasets):
            summary_lines.append(
                f"  [red]• {v}/{var}: shards failed — manifest not written[/red]"
            )
        summary_lines.append("📄 Output files:")
        for f in written_files:
            summary_lines.append(f"  • {f}")

        if written_files:
            # Per-file upload hints with the correct destinations: manifests
            # go under the npy/ tree root, landmask registries under the
            # landmasks/ tree root. (A single recursive cp of output_dir
            # would mix the two trees.)
            def _tree_root(pfx: str, tail_re) -> str:
                parts = pfx.rstrip("/").split("/") if pfx.strip("/") else []
                if parts and tail_re.match(parts[-1]):
                    parts = parts[:-1]
                return "/".join(parts) + "/" if parts else ""

            npy_root = _tree_root(prefix, _DATASET_DIR_RE)
            if args.landmasks_uri:
                lm_hint_bucket, lm_hint_root = _parse_s3_uri(args.landmasks_uri)
            else:
                lm_hint_bucket, lm_hint_root = bucket, prefix
            lm_hint_root = _tree_root(lm_hint_root, _VERSION_RE)

            summary_lines.append("")
            summary_lines.append("[dim]Upload with:[/dim]")
            for f in written_files:
                rel = f.relative_to(output_dir).as_posix()
                if f.name == "manifest.parquet":
                    dest = f"s3://{bucket}/{npy_root}{rel}"
                else:
                    dest = f"s3://{lm_hint_bucket}/{lm_hint_root}{rel}"
                summary_lines.append(
                    f'[dim]  aws s3 cp "{f}" "{dest}" --endpoint-url {endpoint}[/dim]'
                )
        console.print(
            Panel.fit(
                "\n".join(summary_lines),
                style="red" if (scan_had_failures or failed_datasets) else "green",
            )
        )
        return 1 if (scan_had_failures or failed_datasets) else 0
    except Exception as e:
        console.print(f"[red]Error writing parquet files: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


def file_check_command(args):
    """Check multiple inventory parquet files for duplicate year/lon/lat coordinates.

    This command loads multiple parquet files (generated by file-scan) and identifies
    any year/lon/lat coordinates that appear in multiple files or multiple times within
    the same file. This is useful for finding duplicate embeddings across machines.
    """
    from collections import defaultdict

    console = Console()

    # Resolve input files
    parquet_files = [Path(f).resolve() for f in args.parquet_files]

    # Validate that all files exist
    missing_files = [f for f in parquet_files if not f.exists()]
    if missing_files:
        console.print("[red]Error: The following parquet files do not exist:[/red]")
        for f in missing_files:
            console.print(f"  - {f}")
        return 1

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Checking for Duplicate Coordinates[/bold blue]\n"
            f"📊 Files to check: {len(parquet_files)}",
            style="blue",
        )
    )

    # Track coordinates and their sources
    # Key: (year, lon, lat), Value: list of location info dicts
    coord_locations = defaultdict(list)

    # Load each parquet file
    for parquet_file in parquet_files:
        console.print(f"\n[cyan]Loading {parquet_file.name}...[/cyan]")

        try:
            df = pd.read_parquet(parquet_file)

            # Verify required columns
            required_cols = ["year", "lon", "lat", "directory"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                console.print(
                    f"[yellow]Warning: Missing required columns in {parquet_file.name}: {missing}[/yellow]"
                )
                console.print(
                    f"[yellow]Available columns: {df.columns.tolist()}[/yellow]"
                )
                continue

            console.print(f"  Found [green]{len(df):,}[/green] tiles")

            # Record each coordinate and its location
            for _, row in df.iterrows():
                year = row["year"]
                lon, lat = row["lon"], row["lat"]
                directory = row["directory"]
                grid_path = row.get("grid_path", None)
                grid_mtime = row.get("grid_mtime", None)
                scales_mtime = row.get("scales_mtime", None)

                coord_locations[(year, lon, lat)].append(
                    {
                        "parquet_file": parquet_file.name,
                        "directory": directory,
                        "grid_path": grid_path,
                        "grid_mtime": grid_mtime,
                        "scales_mtime": scales_mtime,
                    }
                )

        except Exception as e:
            console.print(f"[red]Error loading {parquet_file.name}: {e}[/red]")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            continue

    # Find duplicates (coordinates that appear more than once)
    duplicates = {
        coord: locations
        for coord, locations in coord_locations.items()
        if len(locations) > 1
    }

    if not duplicates:
        console.print(
            Panel.fit(
                "[green]✅ No duplicate coordinates found![/green]\n"
                f"📊 Total unique coordinates: {len(coord_locations):,}",
                style="green",
            )
        )
        return 0

    # Display duplicates
    console.print(
        Panel.fit(
            f"[yellow]{emoji('⚠️  ')}Found {len(duplicates):,} duplicate coordinates[/yellow]\n"
            f"{emoji('📊 ')}Total unique coordinates: {len(coord_locations):,}",
            style="yellow",
        )
    )

    console.print("\n[bold]Duplicate Coordinates:[/bold]\n")

    # Sort duplicates by coordinate for consistent output
    for (year, lon, lat), locations in sorted(duplicates.items()):
        console.print(
            f"[bold cyan]Year {year}, Coordinate: ({lon:.2f}, {lat:.2f})[/bold cyan]"
        )
        console.print(f"  Found in [yellow]{len(locations)}[/yellow] locations:")

        for i, loc in enumerate(locations, 1):
            console.print(
                f"\n  [dim]{i}.[/dim] Parquet: [green]{loc['parquet_file']}[/green]"
            )
            console.print(f"     Directory: {loc['directory']}")
            if loc["grid_path"]:
                console.print(f"     Grid path: {loc['grid_path']}")
            if loc["grid_mtime"]:
                console.print(f"     Grid mtime: {loc['grid_mtime']}")
            if loc["scales_mtime"]:
                console.print(f"     Scales mtime: {loc['scales_mtime']}")

        console.print()

    # Summary statistics
    console.print("\n[bold]Summary by parquet file:[/bold]")
    file_dup_counts = defaultdict(int)
    for locations in duplicates.values():
        for loc in locations:
            file_dup_counts[loc["parquet_file"]] += 1

    table = Table(show_header=True)
    table.add_column("Parquet File", style="cyan")
    table.add_column("Duplicate Entries", style="yellow", justify="right")

    for parquet_file in sorted(file_dup_counts.keys()):
        table.add_row(parquet_file, f"{file_dup_counts[parquet_file]:,}")

    console.print(table)

    return 0


def _parse_int_range(s: str) -> list[int]:
    """Parse an int range like '2017-2025' or '29,30,31'."""
    if "-" in s and "," not in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(v.strip()) for v in s.split(",")]


def _find_registry(
    base_dir: str, registry_dir: Optional[str] = None
) -> Tuple[str, str]:
    """Find the registry directory from base_dir. Returns (base_dir, registry_dir).

    Looks for either ``manifest.parquet`` (current per-version format) or the
    legacy ``registry.parquet`` in base_dir and up to 3 parent directories.
    """
    if registry_dir is None:
        candidate = Path(base_dir)
        for _ in range(3):
            if (candidate / "manifest.parquet").exists():
                return base_dir, str(candidate)
            if (candidate / "registry.parquet").exists():
                return base_dir, str(candidate)
            candidate = candidate.parent
    return base_dir, registry_dir or base_dir


def _detect_dataset_metadata(
    base_dir: str,
    explicit_version: Optional[str],
    explicit_variant: Optional[str],
) -> Tuple[str, str]:
    """Resolve the (dataset_version, dataset_variant) for a local mirror.

    If the user passes the flags explicitly we honour them; otherwise we look
    for a ``tessera_metadata.json`` sidecar (written by the CLI download
    flow) and use what it recorded. Falls back to v1 and the version's
    published variant.
    """
    if explicit_version and explicit_variant:
        return explicit_version, explicit_variant

    import json

    sidecar = Path(base_dir) / "tessera_metadata.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            version = (
                explicit_version
                or data.get("dataset_version_path")
                or data.get("dataset_version")
            )
            variant = explicit_variant or data.get("dataset_variant")
            if version and variant:
                return version, variant
        except (OSError, ValueError):
            pass

    from .registry import _parse_dataset_version

    version = explicit_version or "v1"
    variant = explicit_variant or default_variant(_parse_dataset_version(version)[1])
    return version, variant


def _add_source_args(parser) -> None:
    """Register the dataset-selection flags shared by zarr-init and zarr-fill."""
    parser.add_argument(
        "--registry-dir",
        type=str,
        default=None,
        help="Local directory containing manifest.parquet / landmasks.parquet "
        "(default: auto-detected from base_dir and parents, or pulled from a "
        "URL base_dir)",
    )
    parser.add_argument(
        "--source-npy-root",
        type=str,
        default=None,
        help="Location holding {year}/grid_<lon>_<lat>/ tile directories, for "
        "a mirror that does not use the published <root>/npy/<version> "
        "layout",
    )
    parser.add_argument(
        "--source-landmask-root",
        type=str,
        default=None,
        help="Location holding grid_<lon>_<lat>.tiff landmasks, for a mirror "
        "that does not use the published <root>/landmasks/<version> layout",
    )
    parser.add_argument(
        "--manifest-url",
        type=str,
        default=None,
        help="Location of the embeddings manifest, for mirrors that do not "
        "follow the npy/<version>/manifest.parquet layout",
    )
    parser.add_argument(
        "--landmasks-url",
        type=str,
        default=None,
        help="Location of the landmasks registry, for mirrors that do not "
        "follow the landmasks/<version>/landmasks.parquet layout",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Tessera dataset version (e.g. v1, v1.1). "
        "Default: read from tessera_metadata.json in base_dir, else v1.",
    )
    parser.add_argument(
        "--dataset-variant",
        type=str,
        default=None,
        help="Tessera dataset variant (e.g. vultr, cambridge). "
        "Default: read from tessera_metadata.json in base_dir, else the "
        "version's published variant.",
    )


def _add_state_arg(parser) -> None:
    """Register ``--state-url`` (legacy; fills are now stateless)."""
    parser.add_argument(
        "--state-url",
        type=str,
        default=None,
        help="Legacy build-state location from before fills became "
        "stateless. Only zarr-consolidate still reads it (to merge old "
        "ingestion registries); accepted elsewhere for script "
        "compatibility and ignored.",
    )


def _add_storage_args(parser, prefix: str, label: str, writable: bool = False) -> None:
    """Register ``--{prefix}-*`` object-store flags on *parser*.

    Access keys are deliberately absent: they come from the environment
    (``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``), a named profile, or an
    instance role, so they never appear in a process listing.

    Args:
        writable: Also offer ``--{prefix}-acl``, which only affects writes.
    """
    group = parser.add_argument_group(f"{label} object-store options")
    group.add_argument(
        f"--{prefix}-endpoint-url",
        type=str,
        default=None,
        help=f"S3-compatible endpoint for the {label.lower()} "
        f"(default: $AWS_ENDPOINT_URL, else the AWS default)",
    )
    group.add_argument(
        f"--{prefix}-anon",
        action="store_true",
        help=f"Access the {label.lower()} anonymously (public buckets)",
    )
    group.add_argument(
        f"--{prefix}-region",
        type=str,
        default=None,
        help=f"Region for the {label.lower()} (default: $AWS_DEFAULT_REGION)",
    )
    group.add_argument(
        f"--{prefix}-profile",
        type=str,
        default=None,
        help=f"Shared-config profile for the {label.lower()} "
        f"(default: $AWS_PROFILE)",
    )
    group.add_argument(
        f"--{prefix}-requester-pays",
        action="store_true",
        help=f"Send requester-pays headers to the {label.lower()}",
    )
    group.add_argument(
        f"--{prefix}-path-style",
        action="store_true",
        help=f"Address the {label.lower()} as endpoint/bucket/key rather "
        f"than bucket.endpoint/key. Needed by most S3-compatible servers "
        f"(MinIO, Ceph, and any endpoint without wildcard DNS).",
    )
    if writable:
        from .remote import OBJECT_ACLS

        group.add_argument(
            f"--{prefix}-acl",
            type=str,
            default=None,
            choices=OBJECT_ACLS,
            metavar="ACL",
            help=f"Canned ACL applied to every object written to the "
            f"{label.lower()} (e.g. bucket-owner-full-control when the "
            f"bucket belongs to another account)",
        )


def _object_store_errors() -> Tuple[type, ...]:
    """Exception types worth reporting as a message rather than a traceback.

    fsspec translates most S3 failures into builtin OSErrors — a 403 arrives
    as PermissionError, a 404 as FileNotFoundError — so catching only
    botocore's own types misses the common cases. botocore's are included for
    the errors raised before fsspec gets a chance to translate them (bad
    endpoint, unresolvable credentials); the tuple is short when the s3
    extra is not installed.
    """
    types: Tuple[type, ...] = (PermissionError, OSError)
    try:
        from botocore.exceptions import BotoCoreError, ClientError

        return types + (BotoCoreError, ClientError)
    except ImportError:
        return types


def _report_store_error(e: Exception, console: "Console") -> int:
    """Print an object-store or missing-backend failure and return exit 1.

    Error text is escaped: it routinely contains bracketed fragments (and
    our own ``geotessera[s3]`` hint) that Rich would otherwise eat as markup.
    """
    from rich.markup import escape

    detail = str(e) or type(e).__name__
    console.print(f"[red]{emoji('❌ ')}{escape(detail)}[/red]")

    if isinstance(e, ImportError):
        return 1

    console.print(
        "Check the endpoint and credentials: --source-endpoint-url / "
        "--store-endpoint-url, --*-profile, --*-anon, or the AWS_* "
        "environment variables."
    )
    if isinstance(e, PermissionError):
        console.print(
            "A 403 on a key that does not exist yet usually means the "
            "credentials lack s3:ListBucket on the prefix — S3 then hides "
            "whether the object is there rather than answering 404."
        )
    return 1


def _storage_options_for(
    args, prefix: str, location: str
) -> Optional[Dict[str, Any]]:
    """Build fsspec storage options from ``--{prefix}-*`` flags.

    Returns None for local paths so plain filesystem access never picks up
    stray ``AWS_*`` environment variables.
    """
    from .remote import build_storage_options, is_url

    if not is_url(location):
        return None

    def opt(name):
        return getattr(args, f"{prefix}_{name}", None)

    return build_storage_options(
        endpoint_url=opt("endpoint_url"),
        anon=bool(opt("anon")),
        region=opt("region"),
        profile=opt("profile"),
        requester_pays=bool(opt("requester_pays")),
        acl=opt("acl"),
        path_style=bool(opt("path_style")),
    )


def _remote_registry_cache_dir(root: str, version_path: str) -> Path:
    """Local cache directory for registries pulled from a remote mirror.

    Keyed by a digest of the source root so mirrors with different contents
    never share a cached manifest.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    digest = hashlib.sha256(root.encode()).hexdigest()[:12]
    return base / "geotessera" / "mirrors" / digest / version_path


def _sync_remote_file(
    loc: str,
    dest: Path,
    storage_options: Optional[Dict[str, Any]],
    console: "Console",
    optional: bool = False,
) -> Optional[Path]:
    """Download *loc* to *dest* unless the cached copy already matches.

    Freshness is judged on the object's size and last-modified stamp, which
    every S3-compatible endpoint reports on a HEAD — far cheaper than
    re-pulling a multi-hundred-megabyte manifest each run.

    A local path is used where it lies; only URLs are ever copied.
    """
    import json

    from . import remote

    if not remote.is_url(loc):
        local = Path(loc)
        if local.exists():
            return local
        if optional:
            return None
        raise FileNotFoundError(loc)

    fs = remote.get_fs(loc, storage_options)
    try:
        info = fs.info(str(loc))
    except FileNotFoundError:
        if optional:
            return None
        raise
    except Exception as e:
        if optional:
            console.print(f"[yellow]Could not stat {loc}: {e}[/yellow]")
            return None
        raise

    stamp = {
        "size": info.get("size"),
        "mtime": str(info.get("LastModified") or info.get("mtime") or ""),
    }
    meta = dest.with_name(dest.name + ".meta.json")

    if dest.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == stamp:
                console.print(f"[dim]Using cached {dest.name} ({loc})[/dim]")
                return dest
        except (OSError, ValueError):
            pass

    size_str = format_bytes(stamp["size"]) if stamp["size"] else "unknown size"
    console.print(f"Downloading {loc} ({size_str})...")
    remote.download(loc, dest, storage_options)
    meta.write_text(json.dumps(stamp, default=str))
    return dest


def _resolve_source(args, console: "Console"):
    """Resolve the tile source and its registry for a zarr-init/fill run.

    ``args.base_dir`` is either a local mirror directory (the historical
    behaviour) or a URL such as ``s3://bucket/prefix`` pointing at a
    repository in the published layout.  In the remote case the manifest and
    landmask registries are pulled from that same root — or from
    ``--manifest-url``/``--landmasks-url`` when a mirror deviates — and
    cached locally, so no local tile mirror is needed at all.

    Returns ``(registry, source, dataset_version, dataset_variant)`` where
    ``source`` is a :class:`~geotessera.zarr.TileSource`, or None to mean
    "use the registry's local mirror".
    """
    from .registry import Registry, _parse_dataset_version
    from .remote import is_url, join
    from .zarr import TileSource

    base_dir = args.base_dir

    # A Zarr store passed where the tile source belongs otherwise fails much
    # later, looking for a manifest inside the store and often with the wrong
    # credentials. Say what happened instead.
    from .remote import exists as _loc_exists

    try:
        looks_like_store = _loc_exists(
            join(base_dir, "zarr.json"),
            _storage_options_for(args, "source", base_dir),
            on_denied=False,
        )
    except Exception:
        looks_like_store = False
    if looks_like_store:
        raise ValueError(
            f"{base_dir} looks like a Zarr store, not a tile source — it has "
            f"a zarr.json. The tile source comes first and the store second: "
            f"<source> <store>."
        )

    def override(loc, name, optional=False):
        """Resolve an explicit --manifest-url / --landmasks-url, if given."""
        if not loc:
            return None
        return _sync_remote_file(
            loc,
            _remote_registry_cache_dir(str(base_dir), "override") / name,
            _storage_options_for(args, "source", loc),
            console,
            optional=optional,
        )

    if not is_url(base_dir):
        base_dir, registry_dir = _find_registry(base_dir, args.registry_dir)
        dataset_version, dataset_variant = _detect_dataset_metadata(
            base_dir, args.dataset_version, args.dataset_variant
        )
        console.print(
            f"[cyan]Using dataset version={dataset_version}, "
            f"variant={dataset_variant}[/cyan]"
        )
        registry = Registry(
            version=dataset_version,
            variant=dataset_variant,
            embeddings_dir=base_dir,
            registry_dir=registry_dir,
            registry_path=override(args.manifest_url, "manifest.parquet"),
            landmasks_registry_path=override(
                args.landmasks_url, "landmasks.parquet", optional=True
            ),
        )
        return registry, None, dataset_version, dataset_variant

    # Remote mirror.
    dataset_version = args.dataset_version or "v1"
    version_path, version_norm = _parse_dataset_version(dataset_version)
    dataset_variant = args.dataset_variant or default_variant(version_norm)
    storage_options = _storage_options_for(args, "source", base_dir)

    console.print(
        f"[cyan]Streaming tiles from {base_dir} "
        f"(version={version_path}, variant={dataset_variant})[/cyan]"
    )

    if args.registry_dir:
        registry = Registry(
            version=dataset_version,
            variant=dataset_variant,
            registry_dir=args.registry_dir,
        )
    else:
        cache_dir = _remote_registry_cache_dir(base_dir, version_path)
        manifest_loc = args.manifest_url or join(
            base_dir, "npy", version_path, "manifest.parquet"
        )
        landmasks_loc = args.landmasks_url or join(
            base_dir, "landmasks", version_path, "landmasks.parquet"
        )
        manifest_path = _sync_remote_file(
            manifest_loc,
            cache_dir / "manifest.parquet",
            _storage_options_for(args, "source", manifest_loc),
            console,
        )
        landmasks_path = _sync_remote_file(
            landmasks_loc,
            cache_dir / "landmasks.parquet",
            _storage_options_for(args, "source", landmasks_loc),
            console,
            optional=True,
        )
        registry = Registry(
            version=dataset_version,
            variant=dataset_variant,
            registry_path=manifest_path,
            landmasks_registry_path=landmasks_path,
        )

    if args.source_npy_root or args.source_landmask_root:
        # A mirror that lays tiles out its own way: take the roots verbatim
        # rather than appending the published npy/<version> convention.
        default = TileSource.for_url(base_dir, version_path, storage_options)
        source = TileSource(
            embeddings_root=args.source_npy_root or default.embeddings_root,
            landmasks_root=args.source_landmask_root or default.landmasks_root,
            storage_options=storage_options,
        )
    else:
        source = TileSource.for_url(base_dir, version_path, storage_options)
    return registry, source, dataset_version, dataset_variant


def zarr_init_command(args):
    """Create an empty tessera store with time dimension."""
    from rich.console import Console
    from .zarr import STRETCH_SAMPLE_K, init_store

    console = Console()

    try:
        registry, _source, _version, _variant = _resolve_source(args, console)
    except ValueError as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    years = _parse_int_range(args.years)
    output = args.output
    store_options = _storage_options_for(args, "store", output)

    try:
        import importlib.metadata

        version = importlib.metadata.version("geotessera")
    except Exception:
        version = "unknown"

    # Derive the embedding model version from the chosen dataset version
    # (v1 -> 1.0, v1.1 -> 1.1) so the geoemb:model URI in the Zarr root
    # attrs always matches the actual model the embeddings came from.
    model_version = registry._version_norm
    console.print(
        f"[cyan]geoemb:model -> https://geotessera.org/model/{model_version}[/cyan]"
    )

    try:
        init_store(
            registry,
            output,
            years,
            geotessera_version=version,
            model_version=model_version,
            console=console,
            storage_options=store_options,
            state_url=args.state_url,
            stretch_sample_size=args.stretch_sample_size or STRETCH_SAMPLE_K,
        )
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    return 0


def zarr_fill_command(args):
    """Incrementally fill a tessera store with tile data."""
    import warnings
    from rich.console import Console
    from .zarr import fill_store

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    console = Console()

    # Backfill mode scans the store itself — no tile source, no manifest.
    # Accept the store as the only positional (argparse binds it to base_dir
    # when store_path is absent).
    if args.backfill_stretch_stats:
        from .zarr import backfill_stretch_stats

        store_path = args.store_path or args.base_dir
        if store_path is None:
            console.print(f"[red]{emoji('❌ ')}No store given.[/red]")
            return 1
        store_options = _storage_options_for(args, "store", store_path)
        zones = _parse_int_range(args.zones) if args.zones else None
        years = [args.year] if args.year else None
        try:
            n = backfill_stretch_stats(
                store_path,
                zones=zones,
                years=years,
                console=console,
                storage_options=store_options,
            )
        except (ValueError, RuntimeError) as e:
            console.print(f"[red]{emoji('❌ ')}{e}[/red]")
            return 1
        except (ImportError, *_object_store_errors()) as e:
            return _report_store_error(e, console)
        console.print(
            f"{emoji('✅ ')}{n} (zone, year) statistics rebuilt. "
            f"Run zarr-consolidate to publish them."
        )
        return 0

    if args.store_path is None:
        console.print(
            f"[red]{emoji('❌ ')}Usage: zarr-fill <source> <store> "
            f"(the store-only form is for --backfill-stretch-stats).[/red]"
        )
        return 1

    try:
        registry, source, _version, _variant = _resolve_source(args, console)
    except ValueError as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    store_path = args.store_path
    store_options = _storage_options_for(args, "store", store_path)
    year = args.year
    zones = _parse_int_range(args.zones) if args.zones else None

    consolidate = None
    if args.consolidate:
        consolidate = True
    elif args.no_consolidate:
        consolidate = False

    try:
        n = fill_store(
            registry,
            store_path,
            year=year,
            zones=zones,
            console=console,
            workers=args.workers,
            storage_options=store_options,
            source=source,
            consolidate=consolidate,
            skip_existing_shards=not args.rewrite_existing_shards,
            spill_dir=args.spill_dir,
            collect_stretch_stats=not args.no_stretch_stats,
        )
    except RuntimeError as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    console.print(f"\n{emoji('✅ ')}{n} shards written")
    return 0


def zarr_scan_command(args):
    """Report how much of a store still needs filling."""
    import warnings
    from rich.console import Console

    from .zarr import scan_store, summarise_scan

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    console = Console()

    # The manifest is optional: land coverage alone says which shards can
    # ever hold data, and that comes from the much smaller landmask registry.
    if args.base_dir:
        try:
            registry, source, dataset_version, _variant = _resolve_source(
                args, console
            )
        except ValueError as e:
            console.print(f"[red]{emoji('❌ ')}{e}[/red]")
            return 1
        except (ImportError, *_object_store_errors()) as e:
            return _report_store_error(e, console)
    else:
        registry, source = None, None
        dataset_version = args.dataset_version or "v1"

    store_options = _storage_options_for(args, "store", args.store_path)
    years = _parse_int_range(args.years) if args.years else None
    zones = _parse_int_range(args.zones) if args.zones else None

    try:
        df = scan_store(
            registry,
            args.store_path,
            years=years,
            zones=zones,
            console=console if args.verbose else None,
            storage_options=store_options,
            source=source,
            state_url=args.state_url,
            output=args.output,
            dataset_version=dataset_version,
            landmasks_path=args.landmasks_url,
        )
    except ValueError as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    summarise_scan(df, console)
    return 0


def zarr_extend_command(args):
    """Append new years to an existing store's time axis."""
    import warnings
    from rich.console import Console

    from .zarr import extend_store

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    console = Console()
    store_options = _storage_options_for(args, "store", args.store_path)
    years = _parse_int_range(args.years)
    zones = _parse_int_range(args.zones) if args.zones else None

    try:
        n = extend_store(
            args.store_path,
            years,
            console=console,
            storage_options=store_options,
            zones=zones,
            consolidate=not args.no_consolidate,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    if n == 0:
        console.print(
            f"{emoji('✅ ')}Nothing to do — every zone already has "
            f"{', '.join(str(y) for y in years)}"
        )
        return 0

    console.print(
        f"{emoji('✅ ')}{n} zone(s) extended. "
        f"Fill them with: geotessera-registry zarr-fill <source> "
        f"{args.store_path} --year {years[0]} --zones <n>"
    )
    return 0


def zarr_consolidate_command(args):
    """Re-consolidate store metadata after in-place changes."""
    from rich.console import Console

    from .zarr import consolidate_store

    console = Console()
    store_options = _storage_options_for(args, "store", args.store_path)
    try:
        n = consolidate_store(
            args.store_path,
            console=console,
            storage_options=store_options,
            merge_registry=not args.no_merge_registry,
            state_url=args.state_url,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{emoji('❌ ')}Error: {e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)
    console.print(f"{emoji('✅ ')}Consolidated metadata for {n} nodes")
    return 0


def _require_local_store(store_path: str, command: str, console: "Console") -> bool:
    """Reject a URL store for the commands that still need a local path.

    Without this the URL is silently mangled by ``Path()`` into something
    like ``s3:/bucket/store`` and fails much later with a confusing error.
    """
    from .remote import is_url

    if not is_url(store_path):
        return True
    console.print(
        f"[red]{emoji('❌ ')}{command} needs a local store path; "
        f"got {store_path}.[/red]\n"
        f"Only the legacy shard-sampling path is local-only — the default "
        f"stats-based stretch and zarr-global-preview --output both work "
        f"against remote stores."
    )
    return False


def zarr_global_preview_command(args):
    """Build global EPSG:4326 RGB pyramid from zone-level embeddings."""
    import warnings
    from rich.console import Console
    from .zarr import build_global_preview

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    console = Console()

    zones = _parse_int_range(args.zones) if args.zones else None
    store_options = _storage_options_for(args, "store", args.store_path)
    output_options = (
        _storage_options_for(args, "output", args.output) if args.output else None
    )

    try:
        build_global_preview(
            store_path=args.store_path,
            year=args.year,
            zones=zones,
            num_levels=args.levels,
            workers=args.workers,
            gamma=args.gamma,
            saturation=args.saturation,
            console=console,
            force=args.force,
            storage_options=store_options,
            output_path=args.output,
            output_storage_options=output_options,
            state_url=args.state_url,
            state_storage_options=(
                _storage_options_for(args, "state", args.state_url)
                if args.state_url
                else None
            ),
            reproject_only=args.reproject_only,
            coarsen_only=args.coarsen_only,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    return 0


def zarr_stretch_command(args):
    """Compute a global cross-zone RGB stretch and persist it to the store."""
    import warnings
    from rich.console import Console

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    console = Console()
    zones = _parse_int_range(args.zones) if args.zones else None

    if args.from_shards:
        # Legacy shard-sampling path: re-reads embeddings, local stores only.
        from .zarr import compute_global_stretch

        if not _require_local_store(args.store_path, "zarr-stretch", console):
            return 1
        compute_global_stretch(
            store_path=Path(args.store_path),
            year=args.year,
            target_samples=args.target_samples,
            max_shards=args.max_shards,
            p_low=args.p_low,
            p_high=args.p_high,
            workers=args.workers,
            zones=zones,
            equalise=not args.no_equalise,
            equalise_breakpoints=args.breakpoints,
            mode=args.mode,
            pca_components=args.pca_components,
            pca_total_bands=args.pca_total_bands,
            pca_rgb_order=args.pca_rgb_order,
            console=console,
        )
        return 0

    # Fast path: aggregate the per-zone statistics collected at fill time.
    # A few MiB of reads; works against local and remote stores alike.
    from .zarr import compute_stretch_from_stats

    store_options = _storage_options_for(args, "store", args.store_path)
    try:
        compute_stretch_from_stats(
            args.store_path,
            year=args.year,
            zones=zones,
            p_low=args.p_low,
            p_high=args.p_high,
            equalise=not args.no_equalise,
            equalise_breakpoints=args.breakpoints,
            mode=args.mode,
            pca_components=args.pca_components,
            pca_rgb_order=args.pca_rgb_order,
            drift_threshold=args.drift_threshold,
            console=console,
            storage_options=store_options,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{emoji('❌ ')}{e}[/red]")
        return 1
    except (ImportError, *_object_store_errors()) as e:
        return _report_store_error(e, console)

    return 0


def print_command(args):
    """Print embedding values at a point from both NPY and zarr sources."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    console = Console()
    lon, lat, year = args.lon, args.lat, args.year
    n = 16

    console.print(f"Embedding at ({lon}, {lat}), year {year}")
    console.print()

    # NPY path: use sample_embeddings_at_points with metadata to see which tile
    from .core import GeoTessera

    gt = GeoTessera()

    try:
        npy_emb, npy_meta = gt.sample_embeddings_at_points(
            [(lon, lat)],
            year=year,
            include_metadata=True,
        )
        npy_emb = npy_emb[0]
        meta = npy_meta[0] if npy_meta[0] else {}
    except Exception as e:
        console.print(f"[red]NPY error: {e}[/red]")
        return 1

    npy_finite = np.isfinite(npy_emb).all()
    console.print("[bold]NPY[/bold] (GeoTessera.sample_embeddings_at_points)")
    console.print(
        f"  Tile: ({meta.get('tile_lon')}, {meta.get('tile_lat')}), "
        f"CRS: {meta.get('crs')}, pixel: ({meta.get('pixel_row')}, {meta.get('pixel_col')})"
    )
    console.print(
        f"  Norm: {np.linalg.norm(npy_emb):.4f}"
        if npy_finite
        else "  [yellow]NaN (no data)[/yellow]"
    )
    console.print()

    # Zarr path: use GeoTesseraZarr.sample_at (handles zone routing)
    from .store import GeoTesseraZarr, _zone_for_lon

    gz = GeoTesseraZarr(args.store)
    zarr_zone = _zone_for_lon(lon)

    try:
        zarr_emb = gz.sample_at(lon, lat, year=year)
    except Exception as e:
        console.print(f"[red]Zarr error: {e}[/red]")
        return 1

    zarr_finite = np.isfinite(zarr_emb).all()
    console.print("[bold]Zarr[/bold] (GeoTesseraZarr.sample_at)")
    console.print(f"  Zone: utm{zarr_zone:02d}")
    console.print(f"  Store: {args.store}")
    console.print(
        f"  Norm: {np.linalg.norm(zarr_emb):.4f}"
        if zarr_finite
        else "  [yellow]NaN (no data)[/yellow]"
    )
    console.print()

    # Table
    table = Table(title=f"First {n} bands")
    table.add_column("Band", style="bold", justify="right")
    table.add_column("NPY", justify="right")
    table.add_column("Zarr", justify="right")
    table.add_column("Diff", justify="right")

    for i in range(n):
        nv = npy_emb[i]
        zv = zarr_emb[i]
        if np.isnan(nv) or np.isnan(zv):
            table.add_row(str(i), f"{nv:.6f}", f"{zv:.6f}", "[dim]NaN[/dim]")
        else:
            d = abs(float(nv) - float(zv))
            style = "" if d == 0 else "[red]" if d > 0.5 else "[yellow]"
            end = "" if not style else "[/]"
            table.add_row(str(i), f"{nv:.6f}", f"{zv:.6f}", f"{style}{d:.6f}{end}")

    console.print(table)

    if npy_finite and zarr_finite:
        max_diff = float(np.max(np.abs(npy_emb - zarr_emb)))
        cosine = float(
            np.dot(npy_emb, zarr_emb)
            / (np.linalg.norm(npy_emb) * np.linalg.norm(zarr_emb))
        )
        if max_diff == 0:
            console.print("\n[green]Exact match[/green]")
        else:
            console.print(
                f"\n[yellow]Max diff: {max_diff:.6f}, cosine similarity: {cosine:.6f}[/yellow]"
            )
    return 0


def verify_tile_command(args):
    """Verify that NPY tile and zarr store produce identical embeddings for a full tile."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    console = Console()

    lon, lat = args.lon, args.lat
    store_url = args.store
    years = _parse_int_range(args.years)

    console.print(f"Verifying full tile at ({lon}, {lat})")
    console.print("  NPY source: geotessera tile download")
    console.print(f"  Zarr source: {store_url}")
    console.print(f"  Years: {years}")
    console.print()

    from .core import GeoTessera
    from .store import GeoTesseraZarr

    gt = GeoTessera()
    gz = GeoTesseraZarr(store_url)

    table = Table(title=f"Tile verification at ({lon}, {lat})")
    table.add_column("Year", style="bold")
    table.add_column("Tile size")
    table.add_column("Both valid")
    table.add_column("Breakdown")
    table.add_column("Overlap mean")
    table.add_column("Status", style="bold")

    all_match = True

    for year in years:
        if year not in gz.years:
            table.add_row(
                str(year), "-", "-", "-", "-", "[yellow]not in store[/yellow]"
            )
            continue

        # Download the NPY tile covering this point
        try:
            tiles_needed = gt.registry.load_blocks_for_region(
                (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005),
                year,
            )
            if not tiles_needed:
                table.add_row(str(year), "-", "-", "-", "-", "[yellow]no tile[/yellow]")
                continue
            coords = {(tlon, tlat) for (_, tlon, tlat) in tiles_needed}
            tile_map = gt._ensure_tiles_available(
                required_coords=coords,
                year=year,
                auto_download=True,
                bbox=None,
            )
            # Pick the tile that matches the registry result, not just any available tile
            target_coord = next(iter(coords))
            tile = tile_map.get(target_coord)
            if tile is None or not tile.is_available():
                table.add_row(
                    str(year), "-", "-", "-", "-", "[yellow]tile not available[/yellow]"
                )
                continue
            npy_emb = tile.load_embedding()  # (H, W, 128) float32
        except Exception as e:
            table.add_row(str(year), "-", "-", "-", "-", f"[red]NPY error: {e}[/red]")
            all_match = False
            continue

        h, w, _ = npy_emb.shape
        tile_transform = tile.transform
        tile_crs = tile.crs

        # Map NPY tile pixels to zarr indices using the same logic as
        # zarr-fill: tile origin → zarr offset via round(), then add
        # the pixel's position within the tile.
        #
        # zarr_row = round((zarr_origin_y - tile_origin_y) / px) + tile_row
        # zarr_col = round((tile_origin_x - zarr_origin_x) / px) + tile_col
        try:
            px = tile_transform.a
            ds = gz.open_zone(lon=lon)
            t_attr = ds.attrs["spatial:transform"]
            zarr_ox, zarr_oy = t_attr[2], t_attr[5]

            # Where zarr-fill placed tile pixel (0,0)
            tile_row0 = round((zarr_oy - tile_transform.f) / px)
            tile_col0 = round((tile_transform.c - zarr_ox) / px)

            # Read the tile-sized slab: zarr[tile_row0:tile_row0+h, tile_col0:tile_col0+w]
            sub = ds.isel(
                time=gz.years.index(year),
                y=slice(tile_row0, tile_row0 + h),
                x=slice(tile_col0, tile_col0 + w),
            )
            scales = sub["scales"].values
            emb_int8 = sub["embeddings"].values
            zarr_emb = ds.tessera.dequantise(emb_int8, scales)
        except Exception as e:
            table.add_row(
                str(year), f"{h}x{w}", "-", "-", "-", f"[red]Zarr error: {e}[/red]"
            )
            all_match = False
            continue

        # Compare all valid pixels
        npy_valid = np.isfinite(npy_emb).all(axis=2)
        zarr_valid = np.isfinite(zarr_emb).all(axis=2)
        both_valid = npy_valid & zarr_valid
        n_valid = int(both_valid.sum())

        if n_valid == 0:
            table.add_row(
                str(year), f"{h}x{w}", "0", "-", "-", "[yellow]no valid pixels[/yellow]"
            )
            continue

        # Categorise every pixel
        exact_match = both_valid & (np.abs(npy_emb - zarr_emb).max(axis=2) == 0)
        water_masked = npy_valid & np.isnan(scales)  # NPY has data, zarr is NaN (water)
        overlap_diff = both_valid & ~exact_match  # both finite, different values
        npy_nodata = ~npy_valid  # NPY is NaN

        n_exact = int(exact_match.sum())
        n_water = int(water_masked.sum())
        n_overlap = int(overlap_diff.sum())

        table.add_row(
            str(year),
            f"{h}x{w}",
            f"{n_valid:,}",
            f"{n_exact:,} exact"
            if n_exact == n_valid
            else f"[green]{n_exact:,}[/green] match, [cyan]{n_water:,}[/cyan] water, [yellow]{n_overlap:,}[/yellow] overlap",
            f"{float(np.abs(npy_emb[overlap_diff] - zarr_emb[overlap_diff]).mean()):.4f}"
            if n_overlap > 0
            else "-",
            "[green]PASS[/green]"
            if n_overlap == 0 and n_water == 0
            else f"[yellow]{n_water + n_overlap:,} differ[/yellow]",
        )
        if n_overlap > 0 or n_water > 0:
            all_match = False

        # Build RGB preview from first 3 NPY embedding bands (stretch to 0-255)
        import rasterio

        emb_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for b in range(3):
            band = npy_emb[:, :, b].copy()
            valid_band = band[npy_valid]
            if len(valid_band) > 0:
                lo = np.percentile(valid_band, 2)
                hi = np.percentile(valid_band, 98)
                if hi > lo:
                    band = np.clip((band - lo) / (hi - lo), 0, 1)
                else:
                    band = np.zeros_like(band)
            emb_rgb[:, :, b] = (band * 255).astype(np.uint8)
        emb_rgb[~npy_valid] = 0

        # Diagnostic overlay: green=match, cyan=water, yellow=overlap
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[exact_match] = [0, 200, 0, 100]  # green, light
        rgba[water_masked] = [0, 200, 200, 200]  # cyan, strong
        rgba[overlap_diff] = [255, 200, 0, 200]  # yellow, strong
        rgba[npy_nodata] = [0, 0, 0, 0]  # transparent

        # Fetch satellite basemap and composite embedding RGB + overlay on top
        from rasterio.warp import reproject, Resampling

        from datetime import datetime as _dt

        ts = _dt.now().strftime("%H%M%S")
        tif_path = f"verify_{year}_{ts}.tif"
        try:
            import contextily as cx
            from pyproj import Transformer as ProjTransformer

            # Get tile extent in WGS84 for basemap fetch
            epsg = int(str(tile_crs).split(":")[1])
            to_wgs = ProjTransformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
            tile_east = tile_transform.c + w * px
            tile_south = tile_transform.f - h * px
            lon_w, lat_s = to_wgs.transform(tile_transform.c, tile_south)
            lon_e, lat_n = to_wgs.transform(tile_east, tile_transform.f)

            # Fetch basemap in Web Mercator (contextily native)
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = tmp.name
            cx.bounds2raster(
                lon_w,
                lat_s,
                lon_e,
                lat_n,
                ll=True,
                path=tmp_path,
                source=cx.providers.Esri.WorldImagery,
                zoom="auto",
            )
            basemap_ds = rasterio.open(tmp_path)
            basemap = basemap_ds.read()  # (3, bh, bw) or (4, bh, bw)
            basemap_extent = basemap_ds.transform
            basemap_crs = basemap_ds.crs
            basemap_ds.close()
            os.unlink(tmp_path)

            # Reproject basemap to tile's UTM CRS at the tile's resolution
            dst_transform = tile_transform
            n_bands_bm = min(3, basemap.shape[0])  # may have alpha
            base_utm = np.zeros((3, h, w), dtype=np.uint8)
            for band in range(n_bands_bm):
                reproject(
                    source=basemap[band],
                    destination=base_utm[band],
                    src_transform=basemap_extent,
                    src_crs=basemap_crs,
                    dst_transform=dst_transform,
                    dst_crs=f"EPSG:{epsg}",
                    resampling=Resampling.bilinear,
                )

            # Composite: basemap → embedding bands (40% opacity) → diagnostic overlay
            composite = base_utm.copy()  # (3, H, W)

            # Blend embedding RGB onto basemap at 40% opacity
            emb_alpha = 0.4
            for band in range(3):
                bg = composite[band].astype(np.float32)
                fg = emb_rgb[:, :, band].astype(np.float32)
                composite[band] = (fg * emb_alpha + bg * (1 - emb_alpha)).astype(
                    np.uint8
                )

            # Blend diagnostic overlay on top
            diag_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            for band in range(3):
                fg = rgba[:, :, band].astype(np.float32)
                bg = composite[band].astype(np.float32)
                composite[band] = (fg * diag_alpha + bg * (1 - diag_alpha)).astype(
                    np.uint8
                )

            # Draw crosshair marker at the query lon/lat
            from pyproj import Transformer as _ProjT

            _to_utm = _ProjT.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            mark_e, mark_n = _to_utm.transform(lon, lat)
            mark_row, mark_col = rasterio.transform.rowcol(
                tile_transform, mark_e, mark_n
            )
            if 0 <= mark_row < h and 0 <= mark_col < w:
                arm = 12
                marker_color = [255, 0, 0]
                for dr in range(-arm, arm + 1):
                    for dt in range(-1, 2):
                        for r, c in [
                            (mark_row + dt, mark_col + dr),
                            (mark_row + dr, mark_col + dt),
                        ]:
                            if 0 <= r < h and 0 <= c < w:
                                for b in range(3):
                                    composite[b, r, c] = marker_color[b]
            else:
                console.print(
                    f"  [yellow]Crosshair out of tile bounds: pixel ({mark_row}, {mark_col}), tile {h}x{w}[/yellow]"
                )

            # Write composited GeoTIFF (RGB, no alpha)
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=3,
                dtype="uint8",
                crs=str(tile_crs),
                transform=tile_transform,
                compress="lzw",
            ) as dst:
                dst.write(composite)
            console.print(f"  Wrote [bold]{tif_path}[/bold] (with satellite basemap)")

        except ImportError:
            # contextily not available — composite embedding bands + overlay
            composite = np.zeros((3, h, w), dtype=np.uint8)
            for band in range(3):
                composite[band] = emb_rgb[:, :, band]
            diag_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            for band in range(3):
                fg = rgba[:, :, band].astype(np.float32)
                bg = composite[band].astype(np.float32)
                composite[band] = (fg * diag_alpha + bg * (1 - diag_alpha)).astype(
                    np.uint8
                )
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=3,
                dtype="uint8",
                crs=str(tile_crs),
                transform=tile_transform,
                compress="lzw",
            ) as dst:
                dst.write(composite)
            console.print(
                f"  Wrote [bold]{tif_path}[/bold] (embedding bands + overlay)"
            )

    console.print(table)

    if all_match:
        console.print("\n[green]All years match exactly.[/green]")
        return 0
    else:
        console.print(
            "\n[yellow]Some pixels differ (see verify_{year}.tif overlays).[/yellow]"
        )
        return 1


def main():
    """Main entry point for the geotessera-registry CLI tool."""
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

    # The object-store libraries log at INFO on every client build; left on,
    # they interleave with the progress bars during a fill.
    from .remote import quieten_dependency_logging

    quieten_dependency_logging()

    parser = argparse.ArgumentParser(
        description="GeoTessera Registry Management Tool - Generate and maintain Pooch registry files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List existing registry files
  geotessera-registry list /path/to/data
  
  # Generate SHA256 checksums for embeddings and TIFF files
  geotessera-registry hash /path/to/v1

  # This will:
  # - Create SHA256 files in each grid subdirectory under global_0.1_degree_representation/YYYY/
  # - Create SHA256SUM file in global_0.1_degree_tiff_all/ using chunked processing
  # - Skip directories that already have SHA256 files (use --force to regenerate)

  # Dry run: see what would be recalculated without modifying files
  geotessera-registry hash /path/to/v1 --dry-run

  # Check a specific year only
  geotessera-registry hash /path/to/v1 --dry-run --year 2024

  # Force regeneration of all checksums
  geotessera-registry hash /path/to/v1 --force
  
  # Scan existing SHA256 checksum files and generate both parquet database and registry files
  geotessera-registry scan /path/to/v1
  
  # This will:
  # - Create a Parquet database with all tile metadata (lat/lon/year/hash/mtime)
  # - Read SHA256 files from grid subdirectories and generate block-based registry files
  # - Read SHA256SUM file from TIFF directory and generate landmask registry files
  # - Use atomic file writing (temp files) for cron-safe operation
  
  # Check filesystem structure integrity
  geotessera-registry check /path/to/v1
  
  # This will:
  # - Validate all directory structures are correct
  # - Check that all SHA256 files exist
  # - Verify that embedding and scales files exist
  # - Optionally verify SHA256 hashes (use --verify-hashes)
  
  # Analyze registry changes and create a git commit with detailed summary
  geotessera-registry commit

  # This will:
  # - Analyze git changes in registry files
  # - Summarize changes by year (tiles added/removed/modified)
  # - Stage registry files and create a commit with detailed message

  # Export Parquet registry to text manifests for backwards compatibility
  geotessera-registry export-manifests /path/to/v1

  # This will:
  # - Read registry.parquet and landmasks.parquet
  # - Generate block-based text registry files in registry/embeddings/ and registry/landmasks/
  # - Create separate entries for .npy and _scales.npy with their respective hashes
  # - Useful for maintaining the tessera-manifests repository

  # Export to custom output directory
  geotessera-registry export-manifests /path/to/v1 --output-dir ~/src/git/ucam-eo/tessera-manifests

  # Scan year directories for embedding tiles and create an inventory
  geotessera-registry file-scan /path/to/embeddings

  # This will:
  # - Scan for year subdirectories (e.g., 2024/, 2023/) in the input directory
  # - Recursively scan each year directory for grid*.npy files
  # - Extract lon/lat coordinates from filenames
  # - Record year, modification times for both grid*.npy and *_scales.npy files
  # - Record full real paths for embedding files
  # - Generate a parquet file with: year, lon, lat, directory, grid_path, scales_path, grid_mtime, scales_mtime, file sizes
  # - Useful for finding potential duplicate embeddings across machines

  # Specify custom output path
  geotessera-registry file-scan /path/to/embeddings --output /path/to/inventory.parquet

  # Spider the Source Cooperative repository to rebuild a per-dataset manifest.
  # Dataset dirs encode (version, variant): v1, v1.1-cam, v2-2B-L~beta1, ...
  geotessera-registry s3scan s3://tessera/tessera/npy/v1.1-cam/ --landmasks-uri s3://tessera/tessera/landmasks/v1.1/

  # Or rescan every dataset in one go:
  geotessera-registry s3scan s3://tessera/tessera/npy/ --landmasks-uri s3://tessera/tessera/landmasks/

  # This will:
  # - List year subprefixes (e.g. 2024/, 2023/) via anonymous ListObjectsV2 HTTPS calls
  # - Paginate listings for each year in parallel and pair grid_*.npy with grid_*_scales.npy
  # - Use S3 LastModified/Size for the mtime/size columns (no per-object stat calls)
  # - Emit one manifest.parquet per dataset dir with the file-scan schema plus s3:// URLs in path columns

  # Check multiple inventory files for duplicate coordinates (year/lon/lat)
  geotessera-registry file-check machine1_inventory.parquet machine2_inventory.parquet machine3_inventory.parquet

  # This will:
  # - Load all specified parquet files (generated by file-scan)
  # - Find any year/lon/lat coordinates that appear in multiple files or multiple times
  # - Display duplicates with their year, source files, full paths, directories, and modification times
  # - Show summary statistics by parquet file
  # - Useful for identifying duplicate embeddings across generation machines

This tool is intended for GeoTessera data maintainers to generate the registry
files that are distributed with the package. End users typically don't need
to use this tool.

Note: This tool creates block-based registries for efficient lazy loading:
  - Embeddings: Organized into 5x5 degree blocks (embeddings_YYYY_lonX_latY.txt)
  - Landmasks: Organized into 5x5 degree blocks (landmasks_lonX_latY.txt)
  - Each block contains ~2,500 tiles instead of one massive registry
  - Registry files are created in the registry/ subdirectory

Directory Structure:
  The commands expect to find these subdirectories:
  - global_0.1_degree_representation/  (contains .npy files organized by year)
  - global_0.1_degree_tiff_all/        (contains .tiff files in flat structure)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List existing registry files")
    list_parser.add_argument(
        "base_dir", help="Base directory to scan for registry files"
    )
    list_parser.set_defaults(func=list_command)

    # Hash command
    hash_parser = subparsers.add_parser(
        "hash", help="Generate SHA256 checksums for embeddings and TIFF files"
    )
    hash_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories",
    )
    hash_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all checksums, even if SHA256 files already exist",
    )
    hash_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recalculated without actually modifying files",
    )
    hash_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Only process a specific year (e.g., 2024). Applies to embeddings only.",
    )
    hash_parser.set_defaults(func=hash_command)

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan existing SHA256 checksum files and generate both parquet database and registry files",
    )
    scan_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories with checksum files",
    )
    scan_parser.add_argument(
        "--registry-dir",
        type=str,
        default=None,
        help="Output directory for registry and parquet files (default: same as base_dir)",
    )
    scan_parser.add_argument(
        "--only",
        choices=["embeddings", "landmasks"],
        default=None,
        help="Only generate the specified parquet database (default: both)",
    )
    scan_parser.set_defaults(func=scan_command)

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check the integrity of tessera filesystem structure and validate files",
    )
    check_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation subdirectory",
    )
    check_parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Recalculate and verify SHA256 hashes (slower but thorough)",
    )
    check_parser.set_defaults(func=check_command)

    # Commit command
    commit_parser = subparsers.add_parser(
        "commit",
        help="Analyze registry changes and create a git commit with detailed summary",
    )
    commit_parser.set_defaults(func=commit_command)

    # Export-manifests command
    export_parser = subparsers.add_parser(
        "export-manifests",
        help="Convert Parquet registry files to Pooch-format text manifests for backwards compatibility",
    )
    export_parser.add_argument(
        "input_dir",
        help="Directory containing registry.parquet and landmasks.parquet files",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for text manifest files (default: INPUT_DIR/registry)",
    )
    export_parser.set_defaults(func=export_manifests_command)

    # File-scan command
    file_scan_parser = subparsers.add_parser(
        "file-scan",
        help="Recursively scan year directories for embedding tiles and generate an inventory parquet file",
    )
    file_scan_parser.add_argument(
        "input_dir",
        help="Base directory containing year subdirectories (e.g., 2024/, 2023/) with embedding tiles",
    )
    file_scan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: INPUT_DIR/embedding_inventory.parquet)",
    )
    file_scan_parser.set_defaults(func=file_scan_command)

    # S3-scan command
    s3scan_parser = subparsers.add_parser(
        "s3scan",
        help="Spider an S3 bucket for embedding tiles across versions and "
        "variants, and write a manifest parquet",
    )
    s3scan_parser.add_argument(
        "s3_uri",
        help="S3 URI at any level: tree root (discovers all datasets), "
        "dataset dir (e.g. s3://tessera/tessera/npy/v1.1-cam/ — the "
        "directory name encodes the (version, variant) pair), or legacy "
        "variant dir "
        "(e.g. s3://bucket/v1.1/global_0.1_degree_representation.cambridge/)",
    )
    s3scan_parser.add_argument(
        "--endpoint-url",
        type=str,
        default=TESSERA_MIRROR_ENDPOINT,
        help="S3-compatible endpoint for path-style listing requests "
        f"(default: {TESSERA_MIRROR_ENDPOINT})",
    )
    s3scan_parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant recorded in the manifest when the layout carries no "
        "variant in the path, i.e. years sit directly under a bare version "
        "dir like v1/ (default: the version's default variant, e.g. vultr "
        "for v1). Suffixed dataset dirs (v1.1-cam, v2-2B-L~beta1) encode "
        "the variant in their name and ignore this.",
    )
    s3scan_parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent lon-shard listings (default: 32). "
        "Each year is split into ~360 lon-prefix shards listed in parallel.",
    )
    s3scan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory under which per-version manifests are written "
        "as {output}/{version_path}/manifest.parquet (default: current dir). "
        "Suitable for 'aws s3 cp --recursive <output>/ s3://<bucket>/'.",
    )
    s3scan_parser.add_argument(
        "--no-landmasks",
        action="store_true",
        help="Skip scanning landmask TIFFs and writing landmasks.parquet",
    )
    s3scan_parser.add_argument(
        "--landmasks-uri",
        type=str,
        default=None,
        help="S3 URI of the landmask tree when it is not under the embeddings "
        "prefix, e.g. s3://tessera/tessera/landmasks/ or a version dir within "
        "it. With this set, the landmask scan runs even when the embeddings "
        "prefix yields no tiles, so a landmasks-only regeneration is possible.",
    )
    s3scan_parser.set_defaults(func=s3scan_command)

    # File-check command
    file_check_parser = subparsers.add_parser(
        "file-check",
        help="Check multiple inventory parquet files for duplicate year/lon/lat coordinates",
    )
    file_check_parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Parquet files to check for duplicates (output from file-scan command)",
    )
    file_check_parser.set_defaults(func=file_check_command)

    # Zarr-init command
    zarr_init_parser = subparsers.add_parser(
        "zarr-init",
        help="Create an empty tessera store with time dimension",
    )
    zarr_init_parser.add_argument(
        "base_dir",
        help="Base directory containing downloaded tile data, or a URL of a "
        "repository in the published layout (e.g. s3://bucket/tessera)",
    )
    zarr_init_parser.add_argument(
        "--years",
        required=True,
        help="Year range (e.g. 2017-2025 or 2017,2019,2024)",
    )
    zarr_init_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output store path or URL (e.g. tessera.zarr, "
        "s3://bucket/tessera.zarr)",
    )
    zarr_init_parser.add_argument(
        "--stretch-sample-size",
        type=int,
        default=None,
        help="Per-(zone, year) capacity of the raw pixel sample kept for the "
        "global stretch quantiles (default: 20000)",
    )
    _add_source_args(zarr_init_parser)
    _add_storage_args(zarr_init_parser, "source", "Tile source")
    _add_state_arg(zarr_init_parser)
    _add_storage_args(zarr_init_parser, "store", "Output store", writable=True)
    zarr_init_parser.set_defaults(func=zarr_init_command)

    # Zarr-fill command
    zarr_fill_parser = subparsers.add_parser(
        "zarr-fill",
        help="Incrementally fill a tessera store with tile data",
        description="Fill a tessera store, optionally streaming tiles from a "
        "remote bucket into a store on another. Restricting a run to one UTM "
        "zone with --zones makes it safe to run many fills concurrently "
        "against the same store: each zone owns its own group, tracking file "
        "and lock. Run zarr-consolidate once after the sweep.",
    )
    zarr_fill_parser.add_argument(
        "base_dir",
        help="Base directory containing downloaded tile data, or a URL of a "
        "repository in the published layout (e.g. s3://bucket/tessera)",
    )
    zarr_fill_parser.add_argument(
        "store_path",
        type=str,
        nargs="?",
        default=None,
        help="Path or URL of an existing tessera store",
    )
    zarr_fill_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to fill (default: all years)",
    )
    zarr_fill_parser.add_argument(
        "--zones",
        default=None,
        help="Zone numbers to fill (e.g. 30 or 29-34). "
        "Default: all initialised zones",
    )
    zarr_fill_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 4)",
    )
    zarr_fill_parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Rewrite the root consolidated metadata when done. Unsafe while "
        "sibling zone fills are running. Default: on for a whole-store fill, "
        "off when --zones is given.",
    )
    zarr_fill_parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Never rewrite the root consolidated metadata.",
    )
    zarr_fill_parser.add_argument(
        "--skip-existing-shards",
        action="store_true",
        help="Deprecated and now the default; accepted so existing scripts "
        "keep working.",
    )
    zarr_fill_parser.add_argument(
        "--rewrite-existing-shards",
        action="store_true",
        help="Rebuild shards that are already in the store instead of "
        "skipping them. Needed only when the tile inventory has grown, since "
        "a newly-added tile falls inside an existing shard.",
    )
    zarr_fill_parser.add_argument(
        "--no-stretch-stats",
        action="store_true",
        help="Skip folding stretch statistics into the zone group as shards "
        "are written (escape hatch; the global stretch then needs a backfill "
        "or the legacy shard-sampling path)",
    )
    zarr_fill_parser.add_argument(
        "--backfill-stretch-stats",
        action="store_true",
        help="Rebuild the selected zones' stretch statistics by scanning "
        "their existing shards, instead of ingesting tiles. The repair path "
        "for stores filled before fill-time collection existed, interrupted "
        "fills, and suspected double-counting. With this flag the tile "
        "source is not read; `zarr-fill <store> --backfill-stretch-stats` "
        "works with the store as the only positional.",
    )
    zarr_fill_parser.add_argument(
        "--spill-dir",
        type=str,
        default=None,
        help="Memory-map each worker's shard buffers under this directory "
        "instead of holding them in RAM. Trades disk for roughly 2 GiB of "
        "anonymous memory per worker, which is what the OOM killer targets — "
        "worth it on a memory-tight box with spare disk.",
    )
    zarr_fill_parser.add_argument(
        "--force-lock",
        action="store_true",
        help="Deprecated no-op: fills no longer take locks. The store's own "
        "shard objects are the only state, so a dead run leaves nothing to "
        "take over.",
    )
    _add_source_args(zarr_fill_parser)
    _add_storage_args(zarr_fill_parser, "source", "Tile source")
    _add_state_arg(zarr_fill_parser)
    _add_storage_args(zarr_fill_parser, "store", "Output store", writable=True)
    zarr_fill_parser.set_defaults(func=zarr_fill_command)

    # Zarr-scan command
    zarr_scan_parser = subparsers.add_parser(
        "zarr-scan",
        help="Report how much of a store still needs filling",
        description="Inventory a store's shards against the manifest without "
        "writing anything. Each shard is classified written, missing, or "
        "empty (no manifest tiles fall in it — ocean or outside coverage), "
        "so the percentages are over land rather than over the zone's "
        "bounding box. Prints per-zone/year and per-year summaries and can "
        "write the full index as parquet.",
    )
    # Same positional order as zarr-init/zarr-fill — source first, store
    # second — with the source optional. argparse binds a lone argument to
    # store_path, so `zarr-scan <store>` and `zarr-scan <mirror> <store>`
    # both do the obvious thing.
    zarr_scan_parser.add_argument(
        "base_dir",
        nargs="?",
        default=None,
        help="Optional tile mirror or repository URL. Without it, land "
        "coverage comes from the landmask registry, which is all that is "
        "needed and far smaller than the manifest. Give it to measure "
        "against each year's actual embedding coverage instead.",
    )
    zarr_scan_parser.add_argument(
        "store_path",
        type=str,
        help="Path or URL of an existing tessera store",
    )
    zarr_scan_parser.add_argument(
        "--years",
        default=None,
        help="Years to scan (e.g. 2024 or 2017-2025). Default: every year "
        "on the store's time axis",
    )
    zarr_scan_parser.add_argument(
        "--zones",
        default=None,
        help="Zones to scan (e.g. 30 or 29-34). Default: every zone",
    )
    zarr_scan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the per-shard index here as parquet (path or URL)",
    )
    zarr_scan_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each zone/year as it is scanned",
    )
    _add_source_args(zarr_scan_parser)
    _add_state_arg(zarr_scan_parser)
    _add_storage_args(zarr_scan_parser, "source", "Tile source")
    _add_storage_args(zarr_scan_parser, "store", "Store")
    zarr_scan_parser.set_defaults(func=zarr_scan_command)

    # Zarr-extend command
    zarr_extend_parser = subparsers.add_parser(
        "zarr-extend",
        help="Append new years to an existing store's time axis",
        description="Grow a store's time dimension so a new year can be "
        "filled. The time axis is chunked one year per chunk, so this is a "
        "metadata-only edit: existing data is never rewritten. Years may "
        "only be appended to the end. Run it with no fills in flight, then "
        "zarr-fill the new year.",
    )
    zarr_extend_parser.add_argument(
        "store_path",
        type=str,
        help="Path or URL of an existing tessera store",
    )
    zarr_extend_parser.add_argument(
        "--years",
        required=True,
        help="Year(s) to append (e.g. 2026 or 2026-2027)",
    )
    zarr_extend_parser.add_argument(
        "--zones",
        default=None,
        help="Restrict to these zones (e.g. 29-34). Default: every zone. "
        "Leaving zones behind makes the store's time axes disagree, so only "
        "use this to finish an interrupted run.",
    )
    zarr_extend_parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip re-consolidating the root metadata. Array metadata has "
        "changed, so readers need a consolidate before they see the new year.",
    )
    zarr_extend_parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated no-op: fills no longer take locks. Do not extend "
        "while a fill is in flight.",
    )
    _add_state_arg(zarr_extend_parser)
    _add_storage_args(zarr_extend_parser, "store", "Store", writable=True)
    zarr_extend_parser.set_defaults(func=zarr_extend_command)

    # Zarr-consolidate command
    zarr_consolidate_parser = subparsers.add_parser(
        "zarr-consolidate",
        help="Re-consolidate store metadata after in-place changes",
        description="Rewrite a store's root consolidated metadata and merge "
        "the per-zone ingestion registries. This is the single-writer step "
        "that finishes a parallel zone sweep — run it once, with no fills in "
        "flight.",
    )
    zarr_consolidate_parser.add_argument(
        "store_path",
        type=str,
        help="Path or URL of an existing tessera store "
        "(e.g. tessera.zarr or s3://bucket/store.zarr)",
    )
    zarr_consolidate_parser.add_argument(
        "--no-merge-registry",
        action="store_true",
        help="Skip merging _registry/ per-zone files into _registry.parquet",
    )
    _add_state_arg(zarr_consolidate_parser)
    _add_storage_args(zarr_consolidate_parser, "store", "Store", writable=True)
    zarr_consolidate_parser.set_defaults(func=zarr_consolidate_command)

    # Zarr-global-preview command
    zarr_gp_parser = subparsers.add_parser(
        "zarr-global-preview",
        help="Build global EPSG:4326 RGB pyramid from embeddings",
    )
    zarr_gp_parser.add_argument(
        "store_path",
        type=str,
        help="Path to tessera store",
    )
    zarr_gp_parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year to use for the preview (default: 2024)",
    )
    zarr_gp_parser.add_argument(
        "--zones",
        default=None,
        help="Zone numbers to include (e.g. 29-34). Default: all",
    )
    zarr_gp_parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of pyramid levels (default: 10)",
    )
    zarr_gp_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    zarr_gp_parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess zones even if completion markers exist",
    )
    zarr_gp_parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Per-channel gamma applied after normalisation (default: 1.0). "
        "Values < 1.0 brighten midtones (0.6–0.8 is typical for EO previews); "
        "values > 1.0 darken. Combine with `zarr-stretch` for best colour pop.",
    )
    zarr_gp_parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Chroma multiplier applied AFTER gamma (default: 1.0). Each "
        "pixel is decomposed into luma + chroma and the chroma scaled. "
        "Try 1.5–2.5 if colours look washed out. Beyond ~3 most pixels "
        "start clipping at the colour-cube edges.",
    )
    zarr_gp_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path or URL of the store to hold the pyramid, with its own "
        "--output-* credentials. Embeddings stream from the source store "
        "read-only (sub-shard byte ranges, no copy), so this can write to a "
        "bucket while reading from a read-only mirror. Default: write the "
        "pyramid into the source store itself.",
    )
    zarr_gp_parser.add_argument(
        "--reproject-only",
        action="store_true",
        help="Render this zone's level 0 and coarsen only as far as the "
        "depth where zones stay disjoint. Safe to run concurrently across "
        "zones that share no level-0 chunks — an odd-numbered round then an "
        "even-numbered one covers all 60. Finish with --coarsen-only.",
    )
    zarr_gp_parser.add_argument(
        "--coarsen-only",
        action="store_true",
        help="Build the remaining coarse levels in one global pass, without "
        "reprojecting. Single-writer: run once after every zone's "
        "--reproject-only has finished.",
    )
    zarr_gp_parser.add_argument(
        "--state-url",
        type=str,
        default=None,
        help="Where per-zone resume markers live (default: <output>.build, "
        "alongside the pyramid). Point it at a local directory to keep "
        "build bookkeeping out of a published bucket.",
    )
    _add_storage_args(zarr_gp_parser, "store", "Store")
    _add_storage_args(zarr_gp_parser, "output", "Pyramid output", writable=True)
    _add_storage_args(zarr_gp_parser, "state", "Build state", writable=True)
    zarr_gp_parser.set_defaults(func=zarr_global_preview_command)

    # Zarr-stretch command
    zarr_stretch_parser = subparsers.add_parser(
        "zarr-stretch",
        help="Compute a single global RGB stretch across all zones and "
        "store it on the Zarr root so zarr-global-preview can produce a "
        "seamless mosaic (no per-zone colour discontinuities).",
    )
    zarr_stretch_parser.add_argument(
        "store_path", type=str, help="Path to tessera store"
    )
    zarr_stretch_parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year slice to stretch (default: 2024)",
    )
    zarr_stretch_parser.add_argument(
        "--target-samples",
        type=int,
        default=2_000_000,
        help="Stop after this many valid (non-NaN, non-+inf) pixels are "
        "collected across all zones (default: 2_000_000). PCA mode "
        "benefits from more samples — 2-5M is reasonable.",
    )
    zarr_stretch_parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Hard cap on shards visited (default: unbounded — usually only "
        "a few hundred shards are needed for 1M valid pixels)",
    )
    zarr_stretch_parser.add_argument(
        "--p-low",
        type=float,
        default=2.0,
        help="Low percentile (default: 2)",
    )
    zarr_stretch_parser.add_argument(
        "--p-high",
        type=float,
        default=98.0,
        help="High percentile (default: 98)",
    )
    zarr_stretch_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel I/O threads (default: 8)",
    )
    zarr_stretch_parser.add_argument(
        "--zones",
        default=None,
        help="Limit to specific UTM zones (e.g. 29-34). Default: all",
    )
    zarr_stretch_parser.add_argument(
        "--no-equalise",
        action="store_true",
        help="Disable per-channel CDF (histogram) equalisation. With "
        "equalisation on (default), pixel values are remapped through the "
        "sample's CDF so output bytes are uniformly distributed across "
        "0..255 — usually gives much better colour pop than a linear "
        "stretch alone. Use this flag to fall back to plain "
        "(x - min)/(max - min).",
    )
    zarr_stretch_parser.add_argument(
        "--breakpoints",
        type=int,
        default=257,
        help="Number of CDF breakpoints per channel when equalising "
        "(default: 257). Higher = smoother histogram but more stored bytes.",
    )
    zarr_stretch_parser.add_argument(
        "--mode",
        choices=("bands", "pca"),
        default="bands",
        help="'bands' (default): use embedding bands 0, 1, 2 directly. "
        "'pca': sample all 128 bands and learn 3 orthogonal axes (PC1→R, "
        "PC2→G, PC3→B). PCA fixes the 'washed out' look you get when the "
        "raw bands are correlated, because the projection guarantees the "
        "output channels are mathematically decorrelated.",
    )
    zarr_stretch_parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of principal components to keep when --mode=pca "
        "(default: 3 → RGB).",
    )
    zarr_stretch_parser.add_argument(
        "--pca-total-bands",
        type=int,
        default=128,
        help="Number of embedding bands to consider in PCA "
        "(default: 128, the full Tessera dimensionality).",
    )
    zarr_stretch_parser.add_argument(
        "--pca-rgb-order",
        type=str,
        default="123",
        help="Permutation controlling which principal component lands in "
        "each output channel. Position is the output channel (R, G, B "
        "left-to-right); value is the 1-indexed PC. Defaults to '123' "
        "(PC1->R, PC2->G, PC3->B). Use '213' to swap R and G (PC2->R, "
        "PC1->G, PC3->B), '321' to fully reverse, etc.",
    )
    zarr_stretch_parser.add_argument(
        "--from-shards",
        action="store_true",
        help="Force the legacy shard-sampling path (re-reads embeddings; "
        "local stores only). Default: aggregate the per-zone statistics "
        "collected at fill time — a few MiB of reads, remote-capable.",
    )
    zarr_stretch_parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.25,
        help="Maximum relative Frobenius distance between the stats-derived "
        "and sample-derived covariances before warning of stale statistics "
        "(default: 0.25)",
    )
    _add_storage_args(zarr_stretch_parser, "store", "Store", writable=True)
    zarr_stretch_parser.set_defaults(func=zarr_stretch_command)

    # Verify-tile command
    verify_parser = subparsers.add_parser(
        "verify-tile",
        help="Verify NPY tiles and zarr store produce identical embeddings at a point",
    )
    verify_parser.add_argument(
        "--lon",
        type=float,
        default=-2.969398,
        help="Longitude (default: -2.969398, Liverpool)",
    )
    verify_parser.add_argument(
        "--lat",
        type=float,
        default=53.434288,
        help="Latitude (default: 53.434288, Liverpool)",
    )
    verify_parser.add_argument(
        "--years",
        default="2017,2024,2025",
        help="Years to verify (default: 2017,2024,2025)",
    )
    verify_parser.add_argument(
        "--store",
        default=zarr_store_url("v1"),
        help="Zarr store URL",
    )
    verify_parser.set_defaults(func=verify_tile_command)

    # Print command
    print_parser = subparsers.add_parser(
        "print",
        help="Print embedding values at a point from both NPY and zarr",
    )
    print_parser.add_argument("--lon", type=float, required=True, help="Longitude")
    print_parser.add_argument("--lat", type=float, required=True, help="Latitude")
    print_parser.add_argument("--year", type=int, required=True, help="Year")
    print_parser.add_argument(
        "--store",
        default=zarr_store_url("v1"),
        help="Zarr store URL",
    )
    print_parser.set_defaults(func=print_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute the command. Commands return a shell exit status; propagate it
    # so a failed zone in a parallel sweep is visible to the orchestrator
    # rather than silently reported as success.
    try:
        raise SystemExit(args.func(args) or 0)
    except KeyboardInterrupt:
        # A long fill is routinely interrupted; report it as the shell
        # convention rather than as two pages of traceback.
        console.print(f"\n[yellow]{emoji('⚠️  ')}Interrupted.[/yellow]")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
