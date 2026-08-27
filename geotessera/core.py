"""Core GeoTessera functionality.

The library focusses on:
1. Downloading Tessera tiles for lat/lon bounding boxes to numpy arrays
2. Exporting tiles to individual GeoTIFF files with accurate metadata
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Generator, Iterable
import json
import logging
import numpy as np
import geopandas as gpd

from .registry import (
    Registry,
    EMBEDDINGS_DIR_NAME,
    UnwritableDestination,
    tile_to_geotiff_path,
    tile_from_world,
)

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


def dequantize_embedding(
    quantized_embedding: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Dequantize embedding by multiplying with scale factors.

    This is the standard dequantization process for Tessera embeddings,
    which are stored as quantized int8 values with corresponding float32 scales.

    Args:
        quantized_embedding: Quantized embedding array (typically int8), shape (H, W, 128)
        scales: Scale factors for dequantization (float32)
                Can be either 2D (H, W) or 3D (H, W, 128)

    Returns:
        Dequantized embedding as float32 array, shape (H, W, 128)

    Examples:
        >>> import numpy as np
        >>> from geotessera import dequantize_embedding
        >>> quantized = np.load('grid_0.15_52.05.npy')  # int8
        >>> scales = np.load('grid_0.15_52.05_scales.npy')  # float32
        >>> embedding = dequantize_embedding(quantized, scales)
        >>> embedding.shape
        (1000, 1000, 128)
        >>> embedding.dtype
        dtype('float32')
    """
    # Handle both 2D scales (H, W) and 3D scales (H, W, 128)
    if scales.ndim == 2 and quantized_embedding.ndim == 3:
        # Broadcast 2D scales to match 3D embedding shape
        scales = scales[..., np.newaxis]  # Add channel dimension

    return quantized_embedding.astype(np.float32) * scales


class GeoTessera:
    """Library for downloading Tessera tiles and exporting GeoTIFFs.

    Core functionality:
    - Download tiles to local embeddings_dir
    - Sample embeddings at point locations from local tiles
    - Export individual tiles as GeoTIFF files with correct metadata
    - Manage registry and data access

    Typical workflows:

    Simple (auto-download mode):
        1. Initialize with embeddings_dir (defaults to current directory)
        2. Call sample_embeddings_at_points() - tiles are downloaded automatically

    Manual/offline mode:
        1. Initialize with embeddings_dir
        2. Use download_tiles_for_points() to download required tiles
        3. Use sample_embeddings_at_points(auto_download=False) for offline operation
        4. Use check_tiles_present() to verify which tiles are available

    Attributes:
        registry: geotessera.registry.Registry instance for data discovery and access
        embeddings_dir: Directory where tiles are stored locally
    """

    def __init__(
        self,
        dataset_version: str = "v1",
        dataset_variant: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        registry_url: Optional[str] = None,
        registry_path: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize GeoTessera with Parquet registry.

        Args:
            dataset_version: Tessera dataset version. Accepts ``"v1"`` /
                ``"1.0"`` or ``"v1.1"`` / ``"1.1"`` (the repository uses
                ``v1/`` for the 1.0 series).
            dataset_variant: Variant of the embeddings to load. Defaults to
                the version's published variant (``"vultr"`` for v1,
                ``"cambridge"`` for v1.1). Variants are produced by different
                model runs and are selected by filtering the manifest.
            cache_dir: Directory for caching registry files only (not embedding data)
            embeddings_dir: Directory containing pre-downloaded embedding tiles.
                Defaults to current working directory if not specified.

                To populate embeddings_dir, use one of these approaches:

                1. Download specific tiles using the CLI:
                   $ geotessera download --lat 52.05 --lon 0.15 --year 2024 --output ./embeddings

                2. Download tiles for a region using the CLI:
                   $ geotessera download --region region.geojson --year 2024 --output ./embeddings

                Expected directory structure:
                    embeddings_dir/
                    ├── global_0.1_degree_representation/
                    │   └── 2024/
                    │       ├── grid_0.15_52.05.npy
                    │       ├── grid_0.15_52.05_scales.npy
                    │       └── ...
                    └── global_0.1_degree_tiff_all/
                        ├── grid_0.15_52.05.tiff
                        └── ...
            registry_url: URL to download Parquet registry from (default: remote)
            registry_path: Local path to existing Parquet registry file
            registry_dir: Directory containing registry.parquet and landmasks.parquet files
        """
        self.dataset_version = dataset_version

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Set embeddings_dir to current working directory if not specified
        if embeddings_dir is None:
            self.embeddings_dir = Path.cwd()
        else:
            self.embeddings_dir = Path(embeddings_dir)

        self.registry = Registry(
            version=dataset_version,
            variant=dataset_variant,
            cache_dir=cache_dir,
            embeddings_dir=embeddings_dir,
            registry_url=registry_url,
            registry_path=registry_path,
            registry_dir=registry_dir,
            logger=self.logger,
        )
        # The resolved variant (per-version default applied when the caller
        # passed None).
        self.dataset_variant = self.registry.variant

    @property
    def version(self) -> str:
        """Get the GeoTessera library version."""
        return __version__

    @property
    def embeddings_subdir(self) -> str:
        """Variant-aware embeddings subdirectory name.

        Equals ``global_0.1_degree_representation`` for the default ``vultr``
        variant and ``global_0.1_degree_representation.<variant>`` otherwise.
        Recorded in the ``tessera_metadata.json`` sidecar.
        """
        return self.registry.embeddings_subdir

    def embeddings_count(
        self, bbox: Tuple[float, float, float, float], year: int = 2024
    ) -> int:
        """Get total number of embedding tiles within a bounding box.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to consider

        Returns:
            Total number of tiles in the bounding box
        """
        tiles = self.registry.load_blocks_for_region(bbox, year)
        return len(tiles)

    def export_coverage_map(
        self,
        output_file: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict:
        """Generate global coverage map showing which tiles have embeddings for which years.

        This method loads all registry data and creates a coverage map that can be used
        for visualization. It includes information about:
        - Which tiles have embedding data
        - Which years each tile covers
        - Landmask information (whether a tile location has land)

        When output_file is provided, writes a compact split format:
        - coverage.json: metadata + years only (small)
        - coverage[_dataset_id]_YYYY.json: array of tile coordinate strings per year

        The globe.html viewer uses the coverage texture PNG for land/ocean detection
        (no need to store no_coverage or landmasks in JSON).

        Args:
            output_file: Optional path to write JSON coverage data. If None, returns dict only.
            dataset_id: Optional namespace for the per-year files (e.g. ``"v1_vultr"``)
                so multiple (version, variant) datasets can coexist in one
                output directory. When set, per-year files become
                ``coverage_{dataset_id}_{YYYY}.json``.

        Returns:
            Dictionary with full coverage information (used in-memory for texture generation).
        """
        self.logger.info("Loading all registry data for global coverage analysis...")

        # Load all available embedding blocks
        bbox = (-180, -90, 180, 90)  # Global coverage
        available_years = self.registry.get_available_years()

        # Collect tiles per year (for per-year files) and per location (for texture)
        tiles_by_location = {}
        tiles_by_year = {year: [] for year in available_years}

        for year in available_years:
            self.logger.info(f"Loading embeddings for {year}...")
            tiles = self.registry.load_blocks_for_region(bbox, year)

            for tile in tiles:
                # Handle both 2-tuple (lon, lat) and 3-tuple (year, lon, lat) formats
                if len(tile) == 3:
                    _, lon, lat = tile
                else:
                    lon, lat = tile

                key = f"{lon:.2f},{lat:.2f}"
                if key not in tiles_by_location:
                    tiles_by_location[key] = []
                tiles_by_location[key].append(year)
                tiles_by_year[year].append(key)

        # Get landmask information
        self.logger.info("Loading landmask data...")
        available_landmasks = self.registry.available_landmasks
        landmask_keys = [f"{lon:.2f},{lat:.2f}" for lon, lat in available_landmasks]
        landmask_set = set(landmask_keys)

        # Categorize tiles
        tiles_with_data = set(tiles_by_location.keys())

        # Land tiles without coverage: in landmask but not in tiles
        no_coverage_tiles = sorted(landmask_set - tiles_with_data)

        # Full in-memory coverage map (used for texture generation)
        coverage_map = {
            "tiles": tiles_by_location,
            "no_coverage": no_coverage_tiles,
            "years": available_years,
            "metadata": {
                "total_tiles": len(tiles_by_location),
                "total_landmasks": len(landmask_keys),
                "total_no_coverage": len(no_coverage_tiles),
                "version": self.dataset_version,
            },
        }

        # Write split files if requested
        if output_file:
            output_path = Path(output_file)
            output_dir = output_path.parent
            year_prefix = f"coverage_{dataset_id}_" if dataset_id else "coverage_"

            # Main coverage.json: just metadata + years (compact, no indent)
            main_data = {
                "years": available_years,
                "metadata": coverage_map["metadata"],
                "dataset_id": dataset_id,
                "year_file_prefix": year_prefix,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(main_data, f, separators=(",", ":"))
            self.logger.info(f"Coverage metadata written to {output_file}")

            # Per-year files: coverage[_{dataset_id}]_{YYYY}.json
            for year in available_years:
                year_file = output_dir / f"{year_prefix}{year}.json"
                with open(year_file, "w", encoding="utf-8") as f:
                    json.dump(sorted(tiles_by_year[year]), f, separators=(",", ":"))
                self.logger.info(
                    f"  {year}: {len(tiles_by_year[year]):,} tiles -> {year_file.name}"
                )

        return coverage_map

    def generate_coverage_texture(
        self,
        coverage_data: Dict,
        output_file: Optional[str] = None,
        tint_color: Optional[Tuple[int, int, int]] = None,
    ) -> str:
        """Generate coverage texture image for globe visualization.

        Creates a 3600x1800 pixel equirectangular projection texture where each pixel
        represents a 0.1-degree tile, colored by coverage status.

        Args:
            coverage_data: Coverage data dictionary from export_coverage_map()
            output_file: Optional path to save PNG texture. If None, saves as 'coverage_texture.png'
            tint_color: Optional ``(r, g, b)`` 0-255 tuple. When provided, every
                tile with coverage is drawn in this colour, shaded by the
                fraction of available years that tile actually has data for —
                pale tint for sparse temporal coverage, saturated for full.
                Used to give each dataset a distinct hue family in
                multi-dataset mode while still letting the viewer see *how
                much* of the year range each tile contains at a glance.

        Returns:
            Path to the generated texture file
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError(
                "PIL/Pillow required for texture generation: pip install Pillow"
            )

        # Constants matching JavaScript
        TILE_SIZE = 0.1
        TILE_OFFSET = 0.05

        # Calculate canvas size (one pixel per tile)
        width = int(360 / TILE_SIZE)  # 3600
        height = int(180 / TILE_SIZE)  # 1800

        self.logger.info(f"Generating coverage texture ({width}x{height} pixels)...")

        # Create RGBA image
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Get metadata for coloring logic
        tiles_dict = coverage_data["tiles"]
        no_coverage_set = set(coverage_data.get("no_coverage", []))
        all_years = coverage_data["years"]
        max_years = len(all_years)
        latest_year = max(all_years) if all_years else 0

        tile_count = 0

        # Iterate through grid (same as JavaScript)
        lon = -180 + TILE_OFFSET
        while lon < 180:
            lat = -90 + TILE_OFFSET
            while lat < 90:
                # Generate tile key
                key = f"{lon:.2f},{lat:.2f}"

                # Determine color based on coverage. Single-tint mode varies
                # the shade of one hue per dataset, so within a dataset the
                # viewer can see *temporal* coverage at a glance (pale =
                # sparse years, saturated = all years), and across datasets
                # the hue distinguishes which version/variant.
                if tint_color is not None:
                    tile_years = tiles_dict.get(key)
                    if tile_years:
                        n = len(tile_years)
                        # Interpolate exactly between n=1 -> 50% tint and
                        # n=max_years -> full tint. (max_years=1 collapses
                        # to full tint for everything.)
                        if max_years <= 1:
                            frac = 1.0
                        else:
                            frac = 0.5 + 0.5 * (n - 1) / (max_years - 1)
                        r = int(tint_color[0] * frac + 255 * (1 - frac))
                        g = int(tint_color[1] * frac + 255 * (1 - frac))
                        b = int(tint_color[2] * frac + 255 * (1 - frac))
                        # Alpha also scales lightly with coverage so sparse
                        # tiles fade more into the globe; keeps a clear visual
                        # hierarchy when two datasets overlap.
                        alpha = (
                            int(140 + 60 * (n / max_years)) if max_years > 0 else 200
                        )
                        color = (r, g, b, alpha)
                    else:
                        color = (0, 0, 0, 0)
                else:
                    color = self._get_tile_color(
                        key,
                        tiles_dict,
                        no_coverage_set,
                        all_years,
                        max_years,
                        latest_year,
                    )

                # Convert lat/lon to pixel coordinates (equirectangular projection)
                min_lon = lon - TILE_OFFSET
                max_lon = lon + TILE_OFFSET
                min_lat = lat - TILE_OFFSET
                max_lat = lat + TILE_OFFSET

                x1 = int(((min_lon + 180) / 360) * width)
                x2 = int(((max_lon + 180) / 360) * width)
                y1 = int(((90 - max_lat) / 180) * height)
                y2 = int(((90 - min_lat) / 180) * height)

                # Draw rectangle if not transparent
                if color[3] > 0:  # If alpha > 0
                    draw.rectangle([x1, y1, x2, y2], fill=color)

                tile_count += 1
                lat += TILE_SIZE
            lon += TILE_SIZE

        # Save texture
        if output_file is None:
            output_file = "coverage_texture.png"

        img.save(output_file, "PNG")
        self.logger.info(
            f"Generated texture with {tile_count} tiles, saved to {output_file}"
        )

        return output_file

    def _get_tile_color(
        self,
        key: str,
        tiles_dict: Dict,
        no_coverage_set: set,
        all_years: list,
        max_years: int,
        latest_year: int,
    ) -> tuple:
        """Get RGBA color for a tile based on coverage (matches JavaScript logic)."""

        # Check if tile has coverage data
        if key in tiles_dict:
            years = tiles_dict[key]
            num_years = len(years)
            tile_latest_year = max(years)

            if num_years == 1:
                # Single year coverage
                if tile_latest_year == latest_year:
                    return (255, 200, 0, 255)  # Yellow - latest year only
                else:
                    return (200, 100, 0, 255)  # Orange - older single year
            elif num_years == max_years:
                # Complete coverage - all years
                return (0, 200, 0, 255)  # Green - full coverage
            else:
                # Partial multi-year coverage - gradient from blue to cyan
                ratio = num_years / max_years
                blue = int(100 + ratio * 155)
                green = int(ratio * 200)
                return (0, green, blue, 255)

        # Check if land with no coverage
        if key in no_coverage_set:
            return (100, 100, 100, 76)  # Gray, semi-transparent (0.3 * 255 ≈ 76)

        # Ocean - transparent
        return (0, 0, 0, 0)

    # returns a generator
    def fetch_embeddings(
        self,
        tiles_to_fetch: Iterable[Tuple[int, float, float]],
        progress_callback: Optional[callable] = None,
    ) -> Generator[Tuple[int, float, float, np.ndarray, object, object], None, None]:
        """Lazily fetches all requested tiles with CRS information.
        Use as a generator to process tiles one at a time in a memory-efficient manner.
        The list of tiles to fetch can be obtained by registry.load_blocks_for_region().

        Args:
            tiles_to_fetch: List of tiles to fetch as (year, tile_lon, tile_lat) tuples
            progress_callback: Optional callback function(current, total) for progress tracking

        Returns:
            Generator of (year, tile_lon, tile_lat, embedding_array, crs, transform) tuples where:
            - year: Tile year
            - tile_lon: Tile center longitude
            - tile_lat: Tile center latitude
            - embedding_array: shape (H, W, 128) with dequantized values
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        # Download each tile with progress tracking
        if progress_callback:
            total_tiles = len(tiles_to_fetch)

        for i, (year, tile_lon, tile_lat) in enumerate(tiles_to_fetch):
            try:
                # Create a sub-progress callback for this tile's downloads
                def tile_progress_callback(
                    current: int, total: int, status: str = None
                ):
                    if progress_callback:
                        # Map individual file progress to overall tile progress
                        tile_progress = (
                            i * 100 + (current / max(total, 1)) * 100
                        ) / total_tiles
                        tile_status = (
                            f"Tile {i + 1}/{total_tiles}: {status}"
                            if status
                            else f"Fetching tile {i + 1}/{total_tiles}"
                        )
                        progress_callback(int(tile_progress), 100, tile_status)

                embedding, crs, transform = self.fetch_embedding(
                    tile_lon, tile_lat, year, tile_progress_callback
                )

                yield year, tile_lon, tile_lat, embedding, crs, transform

                # Update progress for completed tile
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Completed tile {i + 1}/{total_tiles}",
                    )

            except Exception as e:
                self.logger.warning(
                    f"Failed to download tile ({tile_lon:.2f}, {tile_lat:.2f}): {e}"
                )
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Failed tile {i + 1}/{total_tiles}",
                    )
                continue

    def _ensure_tiles_available(
        self,
        required_coords: set,
        year: int,
        auto_download: bool,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        progress_callback: Optional[callable] = None,
        progress_offset: int = 0,
        progress_total: Optional[int] = None,
    ) -> dict:
        """Ensure required tiles are available locally, downloading if needed.

        This helper method implements the common pattern of:
        1. Discovering local tiles and building a map for the specified year
        2. Checking which tiles from the required set are missing
        3. Downloading missing tiles if auto_download=True
        4. Raising an error if auto_download=False and tiles are missing

        Args:
            required_coords: Set of (lon, lat) tuples for required tiles
            year: Year of embeddings
            auto_download: Whether to download missing tiles automatically
            bbox: Optional bounding box for error messages (min_lon, min_lat, max_lon, max_lat)
            progress_callback: Optional callback(current, total, status) for progress tracking
            progress_offset: Offset to add to progress current value
            progress_total: Total to use in progress callbacks (defaults to len(required_coords))

        Returns:
            Dictionary mapping (lon, lat) -> Tile object for the requested year

        Raises:
            FileNotFoundError: If auto_download=False and required tiles are missing
            UnwritableDestination: If tiles must be downloaded but
                embeddings_dir cannot be written
        """
        from geotessera.tiles import tiles_for_coords

        # Local tiles carry no dataset identity in their paths.
        self.registry.validate_embeddings_dir()

        # Resolve only the coordinates asked for: scanning embeddings_dir
        # costs a rasterio open per tile on disk, however few are wanted.
        local_tile_map = tiles_for_coords(self.embeddings_dir, required_coords, year)

        # Find missing tiles
        missing_tiles = [
            coord for coord in required_coords if coord not in local_tile_map
        ]

        # Handle missing tiles
        if missing_tiles:
            if auto_download:
                self.logger.info(
                    f"Downloading {len(missing_tiles)} missing tiles to {self.embeddings_dir}"
                )

                # Download each missing tile
                for idx, (tile_lon, tile_lat) in enumerate(missing_tiles):
                    grid_name = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"

                    if progress_callback:
                        current = progress_offset + idx
                        total = progress_total or len(required_coords)
                        progress_callback(
                            current,
                            total,
                            f"Downloading {grid_name} ({idx + 1}/{len(missing_tiles)})",
                        )

                    try:
                        success = self.download_tile(tile_lon, tile_lat, year)
                    except UnwritableDestination as e:
                        # Every remaining tile would fail the same way.
                        raise UnwritableDestination(
                            e.errno,
                            f"{e.strerror}. {len(missing_tiles)} tiles are "
                            f"missing from {self.embeddings_dir}; populate it "
                            f"with 'geotessera download --year {year}' before "
                            f"sampling.",
                        ) from e
                    if not success:
                        self.logger.warning(f"Failed to download {grid_name}")

                # Pick up whatever downloaded; a tile that failed stays absent.
                local_tile_map.update(
                    tiles_for_coords(self.embeddings_dir, missing_tiles, year)
                )
            else:
                # In offline mode, raise error with helpful message
                missing_names = [
                    f"grid_{lon:.2f}_{lat:.2f}" for lon, lat in missing_tiles
                ]

                # Build helpful error message
                error_msg = (
                    f"{len(missing_tiles)} required tiles not found in {self.embeddings_dir}: "
                    f"{missing_names[:5]}"
                )
                if len(missing_names) > 5:
                    error_msg += f"... (and {len(missing_names) - 5} more)"

                error_msg += "\n\nSet auto_download=True to download automatically, or download tiles manually using:\n"

                if bbox:
                    min_lon, min_lat, max_lon, max_lat = bbox
                    error_msg += (
                        f"geotessera download --bbox '{min_lon},{min_lat},{max_lon},{max_lat}' "
                        f"--year {year} --output {self.embeddings_dir}"
                    )
                else:
                    error_msg += f"geotessera download --year {year} --output {self.embeddings_dir}"

                raise FileNotFoundError(error_msg)

        return local_tile_map

    def fetch_mosaic_for_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2024,
        target_crs: str = "EPSG:4326",
        auto_download: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, object, str]:
        """Fetch and merge all embedding tiles within a bounding box into a single mosaic.

        This method is optimized for dense raster operations like classification, where you
        need embedding values for every pixel in a region. It:
        1. Finds all tiles intersecting the bounding box
        2. Checks embeddings_dir for existing tiles
        3. Downloads missing tiles if auto_download=True
        4. Reprojects all tiles to a common CRS
        5. Merges them into a single seamless mosaic array
        6. Returns the mosaic with its geospatial transform

        For sparse point sampling (< 1000s of points), use sample_embeddings_at_points() instead.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
            year: Year of embeddings to fetch
            target_crs: Target CRS for the output mosaic (default: "EPSG:4326")
            auto_download: If True (default), automatically download missing tiles to embeddings_dir.
                If False, operate in offline mode using only local tiles from embeddings_dir.
                Set to False for guaranteed offline operation with no network requests.
            progress_callback: Optional callback(current, total, status) for progress tracking

        Returns:
            Tuple of (mosaic_array, mosaic_transform, crs):
            - mosaic_array: numpy array of shape (height, width, 128) with embedding values
            - mosaic_transform: rasterio Affine transform for the mosaic
            - crs: Coordinate reference system string (same as target_crs)

        Raises:
            ValueError: If no tiles found for the specified region/year
            FileNotFoundError: If auto_download=False and required tiles are missing from embeddings_dir
            ImportError: If rasterio is not installed

        Examples:
            >>> from geotessera import GeoTessera
            >>> gt = GeoTessera(embeddings_dir="./embeddings")
            >>>
            >>> # Auto-download mode (default): downloads missing tiles automatically
            >>> bbox = (0.0, 52.0, 0.2, 52.2)  # (min_lon, min_lat, max_lon, max_lat)
            >>> mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=2024)
            >>> print(f"Mosaic shape: {mosaic.shape}")  # e.g., (2000, 2000, 128)
            >>>
            >>> # Offline mode: use only pre-downloaded tiles
            >>> mosaic, transform, crs = gt.fetch_mosaic_for_region(
            ...     bbox, year=2024, auto_download=False
            ... )
            >>>
            >>> # Use for classification
            >>> from sklearn.neighbors import KNeighborsClassifier
            >>> # Train on labeled pixels...
            >>> # Then classify all pixels in the mosaic
            >>> pixels = mosaic.reshape(-1, 128)
            >>> predictions = classifier.predict(pixels)
            >>> classification_map = predictions.reshape(mosaic.shape[:2])
        """
        try:
            from rasterio.merge import merge
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            from rasterio.io import MemoryFile
            from rasterio.transform import array_bounds
        except ImportError:
            raise ImportError(
                "rasterio required for mosaic creation. Install with: pip install rasterio"
            )

        # Validate bbox
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (-180 <= min_lon <= max_lon <= 180):
            raise ValueError(f"Invalid longitude range: {min_lon} to {max_lon}")
        if not (-90 <= min_lat <= max_lat <= 90):
            raise ValueError(f"Invalid latitude range: {min_lat} to {max_lat}")

        # Find tiles in region
        self.logger.info(f"Finding tiles for region: {bbox}, year: {year}")
        tiles_needed = self.registry.load_blocks_for_region(bbox, year)

        if not tiles_needed:
            raise ValueError(
                f"No embedding tiles found for bbox {bbox} in year {year}. "
                f"Check data availability with: geotessera info --bbox '{min_lon},{min_lat},{max_lon},{max_lat}'"
            )

        self.logger.info(f"Found {len(tiles_needed)} tiles needed for mosaic")

        # Extract unique tile coordinates
        tiles_needed_coords = {(lon, lat) for (y, lon, lat) in tiles_needed}

        # Ensure all required tiles are available (download if needed)
        local_tile_map = self._ensure_tiles_available(
            required_coords=tiles_needed_coords,
            year=year,
            auto_download=auto_download,
            bbox=bbox,
            progress_callback=progress_callback,
            progress_offset=0,
            progress_total=len(tiles_needed_coords) + len(tiles_needed_coords) * 2 + 1,
        )

        # Calculate number of missing tiles for progress tracking
        # (already downloaded by _ensure_tiles_available if auto_download=True)
        num_missing = sum(
            1
            for coord in tiles_needed_coords
            if coord not in local_tile_map or not local_tile_map[coord].is_available()
        )

        # Track progress for loading + reprojecting + merging
        total_steps = (
            len(tiles_needed_coords) * 2 + 1
        )  # load+reproject per tile, then merge
        current_step = num_missing if auto_download else 0

        def update_progress(status: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, status)

        # Load all tiles from embeddings_dir and reproject to target CRS
        reprojected_memfiles = []

        for tile_lon, tile_lat in tiles_needed_coords:
            # Get Tile object from local storage
            tile = local_tile_map.get((tile_lon, tile_lat))
            if tile is None or not tile.is_available():
                self.logger.warning(
                    f"Tile ({tile_lon:.2f}, {tile_lat:.2f}) not available after download, skipping"
                )
                continue

            update_progress(f"Loading tile ({tile_lon:.2f}, {tile_lat:.2f})")

            try:
                # Load embedding from Tile (handles both NPY and GeoTIFF formats)
                embedding = tile.load_embedding()
                src_crs = tile.crs
                src_transform = tile.transform

                # Get source dimensions and bounds
                src_height, src_width = embedding.shape[:2]
                src_bounds = array_bounds(src_height, src_width, src_transform)

                # Calculate destination transform and dimensions
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, target_crs, src_width, src_height, *src_bounds
                )

                # Ensure dimensions are valid integers
                if dst_width is None or dst_height is None:
                    raise ValueError(
                        f"Failed to calculate dimensions for tile ({tile_lon}, {tile_lat})"
                    )
                dst_width = int(dst_width)
                dst_height = int(dst_height)

                # Create empty array for reprojected data
                # rasterio expects (channels, height, width) format
                reprojected_embedding = np.empty(
                    (embedding.shape[2], dst_height, dst_width), dtype=embedding.dtype
                )

                # Reproject each channel
                for band_idx in range(embedding.shape[2]):
                    reproject(
                        source=embedding[:, :, band_idx],
                        destination=reprojected_embedding[band_idx],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )

                # Store in memory file for merging
                memfile = MemoryFile()
                with memfile.open(
                    driver="GTiff",
                    height=dst_height,
                    width=dst_width,
                    count=embedding.shape[2],
                    dtype=embedding.dtype,
                    crs=target_crs,
                    transform=dst_transform,
                ) as dataset:
                    dataset.write(reprojected_embedding)

                reprojected_memfiles.append(memfile)
                update_progress(f"Reprojected tile ({tile_lon:.2f}, {tile_lat:.2f})")

            except Exception as e:
                self.logger.warning(
                    f"Failed to process tile ({tile_lon:.2f}, {tile_lat:.2f}): {e}"
                )
                continue

        if not reprojected_memfiles:
            raise RuntimeError("No tiles successfully reprojected")

        # Merge all reprojected tiles
        self.logger.info(f"Merging {len(reprojected_memfiles)} reprojected tiles...")
        datasets = []
        try:
            for memfile in reprojected_memfiles:
                datasets.append(memfile.open())

            merged_array, mosaic_transform = merge(datasets)

            # Convert from (channels, height, width) to (height, width, channels)
            mosaic_array = np.transpose(merged_array, (1, 2, 0))

            update_progress("Mosaic merge complete")

        finally:
            # Clean up
            for dataset in datasets:
                dataset.close()
            for memfile in reprojected_memfiles:
                memfile.close()

        self.logger.info(
            f"Mosaic created: shape={mosaic_array.shape}, crs={target_crs}"
        )

        return mosaic_array, mosaic_transform, target_crs

    def sample_embeddings_at_points(
        self,
        points: Union[List[Tuple[float, float]], Dict, "gpd.GeoDataFrame"],
        year: int = 2024,
        include_metadata: bool = False,
        auto_download: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]:
        """Sample embedding values at specified point locations from local tiles.

        This method efficiently extracts embedding values at arbitrary lon/lat
        coordinates by:
        1. Grouping points by which tile they fall into
        2. Optionally downloading missing tiles if auto_download=True
        3. Loading tiles from self.embeddings_dir
        4. Extracting all point values from each tile
        5. Returning results in original point order

        Args:
            points: Point coordinates as:
                - List of (lon, lat) tuples
                - GeoJSON FeatureCollection dict
                - GeoPandas GeoDataFrame with Point geometries
            year: Year of embeddings to sample
            include_metadata: If True, also return metadata (tile info, pixel coords)
            auto_download: If True (default), automatically download missing tiles.
                If False, operate in offline mode and raise error for missing tiles.
                Set to False for guaranteed offline operation with no network requests.
            progress_callback: Optional callback(current, total, status)

        Returns:
            If include_metadata=False:
                numpy array of shape (N, 128) with embedding values
                (NaN for points outside coverage)
            If include_metadata=True:
                (embeddings_array, metadata_list) where metadata contains:
                - tile_lon, tile_lat: Which tile the point came from
                - pixel_row, pixel_col: Pixel coordinates within tile
                - crs: Coordinate reference system of tile

        Raises:
            FileNotFoundError: If auto_download=False and required tiles are missing

        Examples:
            >>> # Auto-download mode (default): downloads tiles as needed
            >>> gt = GeoTessera(embeddings_dir="./embeddings")
            >>> points = [(0.15, 52.05), (0.25, 52.15)]
            >>> embeddings = gt.sample_embeddings_at_points(points, year=2024)
            >>> embeddings.shape
            (2, 128)

            >>> # Offline mode: no downloads, fails if tiles missing
            >>> embeddings = gt.sample_embeddings_at_points(
            ...     points, year=2024, auto_download=False
            ... )

            >>> # Manual download workflow (equivalent to auto_download=True)
            >>> gt.download_tiles_for_points(points, year=2024)
            >>> embeddings = gt.sample_embeddings_at_points(
            ...     points, year=2024, auto_download=False
            ... )

            >>> # With metadata
            >>> embeddings, metadata = gt.sample_embeddings_at_points(
            ...     points, year=2024, include_metadata=True
            ... )
            >>> metadata[0]['tile_lon'], metadata[0]['tile_lat']
            (0.15, 52.05)
        """
        try:
            from pyproj import Transformer
            import rasterio.transform
        except ImportError:
            raise ImportError(
                "pyproj and rasterio required for point sampling: "
                "pip install pyproj rasterio"
            )

        # Parse points to standardized format: list of (lon, lat) tuples
        parsed_points = self._parse_points_input(points)
        n_points = len(parsed_points)

        if n_points == 0:
            if include_metadata:
                return np.empty((0, 128), dtype=np.float32), []
            return np.empty((0, 128), dtype=np.float32)

        # Group points by which tile they belong to
        points_by_tile = self._group_points_by_tile(parsed_points, year)

        # Ensure all required tiles are available (download if needed)
        required_tile_coords = set(points_by_tile.keys())
        tile_map = self._ensure_tiles_available(
            required_coords=required_tile_coords,
            year=year,
            auto_download=auto_download,
            bbox=None,  # No bbox available for point-based queries
            progress_callback=progress_callback,
            progress_offset=0,
            progress_total=len(points_by_tile),
        )

        # Initialize result arrays
        result_embeddings = np.full((n_points, 128), np.nan, dtype=np.float32)
        result_metadata = [None] * n_points if include_metadata else None

        if progress_callback:
            progress_callback(
                0,
                len(points_by_tile),
                f"Found {len(tile_map)} local tiles for year {year}",
            )

        # Process each tile
        total_tiles = len(points_by_tile)
        for tile_idx, ((tile_lon, tile_lat), point_indices) in enumerate(
            points_by_tile.items()
        ):
            if progress_callback:
                progress_callback(
                    tile_idx,
                    total_tiles,
                    f"Processing tile {tile_idx + 1}/{total_tiles}: ({tile_lon:.2f}, {tile_lat:.2f})",
                )

            try:
                # Use Tile abstraction for local files
                tile = tile_map.get((tile_lon, tile_lat))
                if tile is None:
                    error_msg = f"Tile ({tile_lon:.2f}, {tile_lat:.2f}) not found in {self.embeddings_dir}. "
                    if auto_download:
                        error_msg += (
                            "Download may have failed. Check network connectivity."
                        )
                    else:
                        error_msg += (
                            "Set auto_download=True to download automatically, or use "
                            "download_tiles_for_points() to download tiles manually."
                        )
                    raise FileNotFoundError(error_msg)

                # Load embedding and metadata from Tile
                embedding = tile.load_embedding()
                crs = tile.crs
                transform = tile.transform

                # Create coordinate transformer from WGS84 to tile's CRS
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

                # Extract embedding values for all points in this tile
                for original_idx in point_indices:
                    lon, lat = parsed_points[original_idx]

                    # Transform from WGS84 to tile's projected coordinates
                    x, y = transformer.transform(lon, lat)

                    # Convert projected coordinates to pixel row/col
                    row, col = rasterio.transform.rowcol(transform, x, y)

                    # Check if pixel is within bounds
                    height, width = embedding.shape[:2]
                    if 0 <= row < height and 0 <= col < width:
                        # Extract embedding value
                        result_embeddings[original_idx] = embedding[row, col]

                        # Store metadata if requested
                        if include_metadata:
                            result_metadata[original_idx] = {
                                "tile_lon": tile_lon,
                                "tile_lat": tile_lat,
                                "pixel_row": row,
                                "pixel_col": col,
                                "crs": str(crs),
                            }
                    else:
                        # Point is outside tile bounds (shouldn't happen, but handle gracefully)
                        if include_metadata:
                            result_metadata[original_idx] = {
                                "tile_lon": tile_lon,
                                "tile_lat": tile_lat,
                                "pixel_row": None,
                                "pixel_col": None,
                                "crs": str(crs),
                                "error": "pixel_out_of_bounds",
                            }

            except Exception as e:
                # If tile fetch/load fails, leave those points as NaN
                self.logger.warning(
                    f"Failed to process tile ({tile_lon:.2f}, {tile_lat:.2f}): {e}"
                )
                if include_metadata:
                    for original_idx in point_indices:
                        result_metadata[original_idx] = {
                            "tile_lon": tile_lon,
                            "tile_lat": tile_lat,
                            "error": str(e),
                        }

        if progress_callback:
            progress_callback(total_tiles, total_tiles, "Sampling complete")

        if include_metadata:
            return result_embeddings, result_metadata
        return result_embeddings

    def _parse_points_input(
        self, points: Union[List[Tuple[float, float]], Dict, "gpd.GeoDataFrame"]
    ) -> List[Tuple[float, float]]:
        """Parse various point input formats to list of (lon, lat) tuples.

        Args:
            points: Points in various formats

        Returns:
            List of (lon, lat) tuples
        """
        # Handle GeoJSON FeatureCollection (a dict is also Iterable, so this must
        # be checked before the generic list/array fallback below).
        if isinstance(points, dict):
            if points.get("type") == "FeatureCollection":
                result = []
                for feature in points.get("features", []):
                    geom = feature.get("geometry", {})
                    if geom.get("type") == "Point":
                        coords = geom.get("coordinates", [])
                        if len(coords) >= 2:
                            result.append((coords[0], coords[1]))
                return result
            raise ValueError(
                "Dict input must be a GeoJSON FeatureCollection with Point geometries"
            )

        # Handle GeoDataFrame (also Iterable, so check before the generic fallback).
        if isinstance(points, gpd.GeoDataFrame):
            result = []
            for geom in points.geometry:
                if geom.geom_type == "Point":
                    result.append((geom.x, geom.y))
                else:
                    raise ValueError("GeoDataFrame must contain only Point geometries")
            return result

        # Handle a plain list/tuple/array of (lon, lat) pairs (most common case).
        if isinstance(points, Iterable):
            return list(points)

        raise ValueError(
            "points must be a list of (lon, lat) tuples, GeoJSON FeatureCollection, "
            "or GeoPandas GeoDataFrame"
        )

    def _group_points_by_tile(
        self, points: List[Tuple[float, float]], year: int
    ) -> Dict[Tuple[float, float], List[int]]:
        """Group points by which 0.1-degree tile they belong to.

        Args:
            points: List of (lon, lat) tuples
            year: Year of embeddings

        Returns:
            Dictionary mapping (tile_lon, tile_lat) -> list of point indices
        """
        # Group points by tile
        points_by_tile = {}
        for idx, (lon, lat) in enumerate(points):
            tile_lon, tile_lat = tile_from_world(lon, lat)
            tile_key = (tile_lon, tile_lat)
            if tile_key not in points_by_tile:
                points_by_tile[tile_key] = []
            points_by_tile[tile_key].append(idx)

        # Filter to only tiles that exist in the registry
        available_tiles = set(
            self.registry.load_blocks_for_region((-180, -90, 180, 90), year)
        )

        # Convert to set of (lon, lat) for faster lookup
        available_tile_coords = {(lon, lat) for (y, lon, lat) in available_tiles}

        # Filter out points that fall in tiles without data
        filtered_points_by_tile = {
            tile_key: indices
            for tile_key, indices in points_by_tile.items()
            if tile_key in available_tile_coords
        }

        # Warn about points outside coverage
        missing_tiles = set(points_by_tile.keys()) - set(filtered_points_by_tile.keys())
        if missing_tiles:
            n_missing = sum(len(points_by_tile[tile]) for tile in missing_tiles)
            self.logger.warning(
                f"{n_missing} points fall in tiles without coverage "
                f"(will be returned as NaN)"
            )

        return filtered_points_by_tile

    def fetch_embedding(
        self,
        lon: float,
        lat: float,
        year: int,
        progress_callback: Optional[callable] = None,
        refresh: bool = False,
    ) -> Tuple[np.ndarray, object, object]:
        """Fetch and dequantize a single embedding tile with CRS information.

        Coordinates are automatically snapped to valid tile centers. Tiles are
        0.1×0.1 degree squares centered at 0.05-degree offsets (e.g., 0.05, 0.15).
        For example, coordinates (-120, 36) will be snapped to (-119.95, 36.05).

        Args:
            lon: Longitude (will be snapped to nearest tile center)
            lat: Latitude (will be snapped to nearest tile center)
            year: Year of embeddings
            progress_callback: Optional callback for download progress
            refresh: If True, force re-download even if local files exist in embeddings_dir

        Returns:
            Tuple of (dequantized_embedding, crs, transform) where:
            - dequantized_embedding: array of shape (H, W, 128)
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        # Snap coordinates to valid tile centers
        tile_lon, tile_lat = tile_from_world(lon, lat)
        if tile_lon != lon or tile_lat != lat:
            self.logger.debug(
                f"Snapped coordinates ({lon}, {lat}) to tile center ({tile_lon}, {tile_lat})"
            )
        lon, lat = tile_lon, tile_lat

        # Fetch the files using coordinates
        embedding_file = self.registry.fetch(
            year=year,
            lon=lon,
            lat=lat,
            is_scales=False,
            progress_callback=progress_callback,
            refresh=refresh,
        )
        scales_file = self.registry.fetch(
            year=year,
            lon=lon,
            lat=lat,
            is_scales=True,
            progress_callback=progress_callback,
            refresh=refresh,
        )

        # Load quantized data and scales
        quantized_embedding = np.load(embedding_file)
        scales = np.load(scales_file)

        # Dequantize using the public function
        dequantized = dequantize_embedding(quantized_embedding, scales)

        # Get CRS and transform from landmask
        crs, transform = self._get_utm_projection_from_landmask(lon, lat, refresh)

        return dequantized, crs, transform

    def download_tile(
        self,
        lon: float,
        lat: float,
        year: int,
        progress_callback: Optional[callable] = None,
        refresh: bool = False,
    ) -> bool:
        """Download a single tile and save it to embeddings_dir.

        Coordinates are automatically snapped to valid tile centers. Tiles are
        0.1×0.1 degree squares centered at 0.05-degree offsets (e.g., 0.05, 0.15).

        Args:
            lon: Longitude (will be snapped to nearest tile center)
            lat: Latitude (will be snapped to nearest tile center)
            year: Year of embeddings
            progress_callback: Optional callback for download progress
            refresh: If True, force re-download even if tile exists locally

        Returns:
            True if download succeeded, False otherwise

        Examples:
            >>> gt = GeoTessera(embeddings_dir="./embeddings")
            >>> gt.download_tile(lon=0.15, lat=52.05, year=2024)
            True
            >>> # Arbitrary coordinates are snapped to tile center
            >>> gt.download_tile(lon=-120, lat=36, year=2024)  # snaps to (-119.95, 36.05)
            True
        """
        # Snap coordinates to valid tile centers
        tile_lon, tile_lat = tile_from_world(lon, lat)
        if tile_lon != lon or tile_lat != lat:
            self.logger.debug(
                f"Snapped coordinates ({lon}, {lat}) to tile center ({tile_lon}, {tile_lat})"
            )
        lon, lat = tile_lon, tile_lat

        try:
            # Download files to embeddings_dir, skipping tiles already cached locally
            # fetch() handles creating directory structure and saving to correct locations
            self.registry.fetch(
                year=year,
                lon=lon,
                lat=lat,
                is_scales=False,
                progress_callback=progress_callback,
                refresh=refresh,
            )

            self.registry.fetch(
                year=year,
                lon=lon,
                lat=lat,
                is_scales=True,
                progress_callback=progress_callback,
                refresh=refresh,
            )

            self.registry.fetch_landmask(lon=lon, lat=lat, refresh=refresh)

            return True

        except UnwritableDestination:
            # embeddings_dir itself is at fault, not this tile: let it out
            # so the caller can stop rather than retry every other tile.
            raise
        except Exception as e:
            self.logger.error(f"Failed to download tile ({lon:.2f}, {lat:.2f}): {e}")
            return False

    def download_tiles_for_points(
        self,
        points: Union[List[Tuple[float, float]], Dict, "gpd.GeoDataFrame"],
        year: int = 2024,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, bool]:
        """Download all tiles needed for the specified points to embeddings_dir.

        Args:
            points: Point coordinates as:
                - List of (lon, lat) tuples
                - GeoJSON FeatureCollection dict
                - GeoPandas GeoDataFrame with Point geometries
            year: Year of embeddings to download
            progress_callback: Optional callback(current, total, status)

        Returns:
            Dictionary mapping grid names to download success status

        Examples:
            >>> gt = GeoTessera(embeddings_dir="./embeddings")
            >>> points = [(0.15, 52.05), (0.25, 52.15)]
            >>> results = gt.download_tiles_for_points(points, year=2024)
            >>> results
            {'grid_0.15_52.05': True, 'grid_0.25_52.15': True}
        """
        # Parse points to standardized format
        parsed_points = self._parse_points_input(points)

        # Group points by tile
        points_by_tile = self._group_points_by_tile(parsed_points, year)

        # Download each tile
        results = {}
        total_tiles = len(points_by_tile)

        for idx, (tile_lon, tile_lat) in enumerate(points_by_tile.keys()):
            grid_name = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"

            if progress_callback:
                progress_callback(
                    idx,
                    total_tiles,
                    f"Downloading tile {idx + 1}/{total_tiles}: {grid_name}",
                )

            success = self.download_tile(tile_lon, tile_lat, year)
            results[grid_name] = success

            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_tiles,
                    f"Completed {idx + 1}/{total_tiles}: {grid_name}",
                )

        return results

    def check_tiles_present(
        self,
        points: Union[List[Tuple[float, float]], Dict, "gpd.GeoDataFrame"],
        year: int = 2024,
    ) -> Dict[str, bool]:
        """Check which tiles needed for the points are present in embeddings_dir.

        Args:
            points: Point coordinates as:
                - List of (lon, lat) tuples
                - GeoJSON FeatureCollection dict
                - GeoPandas GeoDataFrame with Point geometries
            year: Year of embeddings to check

        Returns:
            Dictionary mapping grid names to presence status

        Examples:
            >>> gt = GeoTessera(embeddings_dir="./embeddings")
            >>> points = [(0.15, 52.05), (0.25, 52.15)]
            >>> status = gt.check_tiles_present(points, year=2024)
            >>> status
            {'grid_0.15_52.05': True, 'grid_0.25_52.15': False}
        """
        from geotessera.tiles import discover_tiles

        # Parse points to standardized format
        parsed_points = self._parse_points_input(points)

        # Group points by tile
        points_by_tile = self._group_points_by_tile(parsed_points, year)

        # Discover local tiles
        tiles = discover_tiles(self.embeddings_dir)
        tile_map = {(t.lon, t.lat): t for t in tiles if t.year == year}

        # Check each required tile
        results = {}
        for tile_lon, tile_lat in points_by_tile.keys():
            grid_name = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"
            tile = tile_map.get((tile_lon, tile_lat))
            results[grid_name] = tile is not None and tile.is_available()

        return results

    def _reproject_geotiff_file(self, args):
        """Helper function to reproject a single GeoTIFF file.

        Args:
            args: Tuple containing (source_file, output_file, target_crs, source_resolution, compress)

        Returns:
            Tuple of (output_file, None) on success or (None, error_message) on failure
        """
        source_file, output_file, target_crs, source_resolution, compress = args

        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            with rasterio.open(source_file) as src:
                # Calculate transform and dimensions for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=source_resolution,
                )

                # Create reprojected file
                with rasterio.open(
                    output_file,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=target_crs,
                    transform=transform,
                    compress=compress,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dst:
                    # Reproject each band
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                        )

                    # Copy metadata and band descriptions
                    dst.update_tags(**src.tags())
                    for band_idx in range(1, src.count + 1):
                        if src.descriptions and band_idx <= len(src.descriptions):
                            band_desc = src.descriptions[band_idx - 1]
                            if band_desc:
                                dst.set_band_description(band_idx, band_desc)

            return output_file, None
        except Exception as e:
            return None, str(e)

    def _get_utm_projection_from_landmask(
        self, lon: float, lat: float, refresh: bool = False
    ):
        """Get UTM projection info from corresponding landmask tile.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
            refresh: If True, force re-download even if local file exists in embeddings_dir

        Returns:
            Tuple of (crs, transform) from landmask tile

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile cannot be fetched or read
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for UTM projection retrieval: pip install rasterio"
            )

        try:
            # Fetch landmask file using coordinates
            landmask_path = self.registry.fetch_landmask(
                lon=lon, lat=lat, refresh=refresh
            )

            # Extract CRS and transform
            with rasterio.open(landmask_path) as src:
                if src.crs is None:
                    raise RuntimeError(
                        f"Landmask tile {landmask_path} has no CRS information"
                    )
                if src.transform is None:
                    raise RuntimeError(
                        f"Landmask tile {landmask_path} has no transform information"
                    )
                return src.crs, src.transform

        except Exception as e:
            if isinstance(e, (ImportError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Failed to get UTM projection from landmask for ({lon:.2f}, {lat:.2f}): {e}"
            ) from e

    def _write_embedding_tiff(
        self, output_path, embedding, bands, crs, transform, year, lat, lon, compress
    ):
        """Write an embedding array to a GeoTIFF with standard Tessera metadata.

        Shared by :meth:`export_embedding_geotiff` (single tile) and
        :meth:`export_embedding_geotiffs` (region). ``bands`` selects a subset
        of the 128 channels, or ``None`` for all of them.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        if bands is not None:
            data = embedding[:, :, bands].copy()
            band_count = len(bands)
        else:
            data = embedding.copy()
            band_count = 128

        height, width = data.shape[:2]
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=band_count,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            for i in range(band_count):
                dst.write(data[:, :, i], i + 1)

            if bands is not None:
                for i, band_idx in enumerate(bands):
                    dst.set_band_description(i + 1, f"Tessera_Band_{band_idx}")
            else:
                for i in range(128):
                    dst.set_band_description(i + 1, f"Tessera_Band_{i}")

            # Record the resolved version path (v1, v1.1, …) and variant so the
            # dataset provenance is recoverable from the TIFF alone, even if
            # separated from the tessera_metadata.json sidecar.
            dst.update_tags(
                TESSERA_DATASET_VERSION=self.registry._version_norm,
                TESSERA_DATASET_VERSION_PATH=self.registry._version_path,
                TESSERA_DATASET_VARIANT=self.dataset_variant,
                TESSERA_YEAR=str(year),
                TESSERA_TILE_LAT=f"{lat:.2f}",
                TESSERA_TILE_LON=f"{lon:.2f}",
                TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                GEOTESSERA_VERSION=__version__,
            )

    def export_embedding_geotiff(
        self,
        lon: float,
        lat: float,
        output_path: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
    ) -> str:
        """Export a single embedding tile as a GeoTIFF file with native UTM projection.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
            output_path: Output path for GeoTIFF file
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF

        Returns:
            Path to created GeoTIFF file

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fetch single tile with CRS info
        embedding, crs, transform = self.fetch_embedding(lon, lat, year)

        self._write_embedding_tiff(
            output_path, embedding, bands, crs, transform, year, lat, lon, compress
        )
        return str(output_path)

    def export_embedding_geotiffs(
        self,
        tiles_to_fetch: Iterable[Tuple[int, float, float]],
        output_dir: Union[str, Path],
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """Export all embedding tiles in bounding box as individual GeoTIFF files with native UTM projections.
        The list of tiles to fetch can be obtained by registry.load_blocks_for_region().

        Args:
            tiles_to_fetch: List of tiles to fetch as (year, tile_lon, tile_lat) tuples
            output_dir: Directory to save GeoTIFF files
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF
            progress_callback: Optional callback function(current, total) for progress tracking

        Returns:
            List of paths to created GeoTIFF files

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tiles or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch tiles with progress tracking
        if progress_callback:
            progress_callback(0, 100, "Starting download...")

        tiles_to_fetch = list(tiles_to_fetch)
        total_tiles = len(tiles_to_fetch)

        if total_tiles == 0:
            self.logger.info("No tiles to export")
            if progress_callback:
                progress_callback(100, 100, "No tiles to export")
            return []

        tiles = self.fetch_embeddings(tiles_to_fetch)

        created_files = []

        # Sequential fetching of tile data and writing GeoTIFFs
        for i, (year, tile_lon, tile_lat, embedding, crs, transform) in enumerate(
            tiles
        ):
            # Use centralized path construction from registry
            geotiff_rel_path = tile_to_geotiff_path(tile_lon, tile_lat, year)
            output_path = output_dir / EMBEDDINGS_DIR_NAME / geotiff_rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._write_embedding_tiff(
                output_path,
                embedding,
                bands,
                crs,
                transform,
                year,
                tile_lat,
                tile_lon,
                compress,
            )

            created_files.append(str(output_path))

            # Update progress for GeoTIFF export phase
            if progress_callback:
                progress_callback(
                    i + 1,
                    total_tiles,
                    f"Exported {output_path.name} ({i + 1}/{total_tiles})",
                )

        if progress_callback:
            progress_callback(
                total_tiles,
                total_tiles,
                f"Completed! Exported {len(created_files)} GeoTIFF file(s)",
            )

        self.logger.info(
            f"Exported {len(created_files)} GeoTIFF file(s) to {output_dir}"
        )
        return created_files

    def merge_geotiffs_to_mosaic(
        self,
        geotiff_paths: List[str],
        output_path: Union[str, Path],
        target_crs: str = "EPSG:3857",
        compress: str = "lzw",
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Merge a list of GeoTIFF files into a single mosaic in the target CRS.

        Args:
            geotiff_paths: List of paths to GeoTIFF files to merge
            output_path: Path for output mosaic GeoTIFF
            target_crs: Target CRS for the mosaic (default: Web Mercator EPSG:3857)
            compress: Compression method for output GeoTIFF
            progress_callback: Optional callback function(current, total, status) for progress tracking

        Returns:
            Path to created mosaic file

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If merge fails
        """
        try:
            import rasterio
            from rasterio.merge import merge
            import tempfile
            import os
        except ImportError:
            raise ImportError(
                "rasterio required for mosaic creation: pip install rasterio"
            )

        if not geotiff_paths:
            raise RuntimeError("No GeoTIFF files provided")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine source resolution from first file
        with rasterio.open(geotiff_paths[0]) as first_src:
            source_resolution = min(
                abs(first_src.transform.a), abs(first_src.transform.e)
            )

        # Create temporary directory for reprojected files
        with tempfile.TemporaryDirectory(prefix="geotessera_reproject_") as temp_dir:
            # Prepare reprojection arguments
            total_files = len(geotiff_paths)
            reproject_args = []
            reprojected_files = []

            for i, geotiff_file in enumerate(geotiff_paths):
                reprojected_file = os.path.join(temp_dir, f"reprojected_{i}.tif")
                reprojected_files.append(reprojected_file)
                reproject_args.append(
                    (
                        geotiff_file,
                        reprojected_file,
                        target_crs,
                        source_resolution,
                        compress,
                    )
                )

            if progress_callback:
                progress_callback(0, total_files * 2 + 2, "Starting reprojection...")

            # Sequential reprojection
            failed_files = []
            for i, args in enumerate(reproject_args):
                if progress_callback:
                    progress_callback(
                        i,
                        total_files * 2 + 2,
                        f"Reprojecting file {i + 1}/{total_files}...",
                    )

                _, error = self._reproject_geotiff_file(args)

                if error:
                    failed_files.append((geotiff_paths[i], error))

            if failed_files:
                error_msg = f"Failed to reproject {len(failed_files)} files: "
                error_msg += ", ".join(
                    [f"{Path(f).name}: {e}" for f, e in failed_files[:3]]
                )
                if len(failed_files) > 3:
                    error_msg += f" and {len(failed_files) - 3} more"
                raise RuntimeError(error_msg)

            # Filter out any failed reprojections
            reprojected_files = [f for f in reprojected_files if os.path.exists(f)]

            if progress_callback:
                progress_callback(
                    total_files, total_files * 2 + 2, "Opening files for merging..."
                )

            # Open all reprojected files for merging
            src_files = [rasterio.open(f) for f in reprojected_files]

            try:
                if progress_callback:
                    progress_callback(
                        total_files + 1, total_files * 2 + 2, "Merging tiles..."
                    )

                # Merge tiles
                mosaic_array, mosaic_transform = merge(src_files, method="first")

                # Get metadata from first file
                first_src = src_files[0]
                profile = first_src.profile.copy()
                profile.update(
                    {
                        "height": mosaic_array.shape[1],
                        "width": mosaic_array.shape[2],
                        "transform": mosaic_transform,
                        "dtype": mosaic_array.dtype,  # Use mosaic array dtype
                        "compress": compress,
                        "tiled": True,
                        "blockxsize": 512,
                        "blockysize": 512,
                    }
                )

                if progress_callback:
                    progress_callback(
                        total_files * 2,
                        total_files * 2 + 2,
                        "Writing mosaic to disk...",
                    )

                # Write mosaic
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(mosaic_array)

                    # Copy band descriptions from first file
                    for band_idx in range(1, mosaic_array.shape[0] + 1):
                        band_desc = (
                            first_src.descriptions[band_idx - 1]
                            if first_src.descriptions
                            and band_idx <= len(first_src.descriptions)
                            else None
                        )
                        if band_desc:
                            dst.set_band_description(band_idx, band_desc)

                    # Update metadata
                    dst.update_tags(
                        TESSERA_TARGET_CRS=target_crs,
                        TESSERA_RESOLUTION=str(source_resolution),
                        TESSERA_TILE_COUNT=str(len(geotiff_paths)),
                        TESSERA_DESCRIPTION="GeoTessera satellite embedding mosaic",
                        GEOTESSERA_VERSION=__version__,
                    )

                if progress_callback:
                    progress_callback(
                        total_files * 2 + 2, total_files * 2 + 2, "Complete"
                    )

            finally:
                # Close all source files
                for src in src_files:
                    src.close()

        return str(output_path)

    def apply_pca_to_embeddings(
        self,
        embeddings: List[Tuple[int, float, float, np.ndarray, object, object]],
        n_components: int = 3,
        standardize: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[int, float, float, np.ndarray, object, object, Dict]]:
        """Apply PCA to embedding tiles for visualization.

        Args:
            embeddings: List of (year, tile_lon, tile_lat, embedding_array, crs, transform) tuples
            n_components: Number of principal components to extract (default: 3 for RGB)
            standardize: Whether to standardize features before PCA
            progress_callback: Optional callback function(current, total, status) for progress tracking

        Returns:
            List of tuples with PCA-transformed data:
                (year, tile_lon, tile_lat, pca_array, crs, transform, pca_info)
            where pca_info contains:
                - explained_variance: Explained variance ratio for each component
                - total_variance: Total explained variance

        Raises:
            ImportError: If scikit-learn is not available
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn required for PCA visualization: pip install scikit-learn"
            )

        if not embeddings:
            return []

        pca_results = []
        total_tiles = len(embeddings)

        if progress_callback:
            progress_callback(0, total_tiles, "Starting PCA analysis...")

        for i, (year, tile_lon, tile_lat, embedding, crs, transform) in enumerate(
            embeddings
        ):
            if progress_callback:
                progress_callback(
                    i,
                    total_tiles,
                    f"Processing tile {i + 1}/{total_tiles}: ({tile_lon:.2f}, {tile_lat:.2f})",
                )

            # Reshape for PCA: (height, width, channels) -> (pixels, channels)
            height, width, n_bands = embedding.shape
            data_reshaped = embedding.reshape(-1, n_bands)

            # Handle NaN values by replacing with 0
            data_reshaped = np.nan_to_num(data_reshaped, nan=0.0)

            # Standardize data if requested
            if standardize:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
            else:
                data_scaled = data_reshaped

            # Apply PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)

            # Reshape back to image: (pixels, n_components) -> (height, width, n_components)
            pca_image = pca_result.reshape(height, width, n_components)

            # Create PCA info dictionary
            pca_info = {
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "total_variance": float(pca.explained_variance_ratio_.sum()),
                "n_components": n_components,
                "standardized": standardize,
            }

            pca_results.append(
                (year, tile_lon, tile_lat, pca_image, crs, transform, pca_info)
            )

        if progress_callback:
            progress_callback(total_tiles, total_tiles, "PCA analysis complete")

        return pca_results

    def export_pca_geotiffs(
        self,
        tiles_to_fetch: Iterable[Tuple[int, float, float]],
        output_dir: Union[str, Path],
        n_components: int = 3,
        standardize: bool = True,
        compress: str = "lzw",
        normalize: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """Export PCA-transformed embeddings as GeoTIFF files.

        This method fetches embeddings, applies PCA transformation, and exports
        the results as GeoTIFF files suitable for RGB visualization.

        Args:
            tiles_to_fetch: List of tiles as (year, tile_lon, tile_lat) tuples
            output_dir: Directory to save PCA GeoTIFF files
            n_components: Number of principal components (default: 3 for RGB)
            standardize: Whether to standardize features before PCA
            compress: Compression method for GeoTIFF
            normalize: Whether to use global normalization across tiles (vs per-tile)
            progress_callback: Optional callback function(current, total, status)

        Returns:
            List of paths to created PCA GeoTIFF files
        """
        try:
            import rasterio
            from rasterio.enums import ColorInterp
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Fetch embeddings (0-40% progress)
        def fetch_progress(current, total, status=None):
            overall_progress = int((current / total) * 40)
            progress_callback(
                overall_progress, 100, status or f"Fetching tile {current}/{total}"
            )

        if progress_callback:
            progress_callback(0, 100, "Fetching embedding tiles...")

        # Materialise the generator: apply_pca_to_embeddings needs len() and the
        # results are iterated twice below, so a one-shot generator won't do.
        embeddings = list(
            self.fetch_embeddings(
                tiles_to_fetch, fetch_progress if progress_callback else None
            )
        )

        if not embeddings:
            if progress_callback:
                progress_callback(100, 100, "No tiles found in bounding box")
            return []

        # Phase 2: Apply PCA (40-70% progress)
        def pca_progress(current, total, status=None):
            overall_progress = int(40 + (current / total) * 30)
            progress_callback(
                overall_progress,
                100,
                status or f"Applying PCA to tile {current}/{total}",
            )

        pca_results = self.apply_pca_to_embeddings(
            embeddings,
            n_components,
            standardize,
            pca_progress if progress_callback else None,
        )

        # Phase 3: Export GeoTIFFs (70-100% progress)
        created_files = []
        if progress_callback:
            total_tiles = len(pca_results)

        # Calculate global min/max if normalize is True (for consistent scaling across tiles)
        if normalize:
            # Global normalization: find min/max across ALL tiles first
            global_min = [float("inf")] * n_components
            global_max = [float("-inf")] * n_components

            for _, _, _, pca_img, _, _, _ in pca_results:
                for j in range(n_components):
                    comp = pca_img[:, :, j]
                    global_min[j] = min(global_min[j], np.nanmin(comp))
                    global_max[j] = max(global_max[j], np.nanmax(comp))

        for i, (
            year,
            tile_lon,
            tile_lat,
            pca_image,
            crs,
            transform,
            pca_info,
        ) in enumerate(pca_results):
            if progress_callback:
                export_progress = int(70 + (i / total_tiles) * 30)
                filename = f"grid_{tile_lon:.2f}_{tile_lat:.2f}_{year}_pca.tiff"
                progress_callback(export_progress, 100, f"Writing {filename}...")

            # Always normalize PCA components to 0-255 for visualization
            pca_normalized = np.zeros_like(pca_image)
            for j in range(n_components):
                component = pca_image[:, :, j]

                if normalize:
                    # Use global min/max for consistent scaling across tiles
                    comp_min, comp_max = global_min[j], global_max[j]
                else:
                    # Use local min/max for per-tile normalization
                    comp_min, comp_max = np.nanmin(component), np.nanmax(component)

                if comp_max > comp_min:
                    pca_normalized[:, :, j] = (component - comp_min) / (
                        comp_max - comp_min
                    )
                else:
                    pca_normalized[:, :, j] = 0

            # Always convert to uint8 for visualization output
            output_data = (np.clip(pca_normalized, 0, 1) * 255).astype(np.uint8)
            dtype = "uint8"

            # Create filename and path
            filename = f"grid_{tile_lon:.2f}_{tile_lat:.2f}_{year}_pca.tiff"
            output_path = output_dir / filename

            # Get dimensions
            height, width = output_data.shape[:2]

            # Write PCA GeoTIFF
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=n_components,
                dtype=dtype,
                crs=crs,
                transform=transform,
                compress=compress,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                # Write each component as a band
                for band_idx in range(n_components):
                    dst.write(output_data[:, :, band_idx], band_idx + 1)

                    # Set band description with explained variance
                    variance_pct = pca_info["explained_variance"][band_idx] * 100
                    dst.set_band_description(
                        band_idx + 1, f"PC{band_idx + 1} ({variance_pct:.1f}% variance)"
                    )

                # Set color interpretation for RGB visualization
                if n_components >= 3 and normalize:
                    dst.colorinterp = [
                        ColorInterp.red,
                        ColorInterp.green,
                        ColorInterp.blue,
                    ][:n_components]

                # Add metadata
                dst.update_tags(
                    TESSERA_YEAR=str(year),
                    TESSERA_TILE_LON=str(tile_lon),
                    TESSERA_TILE_LAT=str(tile_lat),
                    PCA_COMPONENTS=str(n_components),
                    PCA_TOTAL_VARIANCE=f"{pca_info['total_variance']:.3f}",
                    PCA_EXPLAINED_VARIANCE=json.dumps(pca_info["explained_variance"]),
                    PCA_STANDARDIZED=str(standardize),
                    PCA_NORMALIZED=str(normalize),
                    GEOTESSERA_VERSION=__version__,
                )

            created_files.append(str(output_path))

        if progress_callback:
            progress_callback(
                100, 100, f"Created {len(created_files)} PCA GeoTIFF files"
            )

        return created_files
