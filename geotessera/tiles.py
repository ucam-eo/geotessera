"""Tile abstraction for format-agnostic embedding access."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import re
from .registry import (
    EMBEDDINGS_DIR_NAME,
    LANDMASKS_DIR_NAME,
    tile_to_embedding_paths,
    tile_to_landmask_filename,
)


class Tile:
    """A single embedding tile that abstracts storage format.

    A tile can be stored in two formats:
    - NPY: quantized embedding + scales + landmask (downloaded format)
    - GeoTIFF: dequantized embedding with CRS/transform baked in

    For zarr store access, use GeoTesseraZarr in store.py instead.

    Every tile has:
    - Geographic identity (lon, lat, year)
    - Spatial metadata (crs, transform, bounds, height, width)
    - Embedding data (loaded on demand via load_embedding())
    """

    def __init__(self, lon: float, lat: float, year: int):
        """Create a tile reference.

        Args:
            lon: Tile center longitude (on 0.05 grid)
            lat: Tile center latitude (on 0.05 grid)
            year: Year of embeddings
        """
        self.lon = lon
        self.lat = lat
        self.year = year

        # Format-specific file paths
        self._format = None  # 'npy' or 'geotiff'
        self._geotiff_path = None
        self._embedding_path = None
        self._scales_path = None
        self._landmask_path = None

        # Spatial metadata (loaded during construction)
        self.crs = None
        self.transform = None
        self.bounds = None
        self.height = None
        self.width = None

    @property
    def grid_name(self) -> str:
        """Grid name like 'grid_0.15_52.05'."""
        return f"grid_{self.lon:.2f}_{self.lat:.2f}"

    # -------------------------------------------------------------------------
    # Loading - format agnostic
    # -------------------------------------------------------------------------

    def load_embedding(self) -> np.ndarray:
        """Load dequantized embedding data.

        Returns:
            Array of shape (height, width, 128) - always dequantized
        """
        if self._format == "npy":
            return self._load_from_npy()
        elif self._format == "geotiff":
            return self._load_from_geotiff()
        else:
            raise ValueError(f"Unknown format: {self._format}")

    def _load_from_npy(self) -> np.ndarray:
        """Load and dequantize from NPY format."""
        from geotessera.core import dequantize_embedding

        quantized = np.load(self._embedding_path)
        scales = np.load(self._scales_path)
        return dequantize_embedding(quantized, scales)

    def _load_from_geotiff(self) -> np.ndarray:
        """Load dequantized data from GeoTIFF."""
        import rasterio

        with rasterio.open(self._geotiff_path) as src:
            # (bands, H, W) -> (H, W, bands)
            return np.transpose(src.read(), (1, 2, 0))

    def is_available(self, require_landmask: bool = True) -> bool:
        """Check if all required files exist.

        Args:
            require_landmask: If True (default), landmask must exist for NPY format tiles.
                             For GeoTIFF format, this parameter is ignored.
        """
        if self._format == "npy":
            has_embedding = self._embedding_path.exists() and self._scales_path.exists()
            if require_landmask:
                return has_embedding and self._landmask_path.exists()
            else:
                return has_embedding
        elif self._format == "geotiff":
            return self._geotiff_path.exists()
        else:
            return False

    # -------------------------------------------------------------------------
    # Factory methods - construct from different formats
    # -------------------------------------------------------------------------

    @classmethod
    def from_npy(cls, embedding_path: Path, base_dir: Path) -> "Tile":
        """Create from NPY format files.

        Args:
            embedding_path: Path to .npy file (e.g., global_0.1_degree_representation/2024/grid_0.15_52.05.npy)
            base_dir: Base directory containing embeddings and landmasks subdirectories

        Returns:
            Tile instance backed by NPY storage
        """
        # Parse coordinates from filename
        lon, lat, year = _parse_npy_filename(embedding_path)
        tile = cls(lon, lat, year)

        # Set format and paths
        tile._format = "npy"
        tile._embedding_path = Path(embedding_path)
        tile._scales_path = (
            tile._embedding_path.parent / f"{tile._embedding_path.stem}_scales.npy"
        )
        tile._landmask_path = (
            Path(base_dir) / LANDMASKS_DIR_NAME / tile_to_landmask_filename(lon, lat)
        )

        # Load spatial metadata from landmask (required)
        if not tile._landmask_path.exists():
            raise FileNotFoundError(
                f"Landmask file not found: {tile._landmask_path}\n"
                f"Landmask files are required for NPY format tiles.\n"
                f"Expected: {base_dir}/{LANDMASKS_DIR_NAME}/{tile_to_landmask_filename(lon, lat)}"
            )
        tile._load_spatial_metadata_from_landmask()

        return tile

    @classmethod
    def from_geotiff(cls, geotiff_path: Path) -> "Tile":
        """Create from GeoTIFF file.

        Args:
            geotiff_path: Path to GeoTIFF file

        Returns:
            Tile instance backed by GeoTIFF storage
        """
        # Parse coordinates from filename or metadata
        lon, lat, year = _parse_geotiff_filename(geotiff_path)
        tile = cls(lon, lat, year)

        # Set format and path
        tile._format = "geotiff"
        tile._geotiff_path = Path(geotiff_path)

        # Load spatial metadata from GeoTIFF
        tile._load_spatial_metadata_from_geotiff()

        return tile

    def _load_spatial_metadata_from_landmask(self):
        """Load spatial metadata from landmask (for NPY format)."""
        import rasterio

        with rasterio.open(self._landmask_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds
            self.height = src.height
            self.width = src.width

    def _load_spatial_metadata_from_geotiff(self):
        """Load spatial metadata from GeoTIFF."""
        import rasterio

        with rasterio.open(self._geotiff_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds
            self.height = src.height
            self.width = src.width

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if this tile contains a point.

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees

        Returns:
            True if point is within tile bounds
        """
        half_size = 0.05
        return (
            self.lon - half_size <= lon < self.lon + half_size
            and self.lat - half_size <= lat < self.lat + half_size
        )

    def sample_at_point(self, lon: float, lat: float) -> np.ndarray:
        """Sample embedding at a single point.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Embedding vector of shape (128,) or array of NaNs if point outside tile
        """
        if not self.contains_point(lon, lat):
            return np.full(128, np.nan)

        # Load embedding data
        data = self.load_embedding()

        # Transform point to pixel coordinates
        from rasterio.transform import rowcol

        row, col = rowcol(self.transform, lon, lat)

        # Check bounds
        if 0 <= row < self.height and 0 <= col < self.width:
            return data[row, col, :]
        else:
            return np.full(128, np.nan)

    def to_dict(self) -> Dict:
        """Convert to dictionary format (for compatibility with visualization code).

        Returns:
            Dict with keys: path, data, crs, transform, bounds, height, width
        """
        return {
            "path": self.grid_name,
            "data": self.load_embedding(),
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"Tile(lon={self.lon}, lat={self.lat}, year={self.year}, format={self._format})"

    def __hash__(self):
        return hash((self.lon, self.lat, self.year))

    def __eq__(self, other):
        return (self.lon, self.lat, self.year) == (other.lon, other.lat, other.year)


# ============================================================================
# Discovery functions - find tiles in a directory
# ============================================================================


def discover_tiles(directory: Path) -> List[Tile]:
    """Auto-detect format and discover all tiles.

    Prefers NPY format when both NPY and GeoTIFF formats are present.
    The local layout always uses ``global_0.1_degree_representation/`` as the
    embeddings subdir regardless of the dataset variant — variant info is
    recorded in the ``tessera_metadata.json`` sidecar file.

    Args:
        directory: Directory containing tiles

    Returns:
        List of Tile objects with spatial metadata loaded, sorted by (year, lat, lon)
    """
    # Check for NPY format first by looking for .npy files in embeddings directory
    # Preferred order is NPY, tiff, as NPY (more efficient, includes scales)
    embeddings_dir = directory / EMBEDDINGS_DIR_NAME
    if embeddings_dir.exists() and embeddings_dir.is_dir():
        # Check if there are any .npy files (not just _scales.npy)
        # The actual pattern validation happens in discover_npy_tiles()
        npy_files = [
            f
            for f in embeddings_dir.rglob("*.npy")
            if not f.name.endswith("_scales.npy")
        ]
        if npy_files:
            return discover_npy_tiles(directory)

    # Then try to search GeoTIFF files (will search recursively)
    tiff_files = discover_geotiff_tiles(directory)

    if tiff_files:
        return tiff_files

    return []


def tile_for_coord(base_dir: Path, lon: float, lat: float, year: int) -> Optional[Tile]:
    """Resolve a single tile from its coordinates, without listing the directory.

    A tile's location under *base_dir* follows from ``(lon, lat, year)``
    alone, so a caller that already knows which tile it wants can go
    straight to it. Prefer this to :func:`discover_tiles`, which reads
    every tile in *base_dir* and is correspondingly slow on a large or
    network-mounted mirror.

    Args:
        base_dir: Directory holding the embeddings and landmasks subdirs.
        lon: Tile centre longitude, on the 0.05 grid.
        lat: Tile centre latitude, on the 0.05 grid.
        year: Year of embeddings.

    Returns:
        The tile, or None if it is absent or unreadable. A tile counts as
        absent unless its embedding, scales and landmask are all present.
    """
    import logging

    base_dir = Path(base_dir)
    embedding_rel, scales_rel = tile_to_embedding_paths(lon, lat, year)
    embeddings_root = base_dir / EMBEDDINGS_DIR_NAME
    embedding_path = embeddings_root / embedding_rel
    landmask_path = base_dir / LANDMASKS_DIR_NAME / tile_to_landmask_filename(lon, lat)

    if not (
        embedding_path.exists()
        and (embeddings_root / scales_rel).exists()
        and landmask_path.exists()
    ):
        return None

    try:
        return Tile.from_npy(embedding_path, base_dir)
    except Exception as e:
        logging.warning(f"Failed to load tile {embedding_path}: {e}")
        return None


def tiles_for_coords(
    base_dir: Path, coords: Iterable[Tuple[float, float]], year: int
) -> Dict[Tuple[float, float], Tile]:
    """Resolve the tiles for *coords*, omitting any that are absent.

    Args:
        base_dir: Directory holding the embeddings and landmasks subdirs.
        coords: Tile centres as ``(lon, lat)`` pairs.
        year: Year of embeddings.

    Returns:
        The tiles that were found, keyed by ``(lon, lat)``.
    """
    found = {}
    for lon, lat in coords:
        tile = tile_for_coord(base_dir, lon, lat, year)
        if tile is not None:
            found[(lon, lat)] = tile
    return found


def discover_npy_tiles(base_dir: Path) -> List[Tile]:
    """Discover NPY format tiles.

    Args:
        base_dir: Directory containing embeddings and landmasks subdirectories.
            Tiles are expected under ``base_dir/global_0.1_degree_representation/``
            regardless of dataset variant (variant info lives in
            ``tessera_metadata.json``).

    Returns:
        List of Tile objects with spatial metadata loaded
    """
    import logging

    tiles = []
    embeddings_dir = base_dir / EMBEDDINGS_DIR_NAME

    if not embeddings_dir.exists():
        logging.warning(f"Embeddings directory not found: {embeddings_dir}")
        return []

    for npy_file in embeddings_dir.rglob("*.npy"):
        # Skip scales files
        if npy_file.name.endswith("_scales.npy"):
            continue

        try:
            tile = Tile.from_npy(npy_file, base_dir)
            if tile.is_available():
                tiles.append(tile)
            else:
                logging.warning(f"Skipping incomplete tile: {npy_file}")
        except ValueError:
            # Skip files that don't match expected filename pattern
            # ValueError is raised by _parse_npy_filename when pattern doesn't match
            continue
        except Exception as e:
            logging.warning(f"Failed to load tile {npy_file}: {e}")

    return sorted(tiles, key=lambda t: (t.year, t.lat, t.lon))


def discover_geotiff_tiles(directory: Path) -> List[Tile]:
    """Discover GeoTIFF tiles.

    Args:
        directory: Directory containing .tif/.tiff files

    Returns:
        List of Tile objects with spatial metadata loaded
    """
    import logging

    tiles = []

    for pattern in ["*.tif", "*.tiff"]:
        for geotiff_file in directory.rglob(pattern):
            # Skip landmask files (they're in a different directory and have different naming)
            if LANDMASKS_DIR_NAME in geotiff_file.parts:
                continue

            try:
                tile = Tile.from_geotiff(geotiff_file)
                tiles.append(tile)
            except ValueError:
                # Skip files that don't match expected filename pattern
                # ValueError is raised by _parse_geotiff_filename when pattern doesn't match
                continue
            except Exception as e:
                logging.warning(f"Failed to load tile {geotiff_file}: {e}")

    return sorted(tiles, key=lambda t: (t.year, t.lat, t.lon))


def discover_formats(directory: Path) -> Dict[str, List[Tile]]:
    """Discover tiles in all available formats.

    Args:
        directory: Directory containing tiles. NPY tiles are scanned under
            ``directory/global_0.1_degree_representation/``.

    Returns:
        Dictionary mapping format names to lists of tiles:
        {'npy': [...], 'geotiff': [...]}
    """
    formats = {}

    # Check for NPY format
    npy_tiles = discover_npy_tiles(directory)
    if npy_tiles:
        formats["npy"] = npy_tiles

    # Check for GeoTIFF format
    geotiff_tiles = discover_geotiff_tiles(directory)
    if geotiff_tiles:
        formats["geotiff"] = geotiff_tiles

    return formats


# ============================================================================
# Helper functions
# ============================================================================


def _parse_npy_filename(path: Path) -> Tuple[float, float, int]:
    """Parse lon, lat, year from NPY filename.

    Example: embeddings/2024/grid_0.15_52.05.npy -> (0.15, 52.05, 2024)

    Args:
        path: Path to NPY file

    Returns:
        Tuple of (lon, lat, year)

    Raises:
        ValueError: If filename cannot be parsed
    """
    # Extract year from grandparent directory (platform-independent)
    # Expected structure: .../embeddings/<year>/grid_<lon>_<lat>/grid_<lon>_<lat>.npy
    grandparent_name = path.parent.parent.name
    if not re.fullmatch(r"\d{4}", grandparent_name):
        raise ValueError(f"Cannot extract year from path: {path}")
    year = int(grandparent_name)

    # Extract coordinates from filename
    match = re.match(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)\.npy", path.name)
    if not match:
        raise ValueError(f"Cannot parse coordinates from filename: {path.name}")

    lon = float(match.group(1))
    lat = float(match.group(2))

    return lon, lat, year


def _parse_geotiff_filename(path: Path) -> Tuple[float, float, int]:
    """Parse lon, lat, year from GeoTIFF filename.

    Tries multiple patterns. If parsing fails, raises ValueError.

    Args:
        path: Path to GeoTIFF file

    Returns:
        Tuple of (lon, lat, year)

    Raises:
        ValueError: If filename cannot be parsed
    """
    # Try pattern: grid_0.15_52.05_2024.tif
    match = re.match(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{4})\.tiff?", path.name)
    if match:
        return float(match.group(1)), float(match.group(2)), int(match.group(3))

    # If no patterns match, raise an error
    raise ValueError(
        f"Cannot parse GeoTIFF filename: {path.name}. Expected format: grid_<lon>_<lat>_<year>.tif"
    )
