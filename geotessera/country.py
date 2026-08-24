"""Country lookup functionality using Natural Earth data."""

from typing import Tuple, Optional, List, Dict, Callable
import geopandas as gpd
import zipfile
import os
from pathlib import Path
import difflib


class CountryLookup:
    """Provides country name to geometry and bounding box lookup using Natural Earth data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """Initialize with optional cache directory for Natural Earth data.

        Args:
            cache_dir: Optional cache directory path
            progress_callback: Optional callback(current, total, status) for progress updates
        """
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            # Use platform-appropriate cache directory
            if os.name == "nt":
                base = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            self._cache_dir = base / "geotessera"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._countries_gdf: Optional[gpd.GeoDataFrame] = None
        self._name_lookup: Optional[Dict[str, str]] = None
        self._progress_callback = progress_callback

    def _get_countries_data_path(self) -> Path:
        """Download and extract Natural Earth countries data, return path to GeoJSON."""
        # Check if data already exists
        extract_dir = self._cache_dir / "natural-earth-data"
        geojson_path = (
            extract_dir
            / "natural-earth-vector-5.1.2"
            / "geojson"
            / "ne_110m_admin_0_countries.geojson"
        )

        if geojson_path.exists():
            if self._progress_callback:
                self._progress_callback(100, 100, "Country data already cached")
            return geojson_path

        # Report download starting
        if self._progress_callback:
            self._progress_callback(0, 100, "Downloading country boundaries...")

        from .registry import download_file_to_temp

        url = "https://github.com/nvkelso/natural-earth-vector/archive/refs/tags/v5.1.2.zip"
        archive_path = self._cache_dir / "natural-earth-v5.1.2.zip"

        # Downloading is the first half of this task's 0-100 scale and
        # extraction the second. GitHub often omits Content-Length, so fall
        # back to the archive's usual ~18 MB as the denominator.
        progress_adapter = None
        if self._progress_callback:
            estimated_total = 18 * 1024 * 1024

            def progress_adapter(downloaded, total, status):
                denom = total or estimated_total
                pct = min(50, int(downloaded / denom * 50))
                self._progress_callback(pct, 100, status)

        download_file_to_temp(
            url, progress_callback=progress_adapter, cache_path=archive_path
        )

        # Extract the specific GeoJSON file we need
        if self._progress_callback:
            self._progress_callback(50, 100, "Extracting country data...")

        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Extract only the file we need
            zip_ref.extract(
                "natural-earth-vector-5.1.2/geojson/ne_110m_admin_0_countries.geojson",
                extract_dir,
            )

        if self._progress_callback:
            self._progress_callback(100, 100, "Country data ready")

        return geojson_path

    def _load_countries_data(self) -> gpd.GeoDataFrame:
        """Load countries data from Natural Earth GeoJSON."""
        if self._countries_gdf is None:
            geojson_path = self._get_countries_data_path()
            self._countries_gdf = gpd.read_file(geojson_path)
        return self._countries_gdf

    def _build_name_lookup(self) -> Dict[str, str]:
        """Build lookup dictionary for country name variations."""
        if self._name_lookup is not None:
            return self._name_lookup

        countries = self._load_countries_data()
        lookup = {}

        for _, row in countries.iterrows():
            name_en = row.get("NAME_EN", "").strip()
            name_long = row.get("NAME_LONG", "").strip()
            iso_a2 = row.get("ISO_A2", "").strip()
            iso_a3 = row.get("ISO_A3", "").strip()

            if not name_en:
                continue

            # Primary name (case-insensitive)
            lookup[name_en.lower()] = name_en

            # Long name if different
            if name_long and name_long != name_en:
                lookup[name_long.lower()] = name_en

            # ISO codes
            if iso_a2 and iso_a2 != "-99":
                lookup[iso_a2.lower()] = name_en
            if iso_a3 and iso_a3 != "-99":
                lookup[iso_a3.lower()] = name_en

        # Add common aliases
        aliases = {
            "uk": "United Kingdom",
            "usa": "United States of America",
            "us": "United States of America",
            "russia": "Russia",
            "south korea": "South Korea",
            "north korea": "North Korea",
        }

        for alias, canonical in aliases.items():
            if canonical.lower() in lookup:
                lookup[alias.lower()] = lookup[canonical.lower()]

        self._name_lookup = lookup
        return lookup

    def _resolve_country_name(self, country_name: str) -> str:
        """Resolve country name to canonical form."""
        lookup = self._build_name_lookup()
        normalized = country_name.strip().lower()

        # Direct lookup
        if normalized in lookup:
            return lookup[normalized]

        # Fuzzy matching
        matches = difflib.get_close_matches(normalized, lookup.keys(), n=1, cutoff=0.8)

        if matches:
            return lookup[matches[0]]

        raise ValueError(
            f"Country '{country_name}' not found. Use list_countries() to see available options."
        )

    def get_bbox(self, country_name: str) -> Tuple[float, float, float, float]:
        """Get bounding box for country as (west, south, east, north)."""
        canonical_name = self._resolve_country_name(country_name)
        countries = self._load_countries_data()

        country_row = countries[countries["NAME_EN"] == canonical_name]
        if country_row.empty:
            raise ValueError(f"Country '{canonical_name}' not found in dataset")

        bounds = country_row.iloc[0].geometry.bounds
        return bounds  # (west, south, east, north)

    def get_geometry(self, country_name: str) -> gpd.GeoDataFrame:
        """Get full country geometry for precise tile intersection."""
        canonical_name = self._resolve_country_name(country_name)
        countries = self._load_countries_data()

        country_gdf = countries[countries["NAME_EN"] == canonical_name].copy()
        if country_gdf.empty:
            raise ValueError(f"Country '{canonical_name}' not found in dataset")

        return country_gdf

    def list_countries(self) -> List[str]:
        """List all available country names."""
        countries = self._load_countries_data()
        return sorted(countries["NAME_EN"].dropna().tolist())

# Global instance for convenience
_country_lookup = None


def get_country_lookup(progress_callback: Optional[Callable] = None) -> CountryLookup:
    """Get global CountryLookup instance.

    Args:
        progress_callback: Optional callback for progress updates when downloading data
    """
    global _country_lookup
    if _country_lookup is None or progress_callback is not None:
        _country_lookup = CountryLookup(progress_callback=progress_callback)
    return _country_lookup


def get_country_bbox(
    country_name: str, progress_callback: Optional[Callable] = None
) -> Tuple[float, float, float, float]:
    """Simple function to get country bounding box.

    Args:
        country_name: Name of the country
        progress_callback: Optional callback for progress updates when downloading data
    """
    return get_country_lookup(progress_callback).get_bbox(country_name)


