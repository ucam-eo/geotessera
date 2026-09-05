# GitHub Copilot Instructions for GeoTessera

## Project Overview

GeoTessera is a Python library for accessing and working with Tessera geospatial foundation model embeddings. The library provides tools to download, process, and visualize satellite imagery embeddings from Sentinel-1 and Sentinel-2 data at 10m resolution.

### Core Architecture

- **Two-step workflow**: Retrieve embeddings (numpy arrays) → Export to desired format (GeoTIFF/NPY)
- **Registry system**: Parquet-based metadata registry for efficient tile lookup
- **0.1-degree grid**: Tiles cover ~11km × 11km, named by center coordinates
- **Direct HTTP downloads**: On-demand tile fetching with automatic cleanup
- **Hash verification**: SHA256 checksums ensure data integrity by default

## Technology Stack

### Core Dependencies

- **Python**: 3.12 or 3.13 required (`requires-python = ">=3.12"`)
- **CLI Framework**: `typer` with `rich` for interactive output
- **Geospatial**: `rasterio`, `geopandas`, `rioxarray` for GIS operations
- **Data Processing**: `numpy`, `pandas`, `pyarrow` (for Parquet registry)
- **Visualization**: `matplotlib`, `scikit-learn` (PCA), `scikit-image`
- **Storage**: `zarr`, `xarray`, `dask` for chunked data handling

### Build System

- **Package Manager**: Uses `uv` for dependency management (preferred) or `pip`
- **Configuration**: `pyproject.toml` with setuptools backend
- **Lock File**: `uv.lock` for reproducible builds
- **Test Frameworks**: `pytest` for Python APIs and `cram` for CLI behavior
- **Linting**: `ruff` for code quality

## Coding Standards

### Python Style

- Follow PEP 8 conventions
- Use type hints for function signatures (e.g., `Optional[str]`, `Path`, etc.)
- Use `typing_extensions.Annotated` for CLI argument annotations with `typer`
- Prefer pathlib `Path` over string paths
- Use f-strings for string formatting

### Code Organization

```
geotessera/
├── __init__.py          # Package exports and version
├── core.py              # Main GeoTessera class
├── registry.py          # Parquet registry management
├── cli.py               # Main CLI commands
├── registry_cli.py      # Registry-specific CLI
├── store.py             # Zarr v3 store access (GeoTesseraZarr)
├── tiles.py             # Tile abstraction (GeoTIFF + NPY)
├── visualization.py     # Visualization functions
├── web.py              # Web map generation
├── country.py          # Country bounding box utilities
└── progress.py         # Progress tracking utilities
```

### Key Patterns

1. **Rich Console Output**: Use `rich.console.Console` and `rich.progress.Progress` for user-facing output
2. **Logging**: Configure with `rich.logging.RichHandler` for pretty logs
3. **Temporary Files**: Use `tempfile` for intermediate data, clean up automatically
4. **Error Handling**: Provide clear error messages with context
5. **CLI Structure**: Commands are typer apps with descriptive help text

## Testing Guidelines

### Test Frameworks

Python API tests are written as pytest modules. CLI behavior is tested in `.t`
files using cram:

```bash
# Example test structure
Setup environment:
  $ export TERM=dumb
  $ export TESTDIR="$CRAMTMP/test_outputs"

Run command and check output:
  $ geotessera version
  [version number]
```

### Test Structure

- `tests/cli.t` - CLI command tests
- `tests/tile.t` - Single-tile / `--tile` tests
- `tests/viz.t` - Visualization tests
- `tests/v11.t` - v1.1 / variant filtering + sidecar tests
- `tests/test_store.py` - Zarr reader and point/region behavior
- `tests/test_zarr.py` - Remote-style Zarr I/O and build state
- `tests/test_tile_lookup.py` - Scoped tile discovery and download failures

### Running Tests

```bash
# Run all tests
env MPLCONFIGDIR=/tmp/geotessera-mpl uv run pytest tests -v
env TERM=dumb TTY_INTERACTIVE=0 uv run cram tests -v

# Run specific test file
env TERM=dumb TTY_INTERACTIVE=0 uv run cram tests/cli.t -v
```

### Testing Best Practices

- Set `TERM=dumb` to disable ANSI output in tests
- Use `$CRAMTMP` for temporary test data
- Override `XDG_CACHE_HOME` for isolated caching
- Check command exit codes and output patterns
- Test both success and error cases

## Build and Development Workflow

### Initial Setup

```bash
# Clone repository
git clone https://github.com/ucam-eo/geotessera
cd geotessera

# Install with uv (preferred)
uv sync --locked --all-extras --dev

# Or with pip
pip install -e .
```

### Development Commands

```bash
# Run tests
env MPLCONFIGDIR=/tmp/geotessera-mpl uv run pytest tests -v
env TERM=dumb TTY_INTERACTIVE=0 uv run cram tests -v

# Run CLI locally
uv run -m geotessera.cli --help
python -m geotessera.cli --help

# Lint code
ruff check .
ruff format .
```

### CI/CD

- GitHub Actions workflow: `.github/workflows/ci.yml`
- Multi-platform testing: Ubuntu, macOS (Intel & Apple Silicon)
- Python versions: 3.12, 3.13
- Dependencies: GDAL must be installed before Python packages
- Tests run with both `uv run pytest tests -v` and `uv run cram tests -v`

## Key Concepts to Remember

### Coordinate System

- Tiles use WGS84 coordinates (longitude, latitude)
- Tile naming: `grid_{lon}_{lat}` (e.g., `grid_0.15_52.05`)
- Bounding box format: `(min_lon, min_lat, max_lon, max_lat)`
- GeoTIFF exports use UTM projection from landmask tiles

### Data Files

1. **Embeddings**: `grid_0.15_52.05.npy` - int8 quantized arrays (H×W×128)
2. **Scales**: `grid_0.15_52.05_scales.npy` - float32 scale factors
3. **Landmasks**: `grid_0.15_52.05.tiff` - UTM projection + land/water masks
4. **Registry**: `registry.parquet` - Parquet metadata with tile locations & hashes

### Hash Verification

- Enabled by default for security
- Verifies embedding, scale, and landmask files
- Can be disabled: `--skip-hash` flag or `GEOTESSERA_SKIP_HASH=1`
- Use `verify_hashes=False` parameter in Python API

### Cache Behavior

- Only registry.parquet is cached (~few MB)
- Embedding/landmask tiles downloaded to temp files, cleaned up immediately
- Cache location: `~/.cache/geotessera` (Linux/macOS) or `%LOCALAPPDATA%/geotessera` (Windows)
- Override with `--cache-dir` or `cache_dir` parameter

## Common Operations

### Adding a New CLI Command

1. Add command function to `cli.py` with `@app.command()` decorator
2. Use type hints with `typer.Option()` or `typer.Argument()`
3. Add docstring for help text
4. Use `rich.console.Console` for output
5. Add test in appropriate `.t` file

### Working with Embeddings

```python
# Fetch single tile
embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)

# Fetch region
tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
embeddings = gt.fetch_embeddings(tiles)

# Export as GeoTIFF
files = gt.export_embedding_geotiffs(tiles_to_fetch=tiles, output_dir="./output")
```

### Registry Operations

```python
# Initialize with custom registry
gt = GeoTessera(registry_path="/path/to/registry.parquet")

# Query available tiles
tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

# Disable hash verification
gt = GeoTessera(verify_hashes=False)
```

## Documentation

- Main docs: `README.md` - comprehensive usage guide
- API docs: Built with Sphinx (see `docs/` directory)
- Hosted at: https://geotessera.readthedocs.io
- Changelog: `CHANGES.md`

## Contributing

When making changes:

1. Keep modifications minimal and focused
2. Follow existing code patterns and style
3. Add tests for new functionality
4. Update documentation if needed
5. Ensure CI passes (build + tests on all platforms)
6. Test with multiple Python versions if possible

## Important Notes

- **GDAL dependency**: Must be installed system-wide before Python packages
- **Rich output**: Disable with `TERM=dumb` for non-interactive environments
- **Registry updates**: New tiles require registry regeneration
- **Embedding requests**: Users can request new geographic coverage via GitHub issues
- **Minimal storage**: Only registry is cached; tiles are ephemeral

## Version Information

Version defined in `pyproject.toml` and exported from `geotessera/__init__.py`.

## Links

- Repository: https://github.com/ucam-eo/geotessera
- PyPI: https://pypi.org/project/geotessera/
- Tessera Model: https://github.com/ucam-eo/tessera
- Documentation: https://geotessera.readthedocs.io/
- Issue Tracker: https://github.com/ucam-eo/geotessera/issues
