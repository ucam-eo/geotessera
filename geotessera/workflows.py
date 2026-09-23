"""Region workflows shared by CLI commands."""

from pathlib import Path
import tempfile
import json
import hashlib
import logging

from .inputs import resolve_region


def open_stream(version="v1.1", variant=None, store_url=None, cache_dir=None):
    from .registry import _parse_dataset_version, variant_note, zarr_store_url
    from .store import GeoTesseraZarr

    if store_url is None:
        if variant is None and (
            note := variant_note(_parse_dataset_version(version)[1], "stream")
        ):
            logging.getLogger(__name__).warning(note)
        store_url = zarr_store_url(version, variant)
    return GeoTesseraZarr(store_url, cache_dir=cache_dir)


def stream_download(
    output,
    *,
    bbox=None,
    tile=None,
    region_file=None,
    country=None,
    year=2024,
    version="v1.1",
    variant=None,
    store_url=None,
    cache_dir=None,
    bands=None,
    depth=None,
    compress="lzw",
    dry_run=False,
):
    bounds, _ = resolve_region(
        bbox=bbox, tile=tile, region_file=region_file, country=country
    )
    client = open_stream(version, variant, store_url, cache_dir)
    selected = None if bands is None else list(map(int, bands.split(",")))
    return client.export_geotiffs(
        bounds,
        year,
        output,
        bands=selected,
        depth=depth,
        compress=compress,
        dry_run=dry_run,
    )


def stream_rgb(
    output,
    *,
    bbox=None,
    tile=None,
    region_file=None,
    country=None,
    year=2024,
    version="v1.1",
    variant=None,
    store_url=None,
    cache_dir=None,
    bands="0,1,2",
    depth=None,
):
    """Stream three embedding bands, then render them with common RGB scaling."""
    from .raster import create_rgb

    selected = list(map(int, bands.split(",")))
    if len(selected) != 3:
        raise ValueError("Web maps require exactly three band indices")
    with tempfile.TemporaryDirectory(prefix="geotessera_web_") as temporary:
        files = stream_download(
            temporary,
            bbox=bbox,
            tile=tile,
            region_file=region_file,
            country=country,
            year=year,
            version=version,
            variant=variant,
            store_url=store_url,
            cache_dir=cache_dir,
            bands=bands,
            depth=depth,
        )
        return Path(create_rgb(files, output))


def file_identity(path):
    """Identify a local artifact without reading a potentially large raster."""
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def stage_matches(path, expected):
    try:
        return json.loads(Path(path).read_text()) == expected
    except (OSError, ValueError):
        return False


def save_stage(path, state):
    from .remote import atomic_output

    with atomic_output(path, suffix=".json") as temporary:
        Path(temporary).write_text(json.dumps(state, sort_keys=True))


def cached_stream_rgb(output, *, force=False, **kwargs):
    """Reuse a completed rendering of the same region and source selection.

    Source contents are assumed immutable; force refreshes an updated store.
    Cache location is deliberately excluded from the rendering identity.
    """
    from .registry import zarr_store_url

    output = Path(output)
    bounds, _ = resolve_region(
        **{key: kwargs.get(key) for key in ("bbox", "tile", "region_file", "country")}
    )
    request = {
        key: value
        for key, value in kwargs.items()
        if key not in ("bbox", "tile", "region_file", "country", "cache_dir")
    }
    request["bounds"] = list(map(float, bounds))
    # Key on the resolved store, since defaults change between releases.
    request["store_url"] = kwargs.get("store_url") or zarr_store_url(
        kwargs.get("version", "v1.1"), kwargs.get("variant")
    )
    request["schema"] = 2
    # Persist a digest so authenticated store URLs are not written to disk.
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    marker = output.with_suffix(".json")
    if (
        not force
        and output.exists()
        and stage_matches(
            marker, {"request": digest, "artifact": file_identity(output)}
        )
    ):
        return output, True
    marker.unlink(missing_ok=True)
    result = stream_rgb(output, **kwargs)
    save_stage(marker, {"request": digest, "artifact": file_identity(result)})
    return result, False
