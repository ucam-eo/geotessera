"""Stream published Zarr embeddings into local GIS products."""

import logging
from pathlib import Path

import numpy as np
import rasterio

from .inputs import parse_bbox
from .registry import check_dataset_dir, dataset_tags, write_tessera_metadata
from .remote import atomic_output
from .store import _region_window, _time_index, _utm_envelope, _zone_for_lon

log = logging.getLogger(__name__)


def region_windows(client, bbox, year):
    """Yield intersecting native-zone windows without fetching embeddings."""
    bbox = parse_bbox(bbox)
    west, south, east, north = bbox
    last = _zone_for_lon(np.nextafter(east, west))
    for zone in range(_zone_for_lon(west), last + 1):
        try:
            ds = client.open_zone(zone=zone)
        except KeyError:
            continue
        _time_index(ds, year)
        for group, years in ds.attrs.get("geotessera:years_incomplete", {}).items():
            if year in years:
                log.warning("%s has no complete %d; it reads as nodata", group, year)
        zone_bbox = (max(west, zone * 6 - 186), south, min(east, zone * 6 - 180), north)
        try:
            window = _region_window(ds, _utm_envelope(zone_bbox, ds.tessera.crs))
        except IndexError:
            continue
        yield zone, ds, window


def export_region(
    client,
    bbox,
    year,
    output_dir,
    *,
    bands=None,
    depth=None,
    strip_rows=128,
    compress="lzw",
    progress_callback=None,
    dry_run=False,
):
    """Write one float32 GeoTIFF per intersecting UTM zone, in row strips.

    Only selected bands are read. Files carry NaN nodata, the native transform,
    and source/model provenance. A published dataset is also recorded in each
    file's tags and in the directory's ``tessera_metadata.json``, and a
    directory holding another dataset is refused. Each file is committed
    atomically. ``dry_run`` returns metadata estimates without reading
    embedding or scale chunks.
    """
    if strip_rows <= 0:
        raise ValueError("strip_rows must be positive")
    array, count = client._embeddings_array(depth)
    bands = list(range(count)) if bands is None else list(bands)
    if not bands or any(not isinstance(b, int) or not 0 <= b < count for b in bands):
        raise ValueError(f"Bands must be zero-based indices below {count}")
    windows = list(region_windows(client, bbox, year))
    if not windows:
        raise ValueError("Region does not intersect any available zone grid")
    estimates = [
        dict(
            zone=z,
            width=w.x1 - w.x0,
            height=w.y1 - w.y0,
            uncompressed_bytes=(w.x1 - w.x0) * (w.y1 - w.y0) * len(bands) * 4,
        )
        for z, _, w in windows
    ]
    if dry_run:
        return estimates
    output_dir = Path(output_dir)
    dataset = client.dataset
    if dataset is not None:
        check_dataset_dir(output_dir, dataset.version, dataset.variant)
        tags = dataset_tags(dataset.version, dataset.variant)
    else:
        tags = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    # Cap the working strip even for very wide regional exports. Account for
    # quantized input, dequantized output and scales (compressed I/O has its own buffers).
    def rows_for(window):
        return min(
            strip_rows,
            max(1, 64 * 1024**2 // ((window.x1 - window.x0) * (len(bands) * 9 + 9))),
        )

    total = sum((w.y1 - w.y0 + rows_for(w) - 1) // rows_for(w) for _, _, w in windows)
    completed = 0
    for zone, ds, window in windows:
        rows = rows_for(window)
        group = client._zone_arrays(zone)
        ti = _time_index(ds, year)
        path = output_dir / f"tessera_{year}_utm{zone:02d}.tif"
        with atomic_output(path, suffix=".tif") as temporary:
            with rasterio.open(
                temporary,
                "w",
                driver="GTiff",
                width=window.x1 - window.x0,
                height=window.y1 - window.y0,
                count=len(bands),
                dtype="float32",
                crs=ds.tessera.crs,
                transform=window.transform,
                nodata=np.nan,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress=compress,
                BIGTIFF="IF_SAFER",
            ) as dst:
                for top in range(window.y0, window.y1, rows):
                    end = min(top + rows, window.y1)
                    emb = group[array].oindex[ti, bands, top:end, window.x0 : window.x1]
                    scales = group["scales"][ti, top:end, window.x0 : window.x1]
                    data = ds.tessera.dequantise(emb, scales).transpose(2, 0, 1)
                    dst.write(
                        data,
                        window=rasterio.windows.Window(
                            0, top - window.y0, data.shape[2], data.shape[1]
                        ),
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, f"Writing UTM zone {zone}")
                dst.update_tags(
                    **tags,
                    TESSERA_YEAR=str(year),
                    TESSERA_SOURCE=client.url,
                    TESSERA_MODEL=client.model_version,
                    TESSERA_BUILD_VERSION=client.build_version,
                    TESSERA_BANDS=",".join(map(str, bands)),
                )
                for i, band in enumerate(bands, 1):
                    dst.set_band_description(i, f"Tessera_Band_{band}")
        files.append(str(path))
    if dataset is not None:
        write_tessera_metadata(
            output_dir,
            dataset.version,
            dataset.variant,
            extra={"format": "tiff", "year": year, "source": client.url},
        )
    return files
