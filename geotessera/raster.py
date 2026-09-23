"""Windowed raster operations, independent of data discovery and downloads."""

from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

from .remote import atomic_output


def _check_one_dataset(paths, sources):
    """Raise ``ValueError`` if *sources* hold more than one dataset.

    Rasters without dataset tags are not checked.
    """
    from .registry import dataset_from_tags

    found = {}
    for path, src in zip(paths, sources):
        dataset = dataset_from_tags(src.tags())
        if dataset is not None:
            found.setdefault(dataset, path)
    if len(found) > 1:
        listed = ", ".join(f"v{v} {var} ({p})" for (v, var), p in found.items())
        raise ValueError(
            f"Cannot merge embeddings of different datasets: {listed}. "
            f"Embeddings of different datasets cannot be interchanged."
        )


def merge_geotiffs(
    paths,
    output_path,
    target_crs="EPSG:3857",
    compress="lzw",
    progress_callback=None,
    *,
    bands=None,
    bounds=None,
):
    """Merge valid pixels to disk with virtual warps and bounded working memory.

    ``bands`` uses zero-based indices; ``bounds`` is in the target CRS.
    Inputs tagged with different datasets raise ``ValueError``.
    Uncovered pixels are NaN. The first valid source wins at overlaps,
    including when its value is zero. Native grids are retained when possible.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("No GeoTIFF files provided")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = rasterio.crs.CRS.from_user_input(target_crs)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        first = sources[0]
        if any(src.crs is None for src in sources):
            raise ValueError("Every input raster must have a CRS")
        _check_one_dataset(paths, sources)
        indexes = None if bands is None else [int(b) + 1 for b in bands]
        if indexes is not None and (
            not indexes
            or min(indexes) < 1
            or any(max(indexes) > src.count for src in sources)
        ):
            raise ValueError("Band indices must be within every input raster")
        if first.crs == target:
            resolution = first.res
        else:
            transform, _, _ = calculate_default_transform(
                first.crs, target, first.width, first.height, *first.bounds
            )
            resolution = abs(transform.a), abs(transform.e)
        warped = []
        for i, src in enumerate(sources):
            if progress_callback:
                progress_callback(
                    i, len(sources) + 1, f"Opening raster {i + 1}/{len(sources)}"
                )
            if src.crs == target and src.res == resolution:
                warped.append(src)
            else:
                transform, width, height = calculate_default_transform(
                    src.crs,
                    target,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=resolution,
                )
                warped.append(
                    stack.enter_context(
                        WarpedVRT(
                            src,
                            crs=target,
                            transform=transform,
                            width=width,
                            height=height,
                            dtype="float32",
                            nodata=np.nan,
                            resampling=rasterio.enums.Resampling.bilinear,
                        )
                    )
                )
        with atomic_output(output_path, suffix=".tif") as temporary:
            merge(
                warped,
                bounds=bounds,
                res=resolution,
                indexes=indexes,
                dtype="float32",
                nodata=np.nan,
                method="first",
                mem_limit=64,
                dst_path=temporary,
                dst_kwds=dict(
                    driver="GTiff",
                    compress=compress,
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    BIGTIFF="IF_SAFER",
                ),
            )
            with rasterio.open(temporary, "r+") as dst:
                dst.update_tags(**first.tags())
                dst.update_tags(
                    TESSERA_TARGET_CRS=str(target), TESSERA_TILE_COUNT=str(len(paths))
                )
                for out_band, in_band in enumerate(
                    indexes or range(1, first.count + 1), 1
                ):
                    description = first.descriptions[in_band - 1]
                    if description:
                        dst.set_band_description(out_band, description)
    if progress_callback:
        progress_callback(len(paths) + 1, len(paths) + 1, "Complete")
    return str(output_path)


def create_rgb(
    paths, output_path, bands=(0, 1, 2), target_crs="EPSG:3857", progress_callback=None
):
    """Select bands before merging, then normalize and write RGB by window."""
    if len(bands) != 3:
        raise ValueError("Exactly three RGB bands are required")
    with tempfile.TemporaryDirectory(prefix="geotessera_rgb_") as temporary:
        merged = Path(temporary) / "selected.tif"
        merge_geotiffs(
            paths, merged, target_crs, bands=bands, progress_callback=progress_callback
        )
        with rasterio.open(merged) as src:
            low, high = np.full(3, np.inf), np.full(3, -np.inf)
            for _, window in src.block_windows(1):
                block = src.read(window=window)
                for i in range(3):
                    valid = block[i][np.isfinite(block[i])]
                    if valid.size:
                        low[i], high[i] = (
                            min(low[i], valid.min()),
                            max(high[i], valid.max()),
                        )
            profile = dict(
                src.profile,
                dtype="uint8",
                nodata=None,
                photometric="RGB",
                BIGTIFF="IF_SAFER",
            )
            with atomic_output(output_path, suffix=".tif") as staged:
                with rasterio.open(staged, "w", **profile) as dst:
                    for _, window in src.block_windows(1):
                        block = src.read(window=window)
                        valid = np.isfinite(block).all(axis=0)
                        rgb = np.zeros(block.shape, np.uint8)
                        for i in range(3):
                            if high[i] > low[i]:
                                scaled = np.nan_to_num(
                                    (block[i] - low[i]) / (high[i] - low[i])
                                )
                                rgb[i] = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
                        dst.write(rgb, window=window)
                        dst.write_mask(valid.astype(np.uint8) * 255, window=window)
                    dst.colorinterp = (
                        rasterio.enums.ColorInterp.red,
                        rasterio.enums.ColorInterp.green,
                        rasterio.enums.ColorInterp.blue,
                    )
                    dst.update_tags(**src.tags())
                    dst.update_tags(
                        TIFFTAG_IMAGEDESCRIPTION=f"RGB visualization using bands {bands}"
                    )
    return str(output_path)
