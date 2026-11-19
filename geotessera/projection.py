"""Shared, reproducible PCA fitting and windowed visualization."""

from pathlib import Path
import logging
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ._typing import ProgressCallback
from .remote import atomic_output


def blocks(source):
    if hasattr(source, "iter_blocks"):
        yield from source.iter_blocks()
    else:
        data = source["data"]
        for top in range(0, data.shape[0], 128):
            yield top, data[top : top + 128]


def metadata(source):
    if isinstance(source, dict):
        return {key: source[key] for key in ("height", "width", "crs", "transform")}
    return {
        key: getattr(source, key) for key in ("height", "width", "crs", "transform")
    }


def fit_projection(sources, n_components=3, standardize=True, max_samples=100_000):
    """Fit one PCA on a deterministic uniform sample of valid pixels.

    Random priorities implement a bounded reservoir across all input blocks;
    large rasters do not need to fit in RAM. Invalid rows never enter the fit.
    """
    rng = np.random.default_rng(0)
    sample, priorities = None, np.empty(0)
    for source in sources:
        for _, block in blocks(source):
            pixels = block.reshape(-1, block.shape[-1])
            pixels = pixels[np.isfinite(pixels).all(axis=1)]
            if not len(pixels):
                continue
            keys = rng.random(len(pixels))
            if len(pixels) > max_samples:
                take = np.argpartition(keys, max_samples - 1)[:max_samples]
                pixels, keys = pixels[take], keys[take]
            if sample is not None:
                pixels, keys = (
                    np.concatenate((sample, pixels)),
                    np.concatenate((priorities, keys)),
                )
            if len(pixels) > max_samples:
                take = np.argpartition(keys, max_samples - 1)[:max_samples]
                pixels, keys = pixels[take], keys[take]
            sample, priorities = pixels, keys
    if sample is None or not 1 <= n_components <= min(sample.shape):
        raise ValueError(
            "Not enough valid pixels/bands for the requested PCA components"
        )
    logging.getLogger(__name__).info("PCA fit sample: %s", sample.shape)
    scaler = StandardScaler().fit(sample) if standardize else None
    scaled = scaler.transform(sample) if scaler is not None else sample
    pca = PCA(n_components=n_components, random_state=0).fit(scaled)
    return scaler, pca, pca.transform(scaled)


def transform_block(block, scaler, pca):
    flat = block.reshape(-1, block.shape[-1])
    valid = np.isfinite(flat).all(axis=1)
    result = np.full((len(flat), pca.n_components), np.nan, np.float32)
    if valid.any():
        pixels = flat[valid]
        result[valid] = pca.transform(
            scaler.transform(pixels) if scaler is not None else pixels
        )
    return result.reshape(*block.shape[:2], pca.n_components)


def write_pca_tiles(
    sources,
    output_dir,
    n_components=3,
    standardize=True,
    balance_method="percentile",
    percentile_range=(2, 98),
    compress="lzw",
    progress_callback: ProgressCallback | None = None,
    local_scaling=False,
):
    """Render inputs through one sampled PCA and one shared color mapping."""
    if balance_method not in ("percentile", "histogram", "adaptive"):
        raise ValueError(f"Unknown balance method: {balance_method}")
    if not 0 <= percentile_range[0] < percentile_range[1] <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    sources = list(sources)
    scaler, pca, sample = fit_projection(sources, n_components, standardize)
    if balance_method == "adaptive":
        low, high = (
            sample.mean(axis=0) - 2.5 * sample.std(axis=0),
            sample.mean(axis=0) + 2.5 * sample.std(axis=0),
        )
    else:
        low, high = np.percentile(sample, percentile_range, axis=0)
    cdfs = None
    if balance_method == "histogram":
        from skimage.exposure import cumulative_distribution

        cdfs = [cumulative_distribution(sample[:, i]) for i in range(n_components)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, source in enumerate(sources):
        if local_scaling:
            low, high = np.full(n_components, np.inf), np.full(n_components, -np.inf)
            for _, block in blocks(source):
                values = transform_block(block, scaler, pca).reshape(-1, n_components)
                values = values[np.isfinite(values).all(axis=1)]
                if values.size:
                    low, high = (
                        np.minimum(low, values.min(axis=0)),
                        np.maximum(high, values.max(axis=0)),
                    )
        path = output_dir / f"pca_{i}.tif"
        info = metadata(source)
        with atomic_output(path, suffix=".tif") as temporary:
            with rasterio.open(
                temporary,
                "w",
                driver="GTiff",
                **info,
                count=n_components,
                dtype="uint8",
                compress=compress,
                tiled=True,
                BIGTIFF="IF_SAFER",
                photometric="MINISBLACK",
            ) as dst:
                for top, block in blocks(source):
                    projected = transform_block(block, scaler, pca)
                    valid = np.isfinite(projected).all(axis=-1)
                    rgb = np.zeros(projected.shape, np.uint8)
                    for band in range(n_components):
                        if cdfs is not None:
                            cdf, bins = cdfs[band]
                            scaled = np.interp(projected[..., band], bins, cdf)
                        elif high[band] > low[band]:
                            scaled = (projected[..., band] - low[band]) / (
                                high[band] - low[band]
                            )
                        else:
                            scaled = np.zeros(valid.shape)
                        rgb[..., band] = (
                            np.clip(np.nan_to_num(scaled), 0, 1) * 255
                        ).astype(np.uint8)
                    window = rasterio.windows.Window(
                        0, top, block.shape[1], block.shape[0]
                    )
                    dst.write(rgb.transpose(2, 0, 1), window=window)
                    dst.write_mask(valid.astype(np.uint8) * 255, window=window)
                if n_components == 3:
                    dst.colorinterp = (
                        rasterio.enums.ColorInterp.red,
                        rasterio.enums.ColorInterp.green,
                        rasterio.enums.ColorInterp.blue,
                    )
                dst.update_tags(
                    PCA_COMPONENTS=str(n_components),
                    PCA_STANDARDIZED=str(standardize),
                    PCA_EXPLAINED_VARIANCE=str(pca.explained_variance_ratio_.tolist()),
                    PCA_TOTAL_VARIANCE=str(pca.explained_variance_ratio_.sum()),
                    PCA_BALANCE_METHOD=balance_method,
                    PCA_MAX_SAMPLES="100000",
                )
                if hasattr(source, "year"):
                    dst.update_tags(
                        TESSERA_YEAR=str(source.year),
                        TESSERA_TILE_LON=str(source.lon),
                        TESSERA_TILE_LAT=str(source.lat),
                    )
        files.append(str(path))
        if progress_callback:
            progress_callback(i + 1, len(sources), "Writing PCA tiles")
    return files
