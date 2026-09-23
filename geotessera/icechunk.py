"""Read the dClimate TESSERA v1.1 global store and its tile registry.

The store is an Icechunk repository holding one Zarr group per UTM zone
and hemisphere (``30N``, ``30S``), each on its own CRS, with arrays
ordered ``(time, northing, easting, band)``. :class:`IcechunkStore`
presents each zone as the ``utm{NN}`` layout the rest of GeoTessera
reads: one grid in the northern CRS, southern northings negative, and
embeddings ordered ``(time, band, y, x)``. Nothing is copied or
resampled; reads are translated onto the source groups.

Rows at or above the equator come from the ``N`` group and rows below it
from ``S``, since each group extends past the equator to whole shards.
Zone-years missing from a group's ``years_complete`` read as fill.

:class:`TileRegistry` reads the store's Parquet registry, one row per
2048-pixel tile per zone and year, in place of the NPY manifests.

See https://github.com/dClimate/tessera-embeddings/blob/main/docs/global-store.md.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Northing of the equator in a southern UTM CRS.
FALSE_NORTHING = 10_000_000.0

# Public buckets opened anonymously, with their regions.
ANONYMOUS_BUCKETS = {"tessera-embeddings": "us-west-2"}

_SPATIAL = {"northing": "y", "easting": "x"}
_COORDS = {"time", "time_bnds", "northing", "easting", "band", "month"}


def is_icechunk_location(location) -> bool:
    """True if *location* names an Icechunk repository (``*.icechunk``)."""
    return isinstance(location, (str, os.PathLike)) and os.fsdecode(location).rstrip(
        "/"
    ).endswith(".icechunk")


def _s3_options(bucket: str) -> dict:
    region = ANONYMOUS_BUCKETS.get(bucket)
    if region is not None:
        return {"region": region, "anonymous": True}
    return {"region": os.environ.get("AWS_REGION"), "from_env": True}


def open_session(location: str, *, branch: str = "main", snapshot_id=None):
    """Open a read-only session on the Icechunk repository at *location*.

    *location* is an ``s3://bucket/prefix`` URL or a local path. The
    public TESSERA bucket opens anonymously; other buckets take
    credentials from the environment. The repository's own reader
    configuration applies unchanged.
    """
    import icechunk

    location = os.fsdecode(location).rstrip("/")
    parsed = urlparse(location)
    if parsed.scheme == "s3":
        storage = icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=parsed.path.strip("/"),
            **_s3_options(parsed.netloc),
        )
    elif "://" in location:
        raise ValueError(f"Unsupported Icechunk location {location!r}")
    else:
        storage = icechunk.local_filesystem_storage(location)
    repo = icechunk.Repository.open(storage)
    if snapshot_id is not None:
        return repo.readonly_session(snapshot_id=snapshot_id)
    return repo.readonly_session(branch=branch)


def _years(array) -> list[int]:
    values = np.asarray(array[:])
    units = array.attrs.get("units", "")
    if np.issubdtype(values.dtype, np.datetime64):
        return [int(y) + 1970 for y in values.astype("datetime64[Y]").astype(int)]
    if units.startswith("nanoseconds since 1970-01-01"):
        years = values.astype("datetime64[ns]").astype("datetime64[Y]").astype(int)
        return [int(y) + 1970 for y in years]
    if np.issubdtype(values.dtype, np.integer):
        return [int(y) for y in values]
    raise ValueError(f"{array.path}: unsupported time encoding {units!r}")


class _Part:
    """One hemisphere group's placement on the merged zone grid."""

    def __init__(self, group, row0, local0, rows, col0, cols, complete):
        self.group = group
        self.row0 = row0  # merged row of the first kept source row
        self.local0 = local0  # first kept source row
        self.rows = rows
        self.col0 = col0
        self.cols = cols
        self.complete = complete  # per time index: year is complete

    def rows_of(self, merged):
        inside = (merged >= self.row0) & (merged < self.row0 + self.rows)
        return inside, merged - self.row0 + self.local0

    def cols_of(self, merged):
        inside = (merged >= self.col0) & (merged < self.col0 + self.cols)
        return inside, merged - self.col0


def _as_index(key, n: int):
    """``(indices, dropped)`` for one axis of an outer key."""
    if isinstance(key, slice):
        return np.arange(*key.indices(n)), False
    if np.ndim(key) == 0:
        k = int(key)
        if not -n <= k < n:
            raise IndexError(f"index {k} is out of bounds for axis of size {n}")
        return np.array([k % n]), True
    idx = np.asarray(key, dtype=np.intp).ravel()
    if idx.size and (idx.min() < -n or idx.max() >= n):
        raise IndexError(f"index out of bounds for axis of size {n}")
    return idx % n, False


def _compact(idx: np.ndarray):
    """A slice for a contiguous ascending run, else the index array."""
    if len(idx) and idx[-1] - idx[0] == len(idx) - 1 and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


class ZoneArray:
    """One array of a zone, merged across hemispheres.

    Supports basic indexing, ``oindex`` and ``vindex`` like a Zarr array,
    in the output axis order: ``(time, band, y, x)`` for embeddings,
    the source order with ``y, x`` in place of ``northing, easting``
    otherwise.
    """

    def __init__(self, name, parts, shape, order, dims, dtype, fill_value):
        self.name = name
        self._parts = parts
        self.shape = tuple(shape)
        self._order = order  # output axis i reads source axis order[i]
        self.dims = tuple(dims)
        self.dtype = np.dtype(dtype)
        self.fill_value = fill_value
        self.ndim = len(self.shape)
        self._t = self.dims.index("time")
        self._y = self.dims.index("y")
        self._x = self.dims.index("x")
        self.oindex = _Indexer(self._outer)
        self.vindex = _Indexer(self._vectorized)

    def __getitem__(self, key):
        return self._outer(key)

    def _key(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = next(j for j, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1 :]
        if len(key) > self.ndim:
            raise IndexError(f"too many indices for {self.ndim}-d array")
        return key + (slice(None),) * (self.ndim - len(key))

    def _outer(self, key):
        axes = [_as_index(k, n) for k, n in zip(self._key(key), self.shape)]
        idx = [a for a, _ in axes]
        out = np.full([len(a) for a in idx], self.fill_value, dtype=self.dtype)
        for part in self._parts:
            y_in, y_src = part.rows_of(idx[self._y])
            x_in, x_src = part.cols_of(idx[self._x])
            t_in = part.complete[idx[self._t]]
            if not (y_in.any() and x_in.any() and t_in.any()):
                continue
            where = [np.arange(len(a)) for a in idx]
            src = list(idx)
            for axis, inside, local in (
                (self._y, y_in, y_src),
                (self._x, x_in, x_src),
                (self._t, t_in, idx[self._t]),
            ):
                where[axis] = where[axis][inside]
                src[axis] = local[inside]
            source_key = [None] * self.ndim
            for axis, s in enumerate(self._order):
                source_key[s] = _compact(src[axis])
            block = part.group[self.name].oindex[tuple(source_key)]
            out[np.ix_(*where)] = np.transpose(block, self._order)
        dropped = tuple(0 if d else slice(None) for _, d in axes)
        return out[dropped]

    def _vectorized(self, key):
        key = self._key(key)
        if any(isinstance(k, slice) for k in key):
            raise IndexError("vindex takes integer arrays only")
        idx = np.broadcast_arrays(
            *[np.asarray(k, dtype=np.intp) % n for k, n in zip(key, self.shape)]
        )
        out = np.full(idx[0].shape, self.fill_value, dtype=self.dtype)
        for part in self._parts:
            y_in, y_src = part.rows_of(idx[self._y])
            x_in, x_src = part.cols_of(idx[self._x])
            mask = y_in & x_in & part.complete[idx[self._t]]
            if not mask.any():
                continue
            src = list(idx)
            src[self._y], src[self._x] = y_src, x_src
            source_key = [None] * self.ndim
            for axis, s in enumerate(self._order):
                source_key[s] = src[axis][mask]
            out[mask] = part.group[self.name].vindex[tuple(source_key)]
        return out


class _Indexer:
    def __init__(self, read):
        self._read = read

    def __getitem__(self, key):
        return self._read(key)


class _BackendArray(xr.backends.BackendArray):
    """Lazy xarray view of a :class:`ZoneArray`: only indexed windows load."""

    def __init__(self, array: ZoneArray):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key):
        from xarray.core import indexing

        if isinstance(key, indexing.VectorizedIndexer):
            return self.array._vectorized(_points_key(key.tuple, self.shape))
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self.array._outer
        )


def _points_key(key, shape):
    """Expand slices in an xarray vectorized key into broadcast arrays.

    xarray orders the result as the broadcast array dimensions followed by
    each slice's dimension, which a key of broadcast integer arrays gives.
    """
    arrays = [k for k in key if not isinstance(k, slice)]
    slices = len(key) - len(arrays)
    points = np.broadcast_shapes(*[np.shape(a) for a in arrays]) if arrays else ()
    out, i = [], 0
    for k, n in zip(key, shape):
        if isinstance(k, slice):
            index = np.arange(*k.indices(n))
            out.append(
                index.reshape((1,) * (len(points) + i) + (-1,) + (1,) * (slices - i - 1))
            )
            i += 1
        else:
            out.append(np.reshape(k, np.shape(k) + (1,) * slices))
    return tuple(out)


def _transform(group) -> tuple[float, float, int, int]:
    """``(west, top, rows, cols)`` of a group, validated as a 10 m grid."""
    t = [float(v) for v in group.attrs["spatial:transform"]]
    rows, cols = (int(v) for v in group.attrs["spatial:shape"])
    if t[1] or t[3] or t[0] <= 0 or t[4] != -t[0]:
        raise ValueError(f"{group.path}: expected a north-up square-pixel grid")
    return t[2], t[5], rows, cols


class IcechunkStore:
    """A TESSERA Icechunk repository, read one merged UTM zone at a time.

    Args:
        location: ``s3://bucket/prefix`` URL or local path of the
            repository; see :func:`open_session`.
        branch: Branch to read. The default is ``main``.
        snapshot_id: Read this snapshot instead of the branch head.
    """

    def __init__(self, location, *, branch: str = "main", snapshot_id=None):
        import zarr

        self.url = os.fsdecode(location).rstrip("/")
        self.session = open_session(self.url, branch=branch, snapshot_id=snapshot_id)
        self.root = zarr.open_group(
            self.session.store, mode="r", use_consolidated=False
        )
        self.groups = sorted(self.root.group_keys())
        self._zones: dict[int, tuple[xr.Dataset, dict[str, ZoneArray]]] = {}
        first = next((g for g in self.groups if re.fullmatch(r"\d\d[NS]", g)), None)
        self.years = _years(self.root[first]["time"]) if first else []

    def __repr__(self) -> str:
        return f"IcechunkStore({self.url!r})"

    def zones(self) -> list[int]:
        """UTM zones with at least one hemisphere group."""
        return sorted({int(g[:2]) for g in self.groups if re.fullmatch(r"\d\d[NS]", g)})

    def incomplete_years(self) -> dict[str, list[int]]:
        """Years each group holds but does not list in ``years_complete``."""
        out = {}
        for name in self.groups:
            complete = self.root[name].attrs.get("years_complete")
            if complete is not None:
                missing = [y for y in self.years if y not in complete]
                if missing:
                    out[name] = missing
        return out

    def open_zone(self, zone: int) -> xr.Dataset:
        """Zone *zone* as a lazy Dataset on the ``utm{NN}`` layout.

        Raises:
            KeyError: If the repository holds neither hemisphere of *zone*.
        """
        return self._zone(zone)[0]

    def zone_arrays(self, zone: int) -> dict[str, ZoneArray]:
        """Arrays of zone *zone* by name, indexable like Zarr arrays."""
        return self._zone(zone)[1]

    def _zone(self, zone: int):
        if zone not in self._zones:
            self._zones[zone] = self._build(int(zone))
        return self._zones[zone]

    def _build(self, zone: int):
        present = [
            (h, self.root[f"{zone:02d}{h}"])
            for h in "NS"
            if f"{zone:02d}{h}" in self.groups
        ]
        if not present:
            raise KeyError(f"utm{zone:02d}")
        epsg = 32600 + zone

        # Each group's kept rows and extent in the northern CRS. North
        # keeps rows centred above the equator and south those below.
        spans = []
        px = None
        for h, group in present:
            west, top, rows, cols = _transform(group)
            step = float(group.attrs["spatial:transform"][0])
            if px is not None and step != px:
                raise ValueError(f"utm{zone:02d}: hemispheres differ in pixel size")
            px = step
            if h == "S":
                top -= FALSE_NORTHING
            centres = top - (np.arange(rows) + 0.5) * px
            kept = np.flatnonzero(centres > 0 if h == "N" else centres < 0)
            if not len(kept):
                continue
            local0 = int(kept[0])
            spans.append((h, group, top - local0 * px, local0, len(kept), west, cols))
        if not spans:
            raise KeyError(f"utm{zone:02d}")

        top = max(s[2] for s in spans)
        west = min(s[5] for s in spans)
        parts = []
        height = width = 0
        for h, group, edge, local0, rows, x0, cols in spans:
            row0, col0 = (top - edge) / px, (x0 - west) / px
            if abs(row0 - round(row0)) > 1e-6 or abs(col0 - round(col0)) > 1e-6:
                raise ValueError(f"utm{zone:02d}: hemisphere grids are out of phase")
            row0, col0 = round(row0), round(col0)
            years = _years(group["time"])
            if years != self.years:
                raise ValueError(
                    f"{group.path}: years {years} differ from {self.years}"
                )
            complete = group.attrs.get("years_complete", years)
            parts.append(
                _Part(
                    group,
                    row0,
                    local0,
                    rows,
                    col0,
                    cols,
                    np.array([y in complete for y in years]),
                )
            )
            height = max(height, row0 + rows)
            width = max(width, col0 + cols)

        from xarray.core import indexing

        reference = present[0][1]
        arrays: dict[str, ZoneArray] = {}
        variables = {}
        for name in sorted(reference.array_keys()):
            if name in _COORDS:
                continue
            source = reference[name]
            source_dims = list(source.metadata.dimension_names or ())
            if "northing" not in source_dims or "easting" not in source_dims:
                continue
            dims = [_SPATIAL.get(d, d) for d in source_dims]
            if name == "embeddings" and "band" in dims:
                dims = ["time", "band", "y", "x"]
            order = [[_SPATIAL.get(d, d) for d in source_dims].index(d) for d in dims]
            sizes = {"y": height, "x": width}
            shape = [sizes.get(d, source.shape[s]) for d, s in zip(dims, order)]
            array = ZoneArray(
                name, parts, shape, order, dims, source.dtype, source.fill_value
            )
            arrays[name] = array
            attrs = {k: v for k, v in source.attrs.items() if k != "_ARRAY_DIMENSIONS"}
            variables[name] = xr.Variable(
                dims, indexing.LazilyIndexedArray(_BackendArray(array)), attrs
            )

        coords = {
            "time": ("time", np.asarray(self.years, dtype=np.int64)),
            "y": ("y", top - (np.arange(height) + 0.5) * px),
            "x": ("x", west + (np.arange(width) + 0.5) * px),
        }
        if "band" in reference:
            coords["band"] = ("band", np.asarray(reference["band"][:]))
        if "month" in reference:
            coords["month"] = ("month", np.asarray(reference["month"][:]))

        root_attrs = dict(self.root.attrs)
        attrs = {
            "proj:code": f"EPSG:{epsg}",
            "spatial:dimensions": ["y", "x"],
            "spatial:transform": [px, 0.0, west, 0.0, -px, top],
            "spatial:shape": [height, width],
            "spatial:registration": "pixel",
            "geoemb:dimensions": root_attrs.get("geoemb:dimensions", 128),
            # NaN scales mark unembedded pixels, not water.
            "geotessera:mask_source": "source_nodata",
            "geotessera:source_groups": [p.group.basename for p in parts],
            "geotessera:years_incomplete": {
                p.group.basename: [y for y, c in zip(self.years, p.complete) if not c]
                for p in parts
                if not p.complete.all()
            },
        }
        ds = xr.Dataset(variables, coords=coords, attrs=attrs)
        log.debug(
            "icechunk: utm%02d from %s, %d x %d",
            zone,
            [p.group.basename for p in parts],
            height,
            width,
        )
        return ds, arrays


# -- Tile registry ------------------------------------------------------------

_PART_RE = re.compile(r"zone=(\d\d)([NS])/year=(\d{4})/[^/]+\.parquet$")


def _zones_for_lons(west: float, east: float) -> list[int]:
    """UTM zones covering ``[west, east]`` plus one either side."""
    first = math.floor((west + 180.0) / 6.0) % 60 + 1
    last = math.floor((min(east, 179.999999) + 180.0) / 6.0) % 60 + 1
    count = (last - first) % 60 + 1
    return sorted({(first - 2 + i) % 60 + 1 for i in range(count + 2)})


class TileRegistry:
    """The Parquet registry beside a dClimate Icechunk store.

    One row per 2048-pixel tile per zone and year, with its WGS84 bounds,
    whether it was embedded and how many pixels the depth rule refused.
    Parts are named by run; a refilled zone-year keeps both, and
    :meth:`tiles` keeps the latest ``assembled_at`` per tile.

    Args:
        url: ``s3://`` URL of the ``parts/`` prefix.
        cache_dir: Keep downloaded parts here. Parts are immutable, so a
            cached part is never refetched. The default is the
            GeoTessera cache directory.
    """

    def __init__(self, url: str, cache_dir: str | Path | None = None):
        self.url = url.rstrip("/")
        if cache_dir is None:
            base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            cache_dir = base / "geotessera"
        parsed = urlparse(self.url)
        self._bucket, self._prefix = parsed.netloc, parsed.path.strip("/")
        self._cache = Path(cache_dir) / "tile-registry" / self._bucket / self._prefix
        self._parts: list[tuple[str, str, int, str]] | None = None

    def _store(self):
        from obstore.store import S3Store

        options = _s3_options(self._bucket)
        config = {"region": options["region"]} if options["region"] else {}
        return S3Store(
            self._bucket,
            prefix=self._prefix,
            skip_signature=options.get("anonymous", False),
            config=config,
        )

    def parts(self) -> list[tuple[str, str, int, str]]:
        """``(zone, hemisphere, year, path)`` for every registry part."""
        if self._parts is None:
            import obstore

            found = []
            for batch in obstore.list(self._store()):
                for meta in batch:
                    m = _PART_RE.search(meta["path"])
                    if m:
                        found.append(
                            (m.group(1), m.group(2), int(m.group(3)), meta["path"])
                        )
            self._parts = sorted(found)
        return self._parts

    def years(self) -> list[int]:
        """Years with at least one registry part."""
        return sorted({year for _, _, year, _ in self.parts()})

    def _read(self, path: str):
        import pandas as pd

        local = self._cache / path
        if not local.exists():
            import obstore

            from .remote import atomic_output

            data = obstore.get(self._store(), path).bytes()
            local.parent.mkdir(parents=True, exist_ok=True)
            with atomic_output(local, suffix=".parquet") as temporary:
                Path(temporary).write_bytes(bytes(data))
        return pd.read_parquet(local)

    def tiles(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        year: int | None = None,
        progress_callback=None,
    ):
        """Registry rows intersecting *bbox* in *year*, latest run only.

        Args:
            bbox: WGS84 ``(west, south, east, north)``. The default is
                the whole registry, about 140 MB.
            year: Select one year. The default is every year.
            progress_callback: Call ``callback(done, total, status)``
                after each part is read.

        Returns:
            A DataFrame with the registry columns plus ``zone``
            (``"30N"``) and ``year``. A row crossing the antimeridian has
            ``bbox_west > bbox_east``.
        """
        import pandas as pd

        selected = self.parts()
        if year is not None:
            selected = [p for p in selected if p[2] == year]
        if bbox is not None:
            west, south, east, north = bbox
            zones = {f"{z:02d}" for z in _zones_for_lons(west, east)}
            # Southern groups reach past the equator, so keep S near it.
            hemispheres = {
                h for h, keep in (("N", north >= 0), ("S", south < 1)) if keep
            }
            selected = [p for p in selected if p[0] in zones and p[1] in hemispheres]

        frames = []
        for i, (zone, hemisphere, part_year, path) in enumerate(selected):
            frame = self._read(path)
            frame["zone"] = zone + hemisphere
            frame["year"] = part_year
            frames.append(frame)
            if progress_callback:
                progress_callback(
                    i + 1, len(selected), f"Reading {zone}{hemisphere} {part_year}"
                )
        if not frames:
            return pd.DataFrame(columns=["zone", "year", "tile"])
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("assembled_at").drop_duplicates(
            ["zone", "year", "tile"], keep="last"
        )
        if bbox is not None:
            wraps = df["bbox_west"] > df["bbox_east"]
            lon_hit = np.where(
                wraps,
                (df["bbox_west"] <= east) | (df["bbox_east"] >= west),
                (df["bbox_west"] <= east) & (df["bbox_east"] >= west),
            )
            lat_hit = (df["bbox_south"] <= north) & (df["bbox_north"] >= south)
            df = df[lon_hit & lat_hit]
        return df.sort_values(["year", "zone", "tile"]).reset_index(drop=True)
