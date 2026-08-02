"""Location-transparent I/O for local paths and remote object stores.

Every path in the Zarr build pipeline — the tile inputs (NPY embeddings,
scales, landmask GeoTIFFs), the output store, and the store's own tracking
parquet — can be either a local filesystem path or an fsspec URL such as
``s3://bucket/prefix``.  This module is the single place that knows the
difference.

Two design points matter for the remote case:

* **Byte-range reads, no scratch disk.**  A ``.npy`` file is a short header
  followed by a flat C-ordered buffer, so the rows a shard needs are one
  contiguous range.  :func:`read_npy_window` parses the header (one small
  ranged GET, memoised per process) and then fetches exactly those rows.
  A 158 MB tile costs only the bytes actually written into the shard.

* **Credentials never go through argv by default.**  :func:`build_storage_options`
  assembles an fsspec option dict from explicit arguments, falling back to
  the standard ``AWS_*`` environment variables, so a sweep can run with keys
  supplied by the environment or an instance profile.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Protocols we can read/write through fsspec. Anything else without a
# scheme is treated as a local filesystem path.
_URL_MARKER = "://"


# Chatty at INFO and of no interest to a user watching a fill. botocore in
# particular logs "Found credentials in shared credentials file" every time
# a client is built — once per worker process, which shreds a progress bar
# the workers share a terminal with.
_NOISY_LOGGERS = (
    "botocore",
    "boto3",
    "aiobotocore",
    "s3fs",
    "urllib3",
    "aiohttp",
)


def quieten_dependency_logging(level: int = logging.WARNING) -> None:
    """Raise the log level of the object-store libraries.

    Call in the parent before a progress bar starts, and again in each worker
    process, since a worker that did not inherit the parent's configuration
    would otherwise start logging into the shared terminal.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def reset_after_fork() -> None:
    """Drop inherited fsspec/asyncio state in a freshly started worker.

    ``s3fs`` drives its client from a background event-loop thread and
    ``fsspec`` caches both the loop and its filesystem instances globally. A
    forked child inherits those objects but not the thread running the loop,
    so the next call deadlocks: the worker's main thread waits on a future
    the loop will never service (main in ``futex_do_wait``, loop thread idle
    in ``ep_poll``). Workers are started with "spawn" to avoid this outright;
    this is the belt-and-braces reset for anything that still leaks in.
    """
    try:
        import fsspec.asyn

        fsspec.asyn.reset_lock()
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(f"Could not reset fsspec event loop: {e}")

    try:
        import fsspec

        fsspec.AbstractFileSystem.clear_instance_cache()
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(f"Could not clear fsspec instance cache: {e}")

    _filesystem_cached.cache_clear()


def die_with_parent() -> None:
    """Ask the kernel to kill this process when its parent goes away.

    Without it, a fill that is killed outright leaves its workers running:
    each holds gigabytes and keeps writing, and they accumulate across runs
    until the machine is out of memory. Linux only; a no-op elsewhere.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL
        )
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(f"Could not set parent-death signal: {e}")


def is_url(loc: str | Path) -> bool:
    """True if *loc* is an fsspec URL rather than a local filesystem path."""
    return _URL_MARKER in str(loc)


def protocol_of(loc: str | Path) -> str:
    """Return the fsspec protocol for *loc* (``"file"`` for local paths)."""
    s = str(loc)
    if _URL_MARKER not in s:
        return "file"
    return s.split(_URL_MARKER, 1)[0]


def join(base: str | Path, *parts: str) -> str:
    """Join path components, using ``/`` for URLs and os.sep for local paths.

    Returns a ``str`` in both cases so callers can pass the result straight
    to zarr, fsspec, or ``open()``.
    """
    if is_url(base):
        out = str(base).rstrip("/")
        for p in parts:
            p = str(p).strip("/")
            if p:
                out = f"{out}/{p}"
        return out
    return str(Path(base).joinpath(*[str(p) for p in parts]))


# Canned ACLs S3 accepts on an object (PutObject's x-amz-acl). Validated up
# front so a typo fails before a long fill rather than on the first PUT.
OBJECT_ACLS = (
    "private",
    "public-read",
    "public-read-write",
    "authenticated-read",
    "aws-exec-read",
    "bucket-owner-read",
    "bucket-owner-full-control",
)


def build_storage_options(
    endpoint_url: Optional[str] = None,
    anon: bool = False,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    requester_pays: bool = False,
    acl: Optional[str] = None,
    path_style: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble an fsspec storage-options dict for an S3-compatible endpoint.

    Unset arguments fall back to the conventional environment variables
    (``AWS_ENDPOINT_URL``, ``AWS_DEFAULT_REGION``/``AWS_REGION``,
    ``AWS_PROFILE``), which is how credentials should normally be supplied
    so keys never appear in a process listing.  Access keys themselves are
    left entirely to botocore's own resolution chain (environment, shared
    config, instance metadata).

    Args:
        acl: Canned ACL to stamp on every object written, e.g.
            ``bucket-owner-full-control`` for a bucket owned by another
            account.  s3fs filters it per operation, so reads are unaffected.
        path_style: Address buckets as ``endpoint/bucket/key`` rather than
            ``bucket.endpoint/key``.  Most S3-compatible servers (MinIO,
            Ceph, and anything on a hostname without wildcard DNS) need
            this; botocore's default would try to resolve
            ``<bucket>.<your-endpoint>``.

    Returns ``None`` when nothing needs configuring, so callers can pass the
    result through to zarr unchanged.
    """
    options: Dict[str, Any] = {}

    endpoint_url = endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or None
    if endpoint_url:
        options["endpoint_url"] = endpoint_url

    if anon:
        options["anon"] = True

    region = (
        region
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or None
    )
    if region:
        options["client_kwargs"] = {"region_name": region}

    profile = profile or os.environ.get("AWS_PROFILE") or None
    if profile and not anon:
        options["profile"] = profile

    if requester_pays:
        options["requester_pays"] = True

    if path_style:
        options["config_kwargs"] = {"s3": {"addressing_style": "path"}}

    if acl:
        if acl not in OBJECT_ACLS:
            raise ValueError(
                f"Unknown canned ACL {acl!r}. Expected one of: "
                + ", ".join(OBJECT_ACLS)
            )
        options["s3_additional_kwargs"] = {"ACL": acl}

    if extra:
        options.update(extra)

    return options or None


def _options_key(storage_options: Optional[Dict[str, Any]]) -> str:
    """Stable hashable key for a storage-options dict."""
    if not storage_options:
        return ""
    return json.dumps(storage_options, sort_keys=True, default=str)


@lru_cache(maxsize=8)
def _filesystem_cached(protocol: str, options_key: str):
    """Build (and memoise) one filesystem per protocol/options pair.

    Worker processes call this on every tile read, so the cache keeps a
    single connection pool per process rather than one per file.
    """
    import fsspec

    options = json.loads(options_key) if options_key else {}
    if protocol in ("file", "local"):
        # Object stores have no directories to create; a file:// store still
        # needs them, and fsspec will not make them unless asked.
        options.setdefault("auto_mkdir", True)
    if protocol in ("http", "https"):
        # The Source Cooperative CDN rejects some default user agents.
        from . import __version__

        options.setdefault(
            "client_kwargs", {"headers": {"User-Agent": f"geotessera/{__version__}"}}
        )
    try:
        return fsspec.filesystem(protocol, **options)
    except ImportError as e:
        hint = (
            "install the s3 extra: pip install 'geotessera[s3]'"
            if protocol == "s3"
            else f"install the fsspec backend for {protocol}://"
        )
        raise ImportError(
            f"{protocol}:// locations need an fsspec backend — {hint} ({e})"
        ) from e


def get_fs(loc: str | Path, storage_options: Optional[Dict[str, Any]] = None):
    """Return an fsspec filesystem for *loc*, or ``None`` for local paths."""
    if not is_url(loc):
        return None
    return _filesystem_cached(protocol_of(loc), _options_key(storage_options))


def exists(
    loc: str | Path,
    storage_options: Optional[Dict[str, Any]] = None,
    on_denied: Optional[bool] = None,
) -> bool:
    """True if *loc* exists, locally or remotely.

    Args:
        on_denied: What to answer when the store refuses to say. S3 returns
            403, not 404, for a key that does not exist when the caller lacks
            ``s3:ListBucket`` on the prefix — it will not reveal whether the
            object is there. Write-scoped credentials hit this constantly, so
            probes that can treat "don't know" as "absent" pass ``False``
            here. ``None`` (the default) re-raises the PermissionError.
    """
    fs = get_fs(loc, storage_options)
    if fs is None:
        return Path(loc).exists()
    try:
        return bool(fs.exists(str(loc)))
    except PermissionError:
        if on_denied is None:
            raise
        logger.warning(
            f"Permission denied probing {loc}; assuming "
            f"{'present' if on_denied else 'absent'}. This usually means the "
            f"credentials lack s3:ListBucket on the prefix, which makes a "
            f"missing key answer 403 instead of 404."
        )
        return on_denied


def read_bytes(
    loc: str | Path, storage_options: Optional[Dict[str, Any]] = None
) -> bytes:
    """Read a whole object/file into memory."""
    fs = get_fs(loc, storage_options)
    if fs is None:
        return Path(loc).read_bytes()
    return fs.cat_file(str(loc))


def write_bytes(
    loc: str | Path,
    data: bytes,
    storage_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Write *data* to *loc*, creating parent directories for local paths."""
    fs = get_fs(loc, storage_options)
    if fs is None:
        path = Path(loc)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    with fs.open(str(loc), "wb") as f:
        f.write(data)


def remove(loc: str | Path, storage_options: Optional[Dict[str, Any]] = None) -> None:
    """Delete *loc*, ignoring a missing target."""
    fs = get_fs(loc, storage_options)
    try:
        if fs is None:
            Path(loc).unlink(missing_ok=True)
        else:
            fs.rm_file(str(loc))
    except FileNotFoundError:
        pass
    except Exception as e:  # pragma: no cover - best-effort cleanup
        logger.warning(f"Could not remove {loc}: {e}")


def listdir(
    loc: str | Path,
    storage_options: Optional[Dict[str, Any]] = None,
    on_denied: Optional[list] = None,
) -> list[str]:
    """List the immediate children of a directory/prefix, as full locations.

    Returns an empty list when the directory does not exist.

    Args:
        on_denied: Value to return when listing is refused (no
            ``s3:ListBucket``). ``None`` re-raises.
    """
    fs = get_fs(loc, storage_options)
    if fs is None:
        path = Path(loc)
        if not path.is_dir():
            return []
        return [str(p) for p in sorted(path.iterdir())]

    try:
        entries = sorted(fs.ls(str(loc), detail=False))
    except FileNotFoundError:
        return []
    except PermissionError:
        if on_denied is None:
            raise
        logger.warning(
            f"Permission denied listing {loc}; treating it as empty. The "
            f"credentials need s3:ListBucket on this prefix."
        )
        return on_denied

    protocol = protocol_of(loc)
    out = []
    for entry in entries:
        # fsspec strips the protocol from listing results; restore it so the
        # entries are usable as standalone locations. Both shapes round-trip:
        # "bucket/key" -> "s3://bucket/key", "/abs/path" -> "file:///abs/path".
        out.append(entry if is_url(entry) else f"{protocol}://{entry}")
    return out


def download(
    loc: str | Path,
    dest: Path,
    storage_options: Optional[Dict[str, Any]] = None,
) -> Path:
    """Copy a remote object to a local file, writing atomically."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_bytes(read_bytes(loc, storage_options))
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# NPY windowed reads
# ---------------------------------------------------------------------------

# Per-process memo of parsed .npy headers: loc -> (shape, dtype, data_offset).
# Each tile is read once per shard it overlaps, so this saves a small ranged
# GET on the second and subsequent reads.
_npy_header_cache: Dict[str, Tuple[Tuple[int, ...], np.dtype, int]] = {}

# Enough for any 1.0/2.0 header; the spec pads them to a 64-byte boundary and
# real Tessera tiles use 128 bytes.
_HEADER_PROBE_BYTES = 4096


def _parse_npy_header(head: bytes) -> Tuple[Tuple[int, ...], np.dtype, int]:
    """Parse a .npy header prefix into (shape, dtype, data_offset).

    Raises ValueError for Fortran-ordered arrays, whose rows are not
    contiguous and so cannot be range-read a row band at a time.
    """
    f = io.BytesIO(head)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"Unsupported .npy format version {version}")
    if fortran_order:
        raise ValueError("Fortran-ordered .npy files are not supported")
    return tuple(shape), dtype, f.tell()


def _npy_header(loc: str, storage_options: Optional[Dict[str, Any]]):
    """Fetch and memoise the header of a remote .npy file."""
    cached = _npy_header_cache.get(loc)
    if cached is not None:
        return cached

    fs = get_fs(loc, storage_options)
    # Keyword arguments are mandatory: S3FileSystem._cat_file takes version_id
    # as its second positional parameter, so a positional (start, end) would
    # silently read from the wrong offset.
    head = fs.cat_file(loc, start=0, end=_HEADER_PROBE_BYTES)
    parsed = _parse_npy_header(head)
    _npy_header_cache[loc] = parsed
    return parsed


def read_npy_window(
    loc: str | Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    storage_options: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Read ``arr[row_start:row_end, col_start:col_end, ...]`` from a .npy file.

    Local files are memory-mapped exactly as before.  Remote files are served
    by a single ranged GET covering the requested rows: rows are contiguous in
    a C-ordered array, so the read is one range even though the column window
    is strided.  Full-width windows therefore transfer no wasted bytes, and a
    narrow column window costs at most the row band it sits in.

    The returned array may be a read-only view of the fetched buffer; callers
    that mutate it must copy first.
    """
    fs = get_fs(loc, storage_options)
    if fs is None:
        arr = np.load(str(loc), mmap_mode="r")
        return arr[row_start:row_end, col_start:col_end, ...]

    loc = str(loc)
    shape, dtype, offset = _npy_header(loc, storage_options)
    row_end = min(row_end, shape[0])
    if row_end <= row_start:
        return np.empty((0, max(col_end - col_start, 0)) + shape[2:], dtype=dtype)

    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    start = offset + row_start * row_bytes
    end = offset + row_end * row_bytes
    buf = fs.cat_file(loc, start=start, end=end)
    expected = (row_end - row_start) * row_bytes
    if len(buf) != expected:
        raise OSError(
            f"Short read from {loc}: got {len(buf)} of {expected} bytes "
            f"for rows {row_start}:{row_end}"
        )

    band = np.frombuffer(buf, dtype=dtype).reshape((row_end - row_start,) + shape[1:])
    return band[:, col_start:col_end, ...]


def read_tiff_window(
    loc: str | Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    band: int = 1,
    storage_options: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Read a window from a single-band GeoTIFF, locally or remotely.

    Landmask TIFFs are small (tens of KB compressed), so the remote path
    fetches the whole object once and opens it from memory rather than
    paying GDAL's multi-request /vsicurl dance.
    """
    import rasterio
    from rasterio.windows import Window

    window = Window.from_slices((row_start, row_end), (col_start, col_end))

    fs = get_fs(loc, storage_options)
    if fs is None:
        with rasterio.open(str(loc)) as src:
            return src.read(band, window=window)

    from rasterio.io import MemoryFile

    data = fs.cat_file(str(loc))
    with MemoryFile(data) as memfile, memfile.open() as src:
        return src.read(band, window=window)


# ---------------------------------------------------------------------------
# Local atomic writes
# ---------------------------------------------------------------------------
# Kept alongside the remote helpers because it answers the same question for
# the local case: how a caller writes a file without anyone observing a partial
# one. registry.py and registry_cli.py both build parquet through it.


@contextmanager
def atomic_output(dest: Optional[str | Path], suffix: str = "") -> Iterator[Path]:
    """Yield a temporary path that becomes *dest* on success.

    Any exception, ``KeyboardInterrupt`` included, removes the temporary file
    instead, so a partial write is never observed at *dest* nor left behind as
    a stray ``.*_tmp_*``. The temporary file is created in *dest*'s directory,
    made if needed, to keep the final rename on one filesystem. With
    ``dest=None`` it goes to the system temp directory and is kept on success
    for the caller to own.
    """
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            dir=dest.parent,
            prefix=f".{dest.name}_tmp_",
            suffix=suffix,
            delete=False,
        )
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    path = Path(tmp.name)
    try:
        yield path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    else:
        if dest is not None:
            # replace(), not rename(): rename() raises FileExistsError on
            # Windows when the destination already exists.
            path.replace(dest)
