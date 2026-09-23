Data Repository Maintenance
===========================

This guide is for **GeoTessera data maintainers** who manage the public
Tessera repository on Source Cooperative. End users never need any of this —
the library downloads manifests and tiles automatically.

Repository Layout
-----------------

All Tessera data is served over plain HTTPS from the public Source
Cooperative repository at ``https://data.source.coop/tessera/tessera``,
which is also reachable as an S3-compatible endpoint (bucket ``tessera``,
prefix ``tessera/``). The layout is one tree per media type. The ``npy/``
tree has **one directory per dataset** — a (version, variant) pair — while
the ``landmasks/`` and ``zarr/`` trees are keyed by plain version::

    https://data.source.coop/tessera/tessera/
    ├── npy/
    │   ├── v1/                                  # 1.0 — all variants share this dir
    │   │   ├── manifest.parquet                 # per-dataset tile manifest
    │   │   └── {year}/grid_{lon}_{lat}/grid_{lon}_{lat}{,_scales}.npy
    │   ├── v1.1-cam/                            # 1.1 / cambridge
    │   │   ├── manifest.parquet
    │   │   └── {year}/grid_{lon}_{lat}/...
    │   └── v2-2B-L~beta1/                       # 2.0 / 2B-L~beta1 (beta)
    │       └── {year}/grid_{lon}_{lat}/...
    ├── landmasks/
    │   ├── v1/
    │   │   ├── landmasks.parquet                # per-version landmask registry
    │   │   └── grid_{lon}_{lat}.tiff
    │   ├── v1.1/
    │   │   └── ...
    │   └── v2/
    │       └── ...
    └── zarr/                                    # cloud-native zarr stores
        ├── v1/                                  # 1.0 / vultr
        ├── v1.1/                                # 1.1 / cambridge
        └── v2-2B-L~beta1/

The dataset directory name is the version path plus a ``-<variant>``
suffix; the v1 series predates this scheme, so every 1.0 variant
(including the default ``vultr``) collapses into the bare ``v1/``
directory, and ``cambridge`` abbreviates to ``cam``. The known mapping
lives in ``DATASETS`` in ``geotessera/registry.py``, one row per
``(version, variant)`` with its NPY, Zarr and Icechunk locations. Add a
row, or a location to an existing row, when publishing. The first row of
a version is its default variant; the first row with a format is that
format's default. Downloads identify their dataset by matching the store
URL against this table, so a store missing from it is read but its
downloads are not recorded or checked.

Clients fetch ``npy/{dataset}/manifest.parquet`` and
``landmasks/{version}/landmasks.parquet`` to discover tiles, then download
the tiles themselves on demand. When tiles are added, removed, or replaced
in the repository, the manifests must be regenerated so clients can see the
changes.

.. _regenerating-manifests:

Regenerating Manifests with ``s3scan``
--------------------------------------

The ``geotessera-registry s3scan`` subcommand rebuilds the per-version
manifests by spidering the Source Cooperative repository directly — no
local copy of the embedding data is required, and the scan itself needs
**no credentials** (it issues anonymous, path-style S3 ``ListObjectsV2``
requests over HTTPS).

Scan everything::

    geotessera-registry s3scan s3://tessera/tessera/npy/ \
        --landmasks-uri s3://tessera/tessera/landmasks/ \
        --output ./manifests

This:

1. **Discovers datasets and years** under the prefix: top-level dataset
   directories (``v1/``, ``v1.1-cam/``, ``v2-2B-L~beta1/``, …), then
   four-digit year directories beneath them. A suffixed directory name
   pins the variant (``v1.1-cam`` → ``cambridge``); a bare version
   directory like ``v1/`` carries no variant, so its rows are labelled
   with the version's default variant — override with ``--variant`` if
   needed.
2. **Lists each (dataset, year) in parallel**, sharded into ~360
   integer-longitude prefixes (``grid_-179.`` … ``grid_179.``) so S3's
   1000-keys-per-page limit does not serialise a multi-million-object
   listing. ``--workers`` controls concurrency (default 32).
3. **Pairs** each ``grid_X_Y.npy`` with its ``grid_X_Y_scales.npy`` and
   drops incomplete pairs, taking sizes and modification times straight
   from the listing (no per-object requests).
4. **Writes one parquet per dataset directory**:
   ``./manifests/{v1,v1.1-cam,v2-2B-L~beta1}/manifest.parquet``, plus a
   per-version ``landmasks.parquet``
   (``./manifests/{v1,v1.1,v2}/landmasks.parquet``, scanned from the
   sibling ``landmasks/`` tree given by ``--landmasks-uri``). Rows are
   deduplicated and files are written atomically.

The URI may point at any level, so a single dataset can be rescanned in
isolation (quote the URI — variant suffixes may contain shell
metacharacters like ``~``)::

    geotessera-registry s3scan "s3://tessera/tessera/npy/v1.1-cam/" \
        --landmasks-uri s3://tessera/tessera/landmasks/v1.1/ \
        --output ./manifests

Useful variations:

* ``--no-landmasks`` — skip the landmask TIFF scan entirely.
* Pointing ``--landmasks-uri`` at the same URI as the positional argument
  performs a **landmasks-only** regeneration (the embedding walk is skipped).
* ``--endpoint-url`` — scan a different S3-compatible host (defaults to
  ``https://data.source.coop``).

Uploading the Updated Manifests
-------------------------------

Uploading requires Source Cooperative **write credentials** for the
``tessera/tessera`` repository. Clients fetch the two parquet files from
*different* trees — manifests from the dataset directory under ``npy/``,
landmask registries from the version directory under ``landmasks/`` — so
upload each to its matching location::

    aws s3 cp manifests/v1.1-cam/manifest.parquet \
        s3://tessera/tessera/npy/v1.1-cam/manifest.parquet \
        --endpoint-url https://data.source.coop

    aws s3 cp manifests/v1.1/landmasks.parquet \
        s3://tessera/tessera/landmasks/v1.1/landmasks.parquet \
        --endpoint-url https://data.source.coop

The summary panel printed at the end of ``s3scan`` includes ready-made
per-file ``aws s3 cp`` commands with these destinations filled in, so you
can paste them directly (adding your credentials). Avoid a single
recursive copy of the whole output directory: it would place
``landmasks.parquet`` under the ``npy/`` tree, where clients will never
look for it.

``s3scan`` retries transient listing failures (Cloudflare 503s, timeouts,
connection resets) with exponential backoff — six attempts per request. If
a shard still fails after all retries, the manifest for that dataset is
**not written** (a silently incomplete manifest is worse than a failed
run), the failure is called out in the summary, and the command exits
non-zero; re-run the scan for that dataset.

.. note::

   As of August 2026, the ``v2-2B-L~beta1`` dataset directory has tiles
   for 2017–2025 but **no manifest.parquet yet**, so
   ``GeoTessera(dataset_version="v2")`` will fail until one is generated
   with ``s3scan`` and uploaded to
   ``npy/v2-2B-L~beta1/manifest.parquet`` as above.

Cache Propagation
-----------------

Two layers of caching sit between an uploaded manifest and end users:

* **Cloudflare** fronts ``data.source.coop``, so a freshly uploaded manifest
  may be served from the CDN cache for a while. If the update must be
  visible immediately, purge the cache for the manifest URLs (or wait for
  the cached copies to expire).
* **Client-side caching**: each client keeps the manifest under
  ``~/.cache/geotessera/{dataset}/`` (e.g. ``v1/``, ``v1.1-cam/``) and
  revalidates it with an
  ``If-Modified-Since`` conditional GET keyed on the cached file's
  modification time. Once the CDN serves the new object, clients pick it
  up automatically on their next run — no manual cache clearing needed.

Related Maintainer Commands
---------------------------

``geotessera-registry`` also includes commands for working with **local**
tile trees (used on the machines that generate embeddings, before upload):

* ``geotessera-registry hash <dir>`` — generate per-directory ``SHA256``
  checksum files for ``.npy`` tiles (incremental; only re-hashes changed
  files).
* ``geotessera-registry scan <dir>`` — build ``registry.parquet`` /
  ``landmasks.parquet`` from a local tree by reading those ``SHA256`` files.
* ``geotessera-registry check <dir>`` — validate a local tree's structure,
  detect truncated ``.npy`` files, and optionally re-verify hashes with
  ``--verify-hashes``.
* ``geotessera-registry file-scan`` / ``file-check`` — build per-machine
  inventories and find duplicate tiles across generation machines.

Run ``geotessera-registry --help`` for the complete list, including the
Zarr store construction commands (``zarr-init``, ``zarr-fill``,
``zarr-consolidate``, …) that build the published cloud-native Zarr store.
