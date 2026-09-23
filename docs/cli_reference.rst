CLI Reference
=============

``geotessera`` reads, exports, and displays Tessera embeddings.

Synopsis
--------

::

    geotessera COMMAND [OPTIONS]
    geotessera COMMAND --help

Commands
--------

download
~~~~~~~~

::

    geotessera download [OPTIONS] --output DIRECTORY

Export a region as GeoTIFF files or download individual embedding tiles.
Specify one region selector from :ref:`cli-regions`.

``--source auto|zarr|tiles``
    Select the data source. The default, ``auto``, uses Zarr for TIFF
    output and individual tiles for NPY output or ``--registry-dir``.
    Zarr supports TIFF output only. Tiles are deprecated and will be
    removed.

``-o, --output DIRECTORY``
    Write files to this directory. This option is required unless
    ``--dry-run`` is set.

``-f, --format tiff|npy``
    Select the output format. The default is ``tiff``. NPY downloads
    contain quantized embeddings, scale arrays, and landmask GeoTIFFs.
    NPY is deprecated and will be removed.

``--year INTEGER``
    Select the embedding year. The default is 2024.

``--bands LIST``
    Select comma-separated, zero-based embedding bands in the given order.
    The default is all bands at the selected depth. This option applies
    to TIFF output.

``--depth INTEGER``
    Read a published matryoshka prefix from Zarr. The default is the full
    embedding. Band indices refer to the selected prefix. A store without
    the requested depth reports an error.

``--compress TEXT``
    Set the GeoTIFF compression method. The default is ``lzw``.

``--dry-run``
    Read metadata and report the size without downloading embeddings.
    Zarr mode reports uncompressed output bytes. NPY mode totals the source embedding, scale,
    and landmask file sizes. Tile TIFF mode estimates output size from the
    source metadata. Compressed GeoTIFF sizes and Zarr transfer sizes depend
    on the data and chunk layout.

``--list-files``
    List individual tile output files with their sizes. Zarr exports
    always print the output paths.

``-v, --verbose``
    Print additional tile download details.

Dataset, store, and cache options are described in :ref:`cli-data-source`.

Output and repeated runs
^^^^^^^^^^^^^^^^^^^^^^^^

Zarr exports write ``tessera_YEAR_utmNN.tif`` for each intersecting UTM zone.
Each file contains dequantized float32 values on the native grid, with NaN
nodata, band descriptions, and source metadata. Output windows enclose the
requested bounds within the available zone grid. Large outputs use BigTIFF
when needed.

Each destination is replaced only after the new Zarr export file is complete.
Rerunning repeats the export, including zones completed before a failure.
Use a separate output directory for each region.

Individual tile downloads use the
``global_0.1_degree_representation/YEAR/grid_LON_LAT/`` layout. NPY scale
files accompany the embedding arrays, and landmasks are stored under
``global_0.1_degree_tiff_all/``.

Both sources record the dataset version and variant in the output
directory's ``tessera_metadata.json``, and GeoTIFFs also record them in
their ``TESSERA_DATASET_VERSION`` and ``TESSERA_DATASET_VARIANT`` tags.
Downloading into a directory that holds another dataset is an error. A
store given by ``--store-url`` that is not a published dataset is not
recorded or checked.

Tile downloads skip existing files. Rerun the same command to continue an
interrupted download. Use a new output directory when changing the
exported bands, since existing tiles are reused by filename.

Examples
^^^^^^^^

Export three embedding bands from Zarr::

    geotessera download --bbox '-3.0,53.4,-2.9,53.5' \
        --year 2024 --bands 0,1,2 --output region/

Export from a local Zarr store::

    geotessera download --store-url /data/tessera.zarr \
        --bbox '-3.0,53.4,-2.9,53.5' --year 2024 --output region/

Estimate an export from a published prefix::

    geotessera download --dataset-version v2 --depth 16 \
        --bbox '-3.0,53.4,-2.9,53.5' --year 2024 --dry-run

Download an individual v1.1 tile, which exists only for the ``cambridge``
variant, as GeoTIFF or raw NPY files::

    geotessera download --source tiles --dataset-variant cambridge \
        --tile '0.17,52.23' --output tiles/
    geotessera download --format npy --dataset-variant cambridge \
        --tile '0.17,52.23' --output arrays/

visualize
~~~~~~~~~

::

    geotessera visualize INPUT_PATH OUTPUT_FILE [OPTIONS]

Create a PCA mosaic from a GeoTIFF file or a directory of GeoTIFF or NPY
tiles. NPY input must include its scale arrays and landmask GeoTIFFs.

The command fits one PCA model to a reproducible sample of up to 100,000
valid pixels across the inputs. It applies the same model and color scale
to every input and processes the rasters in windows. Pixels with missing
values remain masked. Colors may differ from earlier releases that fitted
PCA separately for each tile or used every pixel.

The output contains the first three components as a display-scaled uint8
RGB image, or fewer bands if fewer components are requested. Additional
components contribute to the variance metadata but are not written to the
mosaic. Use the original embeddings for analysis.

``--n-components INTEGER``
    Set the number of PCA components to fit. The default is 3. The count must
    not exceed the number of input bands or valid sampled pixels.

``--crs TEXT``
    Set the output coordinate reference system. The default is
    ``EPSG:3857``.

``--balance histogram|percentile|adaptive``
    Set the color scaling method. The default, ``histogram``, equalizes
    the sampled distribution. ``percentile`` clips to the selected
    percentiles. ``adaptive`` uses the sampled mean and standard deviation.

``--percentile-low FLOAT``, ``--percentile-high FLOAT``
    Set the lower and upper bounds for percentile scaling. The defaults
    are 2 and 98. Use these options with ``--balance percentile``.

Create a PCA image and display it as a web map::

    geotessera visualize region/ pca.tif
    geotessera webmap pca.tif --output map/ --serve

webmap
~~~~~~

::

    geotessera webmap [RGB_MOSAIC] [OPTIONS]

Create web tiles and ``viewer.html`` from a three-band RGB GeoTIFF.
Omit ``RGB_MOSAIC`` and specify one region selector to read directly from
Zarr. Region mode maps three embedding bands to RGB using a common min/max
scale across the region. Use ``visualize`` first to create a PCA map.

Tile generation requires the GDAL command-line tools on ``PATH``. The map
uses Web Mercator (``EPSG:3857``). Serve the output directory over HTTP to
view the map.

``-o, --output DIRECTORY``
    Write the viewer, tiles, and intermediate rasters to this directory.
    The default is ``tessera_webmap`` for a region, or
    ``RGB_MOSAIC_STEM_webmap`` for a local image.

``--bands LIST``
    Select exactly three comma-separated, zero-based embedding bands for
    a Zarr region. The default is ``0,1,2``.

``--year INTEGER``, ``--depth INTEGER``
    Select the year and published embedding prefix for a Zarr region.
    The defaults are 2024 and the full embedding.

``--region-file PATH_OR_URL``
    Read a vector boundary and overlay it on the map. In region mode,
    this also selects the bounding box to stream. With ``RGB_MOSAIC``,
    it adds an overlay without changing the image extent.

``--min-zoom INTEGER``, ``--max-zoom INTEGER``
    Set the range of zoom levels to generate, inclusive. The defaults
    are 8 and 15. Values must satisfy ``0 <= min <= max <= 24``.

``--initial-zoom INTEGER``
    Set the viewer's initial zoom level. The default is 10.

``--force/--no-force``
    Regenerate the streamed RGB mosaic and web tiles. The default is
    ``--no-force``. Use ``--force`` when data changes at the same store URL.

``--serve/--no-serve``
    Start the web server and open a browser after generating the map.
    The default is ``--no-serve``. With ``--serve``, an occupied port
    causes an error before processing starts.

``-p, --port INTEGER``
    Set the web server port. The default is 8000.

``--use-gdal-raster/--use-gdal2tiles``
    Select the GDAL tile generator. The default is ``--use-gdal2tiles``.
    ``--use-gdal-raster`` requires a GDAL installation with ``gdal raster tile``.

Region mode also accepts the options in :ref:`cli-regions` and
:ref:`cli-data-source`, except ``--registry-dir``.

Repeated runs
^^^^^^^^^^^^^

Matching completed RGB mosaics and tiles are reused. Changing the zoom
range rebuilds tiles without reading the embeddings again. Interrupted
tile generation restarts from the completed RGB mosaic; interrupted RGB
generation reads the region again.

Keep the output directory and its JSON completion files to retain this
behavior. The Zarr read cache does not replace the completed map output.
The viewer uses relative tile paths, so the directory can be moved or
served from another location.

Create a map directly from a region::

    geotessera webmap --bbox '-3.0,53.4,-2.9,53.5' \
        --year 2024 --bands 0,1,2 --output map/ --serve

Rebuild the tiles at a different zoom range::

    geotessera webmap --bbox '-3.0,53.4,-2.9,53.5' \
        --year 2024 --bands 0,1,2 --output map/ --min-zoom 6 --max-zoom 16

serve
~~~~~

::

    geotessera serve DIRECTORY [OPTIONS]

Serve all files in ``DIRECTORY`` over HTTP. This command displays an
existing web map without regenerating it. An occupied port causes an
error. Press Ctrl+C to stop the server.

``-p, --port INTEGER``
    Set the listening port. The default is 8000.

``--open/--no-open``
    Open the viewer in a browser. The default is ``--open``.

``--html PATH``
    Select an HTML file relative to ``DIRECTORY``. If omitted, the server
    looks for ``index.html``, ``viewer.html``, ``map.html``, then
    ``coverage.html``.

Serve an existing map on a different port::

    geotessera serve map/ --port 8001 --html viewer.html

coverage
~~~~~~~~

::

    geotessera coverage [OPTIONS]

Show embedding availability as a PNG map, JSON coverage files, and an HTML
globe. Region selectors limit the PNG map and outline vector boundaries;
the globe shows global coverage.

``-o, --output PATH``
    Set the PNG filename or output directory. The default is
    ``tessera_coverage.png``. Supporting JSON files, textures, and
    ``globe.html`` are written alongside the PNG.

``--year INTEGER``
    Show coverage for one year. If omitted, show all available years.

``--by-source``
    Show each dataset version and variant in a separate color, with
    selectable layers in the globe. Omitted version and variant options
    select all datasets with NPY tiles in this mode. Otherwise the
    default is ``v1.1`` and its default variant.

``--tile-color TEXT``
    Set the tile color when year-based colors are disabled. The default
    is ``red``.

``--tile-alpha FLOAT``
    Set tile opacity from 0 to 1. The default is 0.6.

``--tile-size FLOAT``
    Set the tile size multiplier. The default is 1.0.

``--width INTEGER``
    Set the PNG width in pixels. The default is 2000.

``--no-countries``
    Hide country boundaries.

``--no-multi-year-colors``
    Disable the default year-based colors. These colors show tiles with
    all years in green, only the latest year in blue, and other year
    combinations in orange.

``-v, --verbose``
    Print additional coverage details.

This command accepts :ref:`cli-regions` and the dataset and manifest
options in :ref:`cli-data-source`. It does not accept ``--store-url``.

For a dataset published as Icechunk, such as the default, the PNG map is
drawn from the store's tile registry, one rectangle per 2048-pixel tile,
and no globe is written.

Inspect coverage for a region or compare datasets::

    geotessera coverage --country 'United Kingdom' --year 2024
    geotessera coverage --by-source --output coverage/

info
~~~~

::

    geotessera info [OPTIONS]

Show every dataset version and variant, the formats each is published in
(NPY, Zarr, Icechunk) and its store URLs, followed by a summary of the
selected dataset. A ``*`` marks each version's default variant.

``--tiles PATH``
    Inspect a local GeoTIFF or NPY file or directory. Report the files,
    years, bounds, coordinate reference systems, and band counts.

``--geotiffs PATH``
    Use the deprecated alias for ``--tiles``.

``--dataset-version TEXT``, ``--dataset-variant TEXT``
    Select the dataset to summarise. The defaults are ``v1.1`` and that
    version's default variant.

``-v, --verbose``
    Include individual tile details, tile counts per year, and the
    incomplete zone-years of an Icechunk store.

Inspect exported files::

    geotessera info --tiles region/ --verbose

version
~~~~~~~

::

    geotessera version

Print the installed GeoTessera version.

.. _cli-regions:

Region selection
----------------

``--bbox WEST,SOUTH,EAST,NORTH``
    Select WGS84 longitude and latitude bounds. A two-coordinate value,
    ``LON,LAT``, selects the containing 0.1-degree tile.

``--tile LON,LAT``
    Select the 0.1-degree tile containing a WGS84 point.

``--region-file PATH_OR_URL``
    Read a vector region from a local file or URL. GeoJSON, Shapefile,
    and GeoPackage are supported. The input must declare its CRS.

``--country TEXT``
    Select a country by name or code, such as ``United Kingdom`` or ``GB``.

``download`` and region-based ``webmap`` require exactly one selector.
``coverage`` permits no selector to show the world. Zarr exports use the
bounding box of a country or vector region; they do not clip to its polygon.
West must be less than east. Split regions crossing the antimeridian into
two requests.

.. _cli-data-source:

Dataset and storage options
---------------------------

``--dataset-version TEXT``
    Select a dataset version, such as ``v1``, ``v1.1``, or ``v2``. The
    default is ``v1.1``. Run ``geotessera info`` to list known datasets.

``--dataset-variant TEXT``
    Select a variant within the version. If omitted, use the version's
    default variant. If that variant is not published in the requested
    format, use the first variant that is, with a warning: v1.1 NPY
    tiles exist only for ``cambridge``. Variants are separate inference
    runs; embeddings of different versions or variants cannot be
    interchanged.

``--store-url URL_OR_PATH``
    Read a Zarr store from this URL or local path. This overrides the
    store selected by the dataset options. Use it with Zarr downloads
    or region-based web maps.

``--cache-dir DIRECTORY``
    Set the read cache directory. Zarr metadata persists between runs,
    while byte-range reads of sharded embeddings are cached within the
    process. Tile workflows cache manifests here and write embedding
    files to ``--output``.

``--registry-dir DIRECTORY``
    Read ``manifest.parquet`` and ``landmasks.parquet`` from this directory.
    This option applies to tile downloads and coverage. It selects tiles
    when ``download --source auto`` is used and conflicts with an explicit
    ``--source zarr``.

Exit status
-----------

Commands return zero on success and a nonzero status for invalid arguments
or processing failures. An incomplete NPY download returns a nonzero status.
Files completed before a failure may remain in the output directory.

See also
--------

:doc:`zarr_quickstart` describes the Python streaming API.
:doc:`quickstart` describes individual tile downloads.
:doc:`maintenance` describes ``geotessera-registry`` for data maintainers.
