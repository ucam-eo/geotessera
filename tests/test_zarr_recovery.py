"""Recovery after failed Zarr writes, using small real stores."""

from pathlib import PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from rasterio.transform import from_origin

import geotessera.zarr as build


@pytest.fixture(params=[False, True], ids=["local", "file-url"])
def tiny_store(tmp_path, monkeypatch, request):
    monkeypatch.setattr(build, "SHARD_SIZE", 4)
    path = tmp_path / "store.zarr"
    location = build.StoreLocation(path.as_uri() if request.param else str(path))
    root = location.open_group(mode="w", zarr_format=3)
    group = root.create_group("utm31")
    transform = from_origin(500_000, 5_600_000, 10, 10)
    group.attrs.update(
        {
            "spatial:transform": list(transform)[:6],
            "spatial:shape": [4, 8],
            "proj:code": "EPSG:32631",
        }
    )
    group.create_array(
        "embeddings",
        shape=(1, 128, 4, 8),
        chunks=(1, 128, 2, 2),
        shards=(1, 128, 4, 4),
        dtype="i1",
        fill_value=0,
    )
    group.create_array(
        "scales",
        shape=(1, 4, 8),
        chunks=(1, 2, 2),
        shards=(1, 4, 4),
        dtype="f4",
        fill_value=np.inf,
    )
    group.create_array("time", data=np.array([2024], dtype="i4"), chunks=(1,))
    build.create_stretch_arrays(group, 1, 32, 1, 2)
    embeddings = tmp_path / "embeddings.npy"
    scales = tmp_path / "scales.npy"
    np.save(embeddings, np.ones((4, 8, 128), dtype="i1"))
    np.save(scales, np.full((4, 8), 0.25, dtype="f4"))
    tile = build.TileInfo(
        3.05,
        50.55,
        2024,
        32631,
        transform,
        4,
        8,
        "",
        str(embeddings),
        str(scales),
    )
    grid = build.UnifiedZoneGrid(31, [2024], 32631, 500_000, 5_600_000, 8, 4)
    specs = build.build_shard_index([tile], grid, 0)
    monkeypatch.setattr(build, "store_uses_landmask", lambda store: False)
    monkeypatch.setattr(
        build, "gather_tile_infos", lambda *args, **kwargs: {31: [tile]}
    )

    def serial_writes(**kwargs):
        """Use the real workers while replacing only process scheduling.

        Honours ``on_stats``/``flush_every`` the way the pool does: results
        are handed over in batches as they are produced, so a failure
        part-way leaves the earlier batches folded.
        """
        written, failed, statistics = 0, set(), []
        on_stats = kwargs.get("on_stats")
        flush_every = max(1, kwargs.get("flush_every", 64))
        pending = []

        def produced(result):
            if not isinstance(result, dict):
                return
            if on_stats is None:
                statistics.append(result)
                return
            pending.append(result)
            if len(pending) >= flush_every:
                on_stats(list(pending))
                pending.clear()

        for spec in kwargs["shard_specs"]:
            try:
                result = build._write_one_shard(
                    spec,
                    group,
                    sample_cap=kwargs["sample_cap"],
                    depths=kwargs["depths"],
                    use_landmask=False,
                )
                written += bool(result)
                produced(result)
            except OSError:
                failed.add((spec.sr, spec.sc))
        monkeypatch.setattr(build, "_worker_store", group)
        monkeypatch.setattr(build, "_worker_time_index", kwargs["time_index"])
        monkeypatch.setattr(build, "_worker_sample_cap", kwargs["sample_cap"])
        for coord in kwargs.get("stats_coords") or []:
            try:
                produced(build._stats_catchup_worker(coord))
            except OSError:
                failed.add(coord)
        if pending:
            on_stats(list(pending))
        return written, failed, statistics

    monkeypatch.setattr(build, "_write_shards", serial_writes)
    return SimpleNamespace(
        location=location,
        group=group,
        specs=specs,
        embeddings=embeddings,
        scales=scales,
    )


def _fill(store, **kwargs):
    return build.fill_store(
        object(), store.location, year=2024, zones=[31], workers=1, **kwargs
    )


def _fail_array_write(monkeypatch, array_name):
    original = zarr.Array.__setitem__

    def fail(self, selection, value):
        if self.path == f"utm31/{array_name}":
            raise OSError(f"injected {array_name} failure")
        return original(self, selection, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", fail)


@pytest.fixture
def nested_store(tiny_store):
    root = tiny_store.location.open_group(mode="r+")
    root.attrs["geoemb:depths"] = build.depths_attr_value((4, 16))
    for depth in (4, 16):
        tiny_store.group.create_array(
            build.depth_array_name(depth),
            shape=(1, depth, 4, 8),
            chunks=(1, depth, 2, 2),
            shards=(1, depth, 4, 4),
            dtype="i1",
            fill_value=0,
        )
    return tiny_store


@pytest.mark.parametrize("zero_depth", [4, 16, 128])
@pytest.mark.parametrize("water", [False, True], ids=["valid", "water"])
def test_zero_shards_remain_complete(nested_store, zero_depth, water):
    embeddings = np.ones((4, 8, 128), dtype="i1")
    embeddings[..., :zero_depth] = 0
    np.save(nested_store.embeddings, embeddings)
    if water:
        np.save(nested_store.scales, np.full((4, 8), np.nan, dtype="f4"))

    assert _fill(nested_store, collect_stretch_stats=False) == 2
    expected = {(0, 0), (0, 1)}
    for depth in (4, 16, 128):
        name = build.depth_array_name(depth)
        assert build._existing_shards(
            nested_store.location, "utm31", 0, expected, array_name=name
        ) == expected

    # Reopen to check persisted contents, including zero prefixes that use
    # the same finite scales as their nonzero full embedding.
    group = nested_store.location.open_group(mode="r", path="utm31")
    for depth in (4, 16, 128):
        np.testing.assert_array_equal(
            group[build.depth_array_name(depth)][0],
            embeddings[..., :depth].transpose(2, 0, 1),
        )
    if water:
        assert np.isnan(group["scales"][:]).all()
    else:
        np.testing.assert_array_equal(group["scales"][:], 0.25)

    # Resume must catch up statistics without rewriting, and an explicit
    # rebuild must count valid zero vectors as real observations.
    assert _fill(nested_store) == 0
    expected_count = 0 if water else 32
    assert int(group["stretch_stats_count"][0]) == expected_count
    expected_sum = embeddings.reshape(-1, 128).sum(axis=0) * (0 if water else 0.25)
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], expected_sum)
    assert build.backfill_stretch_stats(nested_store.location, zones=[31]) == 1
    assert int(group["stretch_stats_count"][0]) == expected_count
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], expected_sum)
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], 1)
    assert _fill(nested_store) == 0


def test_zero_rewrite_recovers_after_depth_failure(nested_store, monkeypatch):
    assert _fill(nested_store) == 2
    np.save(nested_store.embeddings, np.zeros((4, 8, 128), dtype="i1"))
    with monkeypatch.context() as failure:
        _fail_array_write(failure, "embeddings_d16")
        with pytest.raises(RuntimeError, match="2 shard\\(s\\) failed"):
            _fill(nested_store, skip_existing_shards=False)
    assert build._existing_shards(
        nested_store.location, "utm31", 0, {(0, 0), (0, 1)}
    ) == set()
    assert _fill(nested_store) == 2
    assert _fill(nested_store) == 0
    assert int(nested_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(nested_store.group["stretch_stats_sum"][0], 0)


def test_resume_repairs_legacy_zero_shards_with_statistics(tiny_store):
    np.save(tiny_store.embeddings, np.zeros((4, 8, 128), dtype="i1"))
    group = tiny_store.group
    # Older writers let Zarr omit the zero embedding objects, even though
    # their scales and fill-time statistics were successfully published.
    group["embeddings"][:] = 0
    group["scales"][:] = 0.25
    group["stretch_stats_count"][0] = 32
    group["stretch_stats_shards"][0] = 1
    assert build._existing_shards(
        tiny_store.location, "utm31", 0, {(0, 0), (0, 1)}
    ) == set()

    assert _fill(tiny_store) == 2
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 0)
    assert _fill(tiny_store) == 0


@pytest.mark.parametrize("array_name", ["scales", "embeddings"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["initial", "rewrite"])
def test_failed_shard_write_is_retried(tiny_store, monkeypatch, array_name, rewrite):
    if rewrite:
        assert _fill(tiny_store) == 2
        np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    with monkeypatch.context() as failure:
        _fail_array_write(failure, array_name)
        with pytest.raises(RuntimeError, match="2 shard\\(s\\) failed"):
            _fill(tiny_store, skip_existing_shards=not rewrite)
    assert (
        build._existing_shards(tiny_store.location, "utm31", 0, {(0, 0), (0, 1)})
        == set()
    )
    assert _fill(tiny_store) == 2
    np.testing.assert_array_equal(
        tiny_store.group["scales"][:], 0.5 if rewrite else 0.25
    )
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_resume_repairs_legacy_embedding_without_scales(tiny_store):
    tiny_store.group["embeddings"][0, :, :, :4] = 1
    assert build._existing_shards(tiny_store.location, "utm31", 0, {(0, 0)}) == set()
    assert _fill(tiny_store) == 2
    assert np.isfinite(tiny_store.group["scales"][:]).all()


def test_resume_checks_both_arrays_without_listing(tiny_store, monkeypatch):
    _fill(tiny_store)
    monkeypatch.setattr(
        tiny_store.location,
        "listdir",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("no listing")),
    )
    assert build._existing_shards(
        tiny_store.location, "utm31", 0, {(0, 0), (0, 1)}
    ) == {(0, 0), (0, 1)}


def test_resume_accepts_windows_directory_listings(tiny_store, monkeypatch):
    assert _fill(tiny_store) == 2
    listdir = tiny_store.location.listdir

    def windows_listing(*parts, **kwargs):
        return [str(PureWindowsPath(entry)) for entry in listdir(*parts, **kwargs)]

    monkeypatch.setattr(tiny_store.location, "listdir", windows_listing)
    assert build._existing_shards(
        tiny_store.location, "utm31", 0, {(0, 0), (0, 1)}
    ) == {(0, 0), (0, 1)}
    assert _fill(tiny_store) == 0
    assert build.backfill_stretch_stats(tiny_store.location, zones=[31]) == 1
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_extend_recovers_failed_coordinate_write(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        _fail_array_write(failure, "time")
        with pytest.raises(OSError, match="time failure"):
            build.extend_store(tiny_store.location, [2025, 2026], consolidate=False)
    assert tiny_store.group["time"][:].tolist() == [2024, 0, 0]
    with pytest.raises(ValueError, match="re-run zarr-extend"):
        _fill(tiny_store)
    assert build.extend_store(tiny_store.location, [2025, 2026], consolidate=False) == 1
    assert tiny_store.group["time"][:].tolist() == [2024, 2025, 2026]
    assert build._PENDING_TIME_ATTR not in tiny_store.group.attrs
    for name in ("embeddings", "scales", *build.STRETCH_ARRAY_NAMES):
        assert tiny_store.group[name].shape[0] == 3


def test_extend_repairs_legacy_trailing_zeros(tiny_store):
    tiny_store.group["time"].resize((2,))
    build.extend_store(tiny_store.location, [2025], consolidate=False)
    assert tiny_store.group["time"][:].tolist() == [2024, 2025]


def test_extend_retries_failed_consolidation(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        failure.setattr(
            build,
            "consolidate_store",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("consolidation")),
        )
        with pytest.raises(OSError, match="consolidation"):
            build.extend_store(tiny_store.location, [2025])
    assert build.extend_store(tiny_store.location, [2025]) == 0
    root = zarr.open_group(
        tiny_store.location.as_zarr_store(), mode="r", use_consolidated=True
    )
    assert root["utm31/embeddings"].shape[0] == 2


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), OSError("transport"), ValueError("metadata")]
)
def test_fill_propagates_zone_access_errors(tiny_store, monkeypatch, error):
    original = tiny_store.location.open_group

    def fail_zone(*args, **kwargs):
        if kwargs.get("path") == "utm31":
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_store.location, "open_group", fail_zone)
    with pytest.raises(type(error), match=str(error)):
        _fill(tiny_store)


def test_rewritten_shards_replace_stretch_statistics(tiny_store):
    assert _fill(tiny_store) == 2
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    assert _fill(tiny_store, skip_existing_shards=False) == 2
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 16.0)
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_prod"][0], 8.0)
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_rewrite_rebuilds_statistics_without_backfill_scan(tiny_store, monkeypatch):
    """A rewrite folds the writers' own statistics; no second read of the zone."""
    assert _fill(tiny_store) == 2
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))

    def no_backfill(*args, **kwargs):
        raise AssertionError("rewrite must not fall back to backfill_stretch_stats")

    catchups = []
    catchup = build._stats_catchup_worker
    monkeypatch.setattr(build, "backfill_stretch_stats", no_backfill)
    monkeypatch.setattr(
        build, "_stats_catchup_worker", lambda c: catchups.append(c) or catchup(c)
    )
    assert _fill(tiny_store, skip_existing_shards=False) == 2
    assert catchups == []
    group = tiny_store.group
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 16.0)
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], 1)
    assert int(group["stretch_sample_count"][0]) > 0


def test_rewrite_reads_back_shards_it_does_not_touch(tiny_store, monkeypatch):
    """Shards in the store outside the rewrite are read, not forgotten."""
    assert _fill(tiny_store) == 2
    # Only the first shard is rewritten; the second stays as the store has it.
    monkeypatch.setattr(
        build,
        "build_shard_index",
        lambda tiles, grid, t: [s for s in tiny_store.specs if s.sc == 0],
    )
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    catchups = []
    catchup = build._stats_catchup_worker
    monkeypatch.setattr(
        build, "_stats_catchup_worker", lambda c: catchups.append(c) or catchup(c)
    )
    assert _fill(tiny_store, skip_existing_shards=False) == 1
    assert catchups == [(0, 1)]
    group = tiny_store.group
    assert int(group["stretch_stats_count"][0]) == 32
    # 16 px at 0.5 (rewritten) + 16 px at 0.25 (read back), per band.
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 16 * 0.5 + 16 * 0.25)
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], 1)


def test_interrupted_rebuild_resumes_by_reading_the_store(tiny_store, monkeypatch):
    """A slot left at -1 is rebuilt on the next fill from what is in the store."""
    assert _fill(tiny_store) == 2
    tiny_store.group["stretch_stats_count"][0] = -1
    tiny_store.group["stretch_stats_sum"][0] = 1e9
    catchups = []
    catchup = build._stats_catchup_worker
    monkeypatch.setattr(
        build, "_stats_catchup_worker", lambda c: catchups.append(c) or catchup(c)
    )
    assert _fill(tiny_store) == 0
    assert sorted(catchups) == [(0, 0), (0, 1)]
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 8.0)


def _record_catchups(monkeypatch, fail=()):
    """Spy on catch-up reads; coords in *fail* raise instead of reading."""
    seen = []
    real = build._stats_catchup_worker

    def spy(coord):
        seen.append(coord)
        if coord in fail:
            raise OSError("injected statistics read failure")
        return real(coord)

    monkeypatch.setattr(build, "_stats_catchup_worker", spy)
    return seen


def test_backfill_resumes_after_a_failed_read(tiny_store, monkeypatch):
    """Folded batches stay folded; a retry reads only what is left."""
    assert _fill(tiny_store) == 2
    group = tiny_store.group
    marker = build._stretch_backfill_marker_parts("utm31", 2024)
    with monkeypatch.context() as failure:
        _record_catchups(failure, fail={(0, 1)})
        with pytest.raises(RuntimeError, match="1 shard read\\(s\\) failed"):
            build.backfill_stretch_stats(tiny_store.location, zones=[31], batch_size=1)
    # The first shard's batch was folded before the second failed.
    assert int(group["stretch_stats_count"][0]) == 16
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], [[1, 0]])
    assert tiny_store.location.state.exists(*marker)

    catchups = _record_catchups(monkeypatch)
    assert build.backfill_stretch_stats(tiny_store.location, zones=[31]) == 1
    assert catchups == [(0, 1)]
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 8.0)
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], 1)
    assert not tiny_store.location.state.exists(*marker)


def test_backfill_without_marker_starts_from_empty(tiny_store, monkeypatch):
    """No marker means a fresh repair: the old sums are discarded."""
    assert _fill(tiny_store) == 2
    group = tiny_store.group
    group["stretch_stats_sum"][0] = 1e9
    catchups = _record_catchups(monkeypatch)
    assert build.backfill_stretch_stats(tiny_store.location, zones=[31]) == 1
    assert sorted(catchups) == [(0, 0), (0, 1)]
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 8.0)


def test_backfill_restarts_when_resume_state_is_invalid(tiny_store, monkeypatch):
    """A marker beside a slot at -1 cannot be continued from."""
    assert _fill(tiny_store) == 2
    group = tiny_store.group
    marker = build._stretch_backfill_marker_parts("utm31", 2024)
    tiny_store.location.state.write_bytes(b"", *marker)
    group["stretch_stats_count"][0] = -1
    catchups = _record_catchups(monkeypatch)
    assert build.backfill_stretch_stats(tiny_store.location, zones=[31]) == 1
    assert sorted(catchups) == [(0, 0), (0, 1)]
    assert int(group["stretch_stats_count"][0]) == 32
    assert not tiny_store.location.state.exists(*marker)


def test_writer_stats_are_folded_as_they_arrive(tiny_store, monkeypatch):
    """A fill killed mid-way keeps the statistics of the shards it wrote."""
    group = tiny_store.group
    real_write = build._write_one_shard

    def write_or_die(spec, *args, **kwargs):
        if spec.sc == 1:
            raise OSError("injected write failure")
        return real_write(spec, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(build, "_write_one_shard", write_or_die)
        with pytest.raises(RuntimeError, match="1 shard\\(s\\) failed"):
            _fill(tiny_store, stats_batch_size=1)
    # The first shard's statistics were folded before the second failed.
    assert int(group["stretch_stats_count"][0]) == 16
    np.testing.assert_array_equal(group["stretch_stats_shards"][0], [[1, 0]])

    # The retry writes the missing shard only and reads nothing back.
    catchups = _record_catchups(monkeypatch)
    assert _fill(tiny_store) == 1
    assert catchups == []
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 8.0)


def test_fill_catchup_resumes_in_batches(tiny_store, monkeypatch):
    """An emptied slot is caught up a batch at a time by ordinary fills."""
    assert _fill(tiny_store) == 2
    group = tiny_store.group
    build.reset_zone_stretch_stats(group, 0)
    assert int(group["stretch_stats_count"][0]) == 0
    with monkeypatch.context() as failure:
        _record_catchups(failure, fail={(0, 1)})
        with pytest.raises(RuntimeError, match="1 shard\\(s\\) failed"):
            _fill(tiny_store, stats_batch_size=1)
    assert int(group["stretch_stats_count"][0]) == 16
    catchups = _record_catchups(monkeypatch)
    assert _fill(tiny_store) == 0
    assert catchups == [(0, 1)]
    assert int(group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(group["stretch_stats_sum"][0], 8.0)


def test_rewrite_without_stats_invalidates_then_rebuilds(tiny_store):
    _fill(tiny_store)
    np.save(tiny_store.scales, np.full((4, 8), 0.5, dtype="f4"))
    _fill(tiny_store, skip_existing_shards=False, collect_stretch_stats=False)
    assert int(tiny_store.group["stretch_stats_count"][0]) == -1
    with pytest.raises(RuntimeError, match="statistics are incomplete"):
        build.compute_stretch_from_stats(tiny_store.location, 2024, persist=False)
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 16.0)


def test_failed_stats_update_rebuilds_on_retry(tiny_store, monkeypatch):
    with monkeypatch.context() as failure:
        _fail_array_write(failure, "stretch_stats_sum")
        with pytest.raises(OSError, match="stretch_stats_sum failure"):
            _fill(tiny_store)
    assert int(tiny_store.group["stretch_stats_count"][0]) == -1
    assert _fill(tiny_store) == 0
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32
    np.testing.assert_array_equal(tiny_store.group["stretch_stats_sum"][0], 8.0)


def test_failed_stats_catchup_keeps_what_was_read(tiny_store, monkeypatch):
    _fill(tiny_store, collect_stretch_stats=False)
    with monkeypatch.context() as failure:
        _record_catchups(failure, fail={(0, 1)})
        with pytest.raises(RuntimeError, match="1 shard\\(s\\) failed"):
            _fill(tiny_store)
    # The read that succeeded is folded; the slot stays valid and partial.
    assert int(tiny_store.group["stretch_stats_count"][0]) == 16
    catchups = _record_catchups(monkeypatch)
    assert _fill(tiny_store) == 0
    assert catchups == [(0, 1)]
    assert int(tiny_store.group["stretch_stats_count"][0]) == 32


def test_worker_pool_reports_failed_stats_catchup(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    def thread_pool(**kwargs):
        return ThreadPoolExecutor(max_workers=1)

    def fail_catchup(coord):
        raise OSError("injected statistics read failure")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", thread_pool)
    monkeypatch.setattr(build, "_stats_catchup_worker", fail_catchup)
    written, failed, statistics = build._write_shards(
        build.StoreLocation(str(tmp_path)),
        "utm31",
        [],
        workers=1,
        source_options=None,
        label="test",
        console=None,
        stats_coords=[(0, 0)],
    )
    assert (written, failed, statistics) == (0, {(0, 0)}, [])
