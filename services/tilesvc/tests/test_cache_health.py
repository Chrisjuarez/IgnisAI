import datetime as dt

from services.tilesvc import cache_health


def _write(directory, name):
    directory.mkdir(exist_ok=True)
    (directory / name).write_bytes(b"")


def test_noaa_cache_reports_data_age_rather_than_download_age(tmp_path):
    cache = tmp_path / "noaa_grid_cache"
    # Freshly written file describing January 2025 weather — the shape of a
    # container that downloads a stale cache profile at boot.
    _write(cache, "20250112T18_34.08_-118.56.npz")

    status = cache_health.noaa_cycle_status(str(cache))

    assert status["age_seconds"] < 60, "the file itself was just written"
    assert status["data_age_seconds"] > 6 * 3600
    assert status["stale"] is True
    assert status["valid_time"].startswith("2025-01-12T18:00")


def test_noaa_cache_for_the_current_hour_is_not_stale(tmp_path):
    cache = tmp_path / "noaa_grid_cache"
    now = dt.datetime.now(dt.timezone.utc)
    _write(cache, f"{now.strftime('%Y%m%dT%H')}_34.08_-118.56.npz")

    assert cache_health.noaa_cycle_status(str(cache))["stale"] is False


def test_latest_entry_is_chosen_by_validity_not_write_order(tmp_path):
    cache = tmp_path / "noaa_grid_cache"
    _write(cache, "20250112T18_34.08_-118.56.npz")
    _write(cache, "20250107T18_34.08_-118.56.npz")

    assert cache_health.noaa_cycle_status(str(cache))["latest_file"].startswith("20250112")


def test_firms_snapshot_freshness_uses_the_observation_date(tmp_path):
    snapshots = tmp_path / "firms_snapshots"
    _write(snapshots, "2025-01-07.csv")

    status = cache_health.firms_snapshot_status(str(snapshots))

    assert status["stale"] is True
    assert status["valid_time"].startswith("2025-01-07")


def test_unparseable_filenames_report_unknown_rather_than_fresh(tmp_path):
    cache = tmp_path / "noaa_grid_cache"
    _write(cache, "not-a-timestamp.npz")

    status = cache_health.noaa_cycle_status(str(cache))

    assert status["valid_time"] is None
    assert status["stale"] is None


def test_missing_and_empty_directories_are_reported_distinctly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert cache_health.noaa_cycle_status(str(tmp_path / "absent"))["error"] == "missing"
    assert cache_health.noaa_cycle_status(str(empty))["error"] == "empty"
    assert cache_health.noaa_cycle_status(None) == {"configured": False}
