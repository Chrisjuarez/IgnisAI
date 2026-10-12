"""Tests for the hour the event ETL builds each fire's frames at.

Each fire's weather cache holds one hour a day, the hour it was built for.
Asking for any other hour misses every file.
"""
import os

from ignis_ml.scripts.etl_events import latest_cached_time, use_profile


def make_cache(root, profile, firms_days, weather_hours):
    firms = root / profile / "firms_snapshots"
    noaa = root / profile / "noaa_grid_cache"
    firms.mkdir(parents=True)
    noaa.mkdir(parents=True)
    for day in firms_days:
        (firms / f"{day}.csv").write_text("latitude,longitude,acq_date,acq_time,frp\n")
    for hour in weather_hours:
        (noaa / f"{hour}_39.48_-112.27.npz").write_bytes(b"")
        (noaa / f"._{hour}_39.48_-112.27.npz").write_bytes(b"")


def test_the_reference_time_uses_the_cache_hour(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path))
    make_cache(tmp_path, "fire", ["2026-05-13", "2026-05-14"], ["20260513T15", "20260514T15"])

    assert latest_cached_time("fire") == "2026-05-14T15:00:00Z"


def test_the_reference_time_is_the_newest_day_with_both_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path))
    make_cache(tmp_path, "fire", ["2026-05-13", "2026-05-14", "2026-05-15"], ["20260513T20", "20260514T20", "20260516T20"])

    assert latest_cached_time("fire") == "2026-05-14T20:00:00Z"


def test_no_paired_day_means_no_reference_time(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path))
    make_cache(tmp_path, "fire", ["2026-05-13"], ["20260520T20"])

    assert latest_cached_time("fire") is None


def test_use_profile_requires_the_weather_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path))
    # use_profile writes os.environ directly; setting each name first lets
    # monkeypatch restore it after the test.
    for name in ("FIRMS_SNAPSHOT_DIR", "FIRMS_SNAPSHOT_REQUIRED", "NOAA_GRID_CACHE_DIR", "NOAA_GRID_CACHE_REQUIRED"):
        monkeypatch.setenv(name, "")
    make_cache(tmp_path, "fire", ["2026-05-13"], ["20260513T20"])

    assert use_profile("fire")
    assert os.environ["NOAA_GRID_CACHE_REQUIRED"] == "1"
