"""
The on-demand HRRR fetch writes into the same npz cache the backfill pipeline
fills, and the cache reader lives on the other side of that seam. These cover
the handoff: filename, source stamp, and the tag that reaches /healthz.
"""

import datetime as dt

import numpy as np
import pytest

from services.tilesvc import dynamic_builder

UTC = dt.timezone.utc
REF_TIME = dt.datetime(2025, 1, 7, 18, tzinfo=UTC)


def _fake_grids():
    rng = np.random.default_rng(0)
    grids = {
        name: rng.normal(size=(dynamic_builder.SIZE, dynamic_builder.SIZE)).astype(np.float32)
        for name in ("u", "v", "gust", "tempC", "q", "precip")
    }
    grids["temp"] = grids["tempC"]
    grids["prcp"] = grids["precip"]
    grids["rh"] = np.full((dynamic_builder.SIZE, dynamic_builder.SIZE), np.nan, np.float32)
    return grids


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    monkeypatch.setenv("HRRR_ON_DEMAND", "1")
    dynamic_builder._weather_cache.clear()
    return tmp_path


def test_fetched_hour_is_readable_by_the_cache_reader(cache_dir, monkeypatch):
    written = _fake_grids()
    monkeypatch.setattr(
        dynamic_builder, "fetch_hrrr_tile_grids", lambda *a, **k: dict(written)
    )

    fetched = dynamic_builder._fetch_hrrr_weather_grids(34.078, -118.555, REF_TIME)
    assert fetched is not None
    assert dynamic_builder.weather_quality_status()["status"] == "ok"

    # The writer must land on the lat/lon-tagged name the reader looks up first;
    # the legacy {iso_hour}.npz name is why the cache silently never hit before.
    assert [p.name for p in cache_dir.glob("*.npz")] == ["20250107T18_34.08_-118.56.npz"]

    dynamic_builder._weather_cache.clear()
    read_back = dynamic_builder._fetch_noaa_cached_weather_grids(34.078, -118.555, REF_TIME)
    assert read_back is not None
    np.testing.assert_allclose(read_back["u"], written["u"])

    status = dynamic_builder.weather_quality_status()
    assert status["source"] == "noaa_hrrr", "cache must report the real product, not the generic tag"
    assert status["status"] == "ok"


def test_a_failed_fetch_falls_through_instead_of_raising(cache_dir, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("archive unreachable")

    monkeypatch.setattr(dynamic_builder, "fetch_hrrr_tile_grids", boom)

    assert dynamic_builder._fetch_hrrr_weather_grids(34.078, -118.555, REF_TIME) is None
    assert dynamic_builder.weather_quality_status()["status"] == "degraded"
    assert not list(cache_dir.glob("*.npz"))


def test_the_fetch_is_skipped_unless_explicitly_enabled(cache_dir, monkeypatch):
    monkeypatch.setenv("HRRR_ON_DEMAND", "0")
    monkeypatch.setattr(
        dynamic_builder,
        "fetch_hrrr_tile_grids",
        lambda *a, **k: pytest.fail("must not fetch while disabled"),
    )
    assert dynamic_builder._fetch_hrrr_weather_grids(34.078, -118.555, REF_TIME) is None


def test_on_demand_hrrr_counts_as_gridded_quality():
    assert "hrrr_on_demand" in dynamic_builder.GRIDDED_WEATHER_SOURCES
    # Derived from the backfill vocabulary, so the two cannot drift apart.
    for tag in dynamic_builder._NOAA_GRID_SOURCES:
        assert tag in dynamic_builder.GRIDDED_WEATHER_SOURCES
