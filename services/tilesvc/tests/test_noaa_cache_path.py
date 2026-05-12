"""Tests for NOAA gridded weather cache path resolution and source tagging.

These cover the bug where the runtime cache writer produced files named
``{iso_hour}_{lat}_{lon}.npz`` while the tilesvc reader's default fallback
only looked for ``{iso_hour}.npz``. Without ``NOAA_GRID_CACHE_TEMPLATE``
set, the cache effectively never hit and every prediction quietly fell
back to single-point Open-Meteo. The fix makes the reader try the
lat/lon-tagged variant first, then the legacy filename.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from services.tilesvc import dynamic_builder
from services.tilesvc.grid import SIZE


REF_HOUR = dt.datetime(2025, 1, 7, 18, tzinfo=dt.timezone.utc)


def _write_noaa_npz(path, *, source: str | None = None) -> None:
    """Write a minimal valid NOAA cache npz, optionally stamped with a source tag."""
    arrays = {
        name: np.full((SIZE, SIZE), 1.0, dtype=np.float32)
        for name in ("u", "v", "gust", "tempC", "q", "precip")
    }
    if source is not None:
        arrays["source"] = np.array(str(source))
    np.savez_compressed(path, **arrays)


def test_noaa_cache_path_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("NOAA_GRID_CACHE_DIR", raising=False)
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)

    assert dynamic_builder._noaa_cache_path(34.05, -118.55, REF_HOUR) is None


def test_noaa_cache_path_prefers_lat_lon_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    primary = tmp_path / "20250107T18_34.05_-118.55.npz"
    _write_noaa_npz(primary)

    resolved = dynamic_builder._noaa_cache_path(34.05, -118.55, REF_HOUR)

    # The runtime cache writer's filename convention wins by default.
    assert resolved == primary


def test_noaa_cache_path_falls_back_to_legacy_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    legacy = tmp_path / "20250107T18.npz"
    _write_noaa_npz(legacy)

    resolved = dynamic_builder._noaa_cache_path(34.05, -118.55, REF_HOUR)

    # Old caches keep working until they're regenerated.
    assert resolved == legacy


def test_noaa_cache_path_returns_canonical_when_neither_present(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)

    resolved = dynamic_builder._noaa_cache_path(34.05, -118.55, REF_HOUR)

    # Health checks need a stable, canonical path to surface in their
    # error messages even when nothing is cached yet.
    assert resolved == tmp_path / "20250107T18_34.05_-118.55.npz"


def test_noaa_cache_template_overrides_directory_default(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "NOAA_GRID_CACHE_TEMPLATE",
        str(tmp_path / "{date}/{hour}-{lat}-{lon}.npz"),
    )

    resolved = dynamic_builder._noaa_cache_path(34.05, -118.55, REF_HOUR)

    assert resolved == tmp_path / "20250107" / "18-34.05--118.55.npz"


def test_fetch_noaa_cached_weather_grids_propagates_source_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    path = tmp_path / "20250107T18_34.05_-118.55.npz"
    _write_noaa_npz(path, source="noaa_hrrr")

    grids = dynamic_builder._fetch_noaa_cached_weather_grids(34.05, -118.55, REF_HOUR)

    assert grids is not None
    assert grids["u"].shape == (SIZE, SIZE)
    quality = dynamic_builder.weather_quality_status()
    assert quality["status"] == "ok"
    assert quality["source"] == "noaa_hrrr"


def test_fetch_noaa_cached_weather_grids_falls_back_to_generic_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    path = tmp_path / "20250107T18_34.05_-118.55.npz"
    _write_noaa_npz(path, source=None)

    grids = dynamic_builder._fetch_noaa_cached_weather_grids(34.05, -118.55, REF_HOUR)

    assert grids is not None
    assert dynamic_builder.weather_quality_status()["source"] == "noaa_gridded"


def test_weather_quality_status_marks_open_meteo_degraded(tmp_path, monkeypatch):
    """Sanity check: when no cache is present, status falls back to degraded."""
    monkeypatch.delenv("NOAA_GRID_CACHE_DIR", raising=False)
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    # Force the source tracker into the open-meteo state.
    dynamic_builder._set_weather_source("open_meteo_fallback", "test")

    status = dynamic_builder.weather_quality_status()

    assert status["status"] == "degraded"
    assert status["source"] == "open_meteo_fallback"
