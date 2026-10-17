"""Tests for NOAA_GRID_CACHE_REQUIRED.

Serving falls back to Open-Meteo when a weather cache hour is missing. A
training build must not: the fallback is one value across the whole tile,
and the corpus would not say which frames got it.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from services.tilesvc import dynamic_builder
from services.tilesvc.grid import SIZE
from services.tilesvc.static_catalog import InputUnavailable

REF_HOUR = dt.datetime(2026, 5, 14, 20, tzinfo=dt.timezone.utc)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    return tmp_path


def test_a_missing_hour_falls_back_when_not_required(cache_dir, monkeypatch):
    monkeypatch.delenv("NOAA_GRID_CACHE_REQUIRED", raising=False)

    assert dynamic_builder._fetch_noaa_cached_weather_grids(39.48, -112.27, REF_HOUR) is None


def test_a_missing_hour_is_an_error_when_required(cache_dir, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_REQUIRED", "1")

    with pytest.raises(InputUnavailable) as exc:
        dynamic_builder._fetch_noaa_cached_weather_grids(39.48, -112.27, REF_HOUR)

    assert exc.value.reason == "noaa_cache_missing"
    assert exc.value.details["path"].endswith("20260514T20_39.48_-112.27.npz")


def test_an_unreadable_hour_is_an_error_when_required(cache_dir, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_REQUIRED", "1")
    (cache_dir / "20260514T20_39.48_-112.27.npz").write_bytes(b"not an npz")

    with pytest.raises(InputUnavailable) as exc:
        dynamic_builder._fetch_noaa_cached_weather_grids(39.48, -112.27, REF_HOUR)

    assert exc.value.reason == "noaa_cache_invalid"


def test_required_reads_the_cache_even_without_noaa_grib_enabled(cache_dir, monkeypatch):
    monkeypatch.setenv("NOAA_GRID_CACHE_REQUIRED", "1")
    monkeypatch.delenv("NOAA_GRIB_ENABLED", raising=False)
    arrays = {name: np.full((SIZE, SIZE), 2.0, np.float32) for name in ("u", "v", "gust", "tempC", "q", "precip")}
    np.savez_compressed(cache_dir / "20260514T20_39.48_-112.27.npz", **arrays)

    grids, _quality = dynamic_builder._fetch_noaa_cached_weather_grids(39.48, -112.27, REF_HOUR)

    assert float(grids["u"][0, 0]) == 2.0
