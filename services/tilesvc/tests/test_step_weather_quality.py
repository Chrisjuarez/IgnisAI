"""Each forecast day reports the weather it was run on.

HRRR reaches about two days ahead, so a live forecast runs its first days on
gridded weather and the rest on one Open-Meteo value spread across the tile.
Quality used to be read once, after the whole rollout, so every day carried the
last fetch's source - and day 1 read "degraded" on HRRR inputs.

A day is run on every weather frame in its input window: day 1 on the input
days, each later day on the window slid forward by the hour before it.
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("fastapi")

from services.tilesvc import app, dynamic_builder  # noqa: E402
from services.tilesvc.grid import SIZE  # noqa: E402

LON, LAT = -118.40183, 34.561835
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
LIVE = dt.datetime(2026, 9, 5, 12, tzinfo=UTC)
# A forecast dated well before NOW: its input days had happened when it was issued.
HINDCAST = dt.datetime(2026, 8, 1, 12, tzinfo=UTC)
WEATHER = ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")


@pytest.fixture
def hrrr_fails_at(monkeypatch):
    """The real weather tiers over a stubbed model; add a valid hour to fail HRRR there."""
    for name in ("NOAA_GRIB_ENABLED", "NOAA_GRID_CACHE_REQUIRED", "NOAA_GRID_CACHE_DIR",
                 "NOAA_GRID_CACHE_TEMPLATE", "CALIBRATION_PATH", "CALIBRATION_REQUIRED",
                 "HYBRID_SPREAD_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HRRR_ON_DEMAND", "1")
    dynamic_builder._weather_cache.clear()
    failing = set()

    def hrrr(_lat, _lon, cycle, forecast_hour, **_kwargs):
        if cycle + dt.timedelta(hours=forecast_hour) in failing:
            raise RuntimeError("archive unreachable")
        return {name: np.zeros((SIZE, SIZE), dtype=np.float32) for name in WEATHER}

    monkeypatch.setattr(dynamic_builder, "_utcnow", lambda: NOW)
    monkeypatch.setattr(dynamic_builder, "fetch_hrrr_tile_grids", hrrr)
    monkeypatch.setattr(dynamic_builder, "_fetch_hourly_series", lambda *a, **k: {})
    monkeypatch.setattr(dynamic_builder, "_load_firms_points_from_snapshots", lambda *a, **k: [])
    monkeypatch.setattr(app, "_ensure_model_metadata_loaded", lambda *a, **k: None)
    monkeypatch.setattr(app, "_load_static_tensor_with_summary",
                        lambda _tile: (np.zeros((app.MODEL_CS, SIZE, SIZE), dtype=np.float32), {}))
    monkeypatch.setattr(app, "_predict_probability_from_inputs",
                        lambda *a, **k: np.zeros((SIZE, SIZE), dtype=np.float32))
    monkeypatch.setattr(app, "file_sha256", lambda _path: "0" * 64)
    yield failing
    dynamic_builder._weather_cache.clear()


def _multistep(start, steps):
    # Called directly, so every Query default is passed explicitly.
    return app.predict_multistep(
        lat=LAT, lon=LON, model="ignis", Tseq=3, steps=steps, step_hours=24, thr=None,
        # No FIRMS is stubbed in, so the seed is the fire the forecast starts from.
        display_floor=None, crop_frac=0.5, ignition=True, date=start.isoformat(), debug=None,
        burned_area=None,
    )["steps"]


def _site_exposure(start, days):
    return app.site_exposure_endpoint(
        site_lat=LAT, site_lon=LON, ignition_lat=None, ignition_lon=None, days=days,
        Tseq=3, step_hours=24, thr=None, arrival_threshold=app.DEFAULT_ARRIVAL_THRESHOLD,
        date=start.isoformat(),
    )


def test_days_run_on_hrrr_are_not_flagged_degraded(hrrr_fails_at):
    steps = _multistep(LIVE, steps=6)

    # The longest HRRR run reaches +24 h, which day 2 starts from. Days 3-6
    # start from hours only Open-Meteo covers.
    assert [s["quality"]["degraded"] for s in steps] == [False, False, True, True, True, True]
    assert [s["data_sources"]["weather_source"] for s in steps] == (
        ["hrrr_on_demand"] * 2 + ["open_meteo_fallback"] * 4
    )


def test_healthz_still_reports_the_latest_fetch(hrrr_fails_at):
    _multistep(LIVE, steps=6)

    assert dynamic_builder.weather_quality_status()["source"] == "open_meteo_fallback"


def test_an_input_day_on_point_weather_flags_the_days_still_run_on_it(hrrr_fails_at):
    # With Tseq=3, day 1 runs on input days -2, -1 and 0, and day 2 on -1, 0
    # and +24 h, which HRRR covers from the 06Z run.
    hrrr_fails_at.add(HINDCAST - dt.timedelta(hours=48))

    steps = _multistep(HINDCAST, steps=2)

    assert [s["quality"]["degraded"] for s in steps] == [True, False]


def test_site_exposure_reports_each_days_weather(hrrr_fails_at):
    exposure = _site_exposure(LIVE, days=3)

    assert [e["quality"]["degraded"] for e in exposure["series"]] == [False, False, True]
    assert [e["weather_source"] for e in exposure["series"]] == (
        ["hrrr_on_demand"] * 2 + ["open_meteo_fallback"]
    )
    # The headline adds up all three days, so day 3's weather qualifies it.
    assert exposure["quality"]["degraded"] is True


def test_site_exposure_headline_flags_point_weather_in_its_inputs(hrrr_fails_at):
    # The last hour fetched, +24 h, comes from HRRR. The newest input day does not.
    hrrr_fails_at.add(LIVE)

    exposure = _site_exposure(LIVE, days=2)

    assert exposure["quality"]["degraded"] is True
    assert exposure["data_sources"]["weather_source"] == "open_meteo_fallback"
