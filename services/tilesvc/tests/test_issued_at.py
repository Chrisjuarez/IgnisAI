"""A dated forecast uses only what was known when it was issued.

`date` asks for the forecast Ignis would have made at that moment: how it is
replayed against what happened. Hours after that moment must get the weather
forecast of the time, not the weather that followed - which was already
observed by the time anyone replays it, and makes the replay look better than
the forecast was.
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("fastapi")

from services.tilesvc import app, dynamic_builder  # noqa: E402
from services.tilesvc.grid import SIZE  # noqa: E402

LON, LAT = -118.40183, 34.561835
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 17, 18, 0, tzinfo=UTC)
ISSUED = dt.datetime(2026, 9, 11, 12, tzinfo=UTC)
WEATHER = ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")
# Wind speeds that say which Open-Meteo answer a frame came from.
OBSERVED, CURRENT_RUN = 99.0, 55.0
RUN_DAYS_BEFORE = {n: 10.0 * n for n in range(1, 8)}


def _open_meteo(url, params=None, timeout=None):
    """Open-Meteo, with each answer's wind telling where it came from."""
    start = dt.date.fromisoformat(params["start_date"]) if "start_date" in params else NOW.date()
    end = dt.date.fromisoformat(params["end_date"]) if "end_date" in params else NOW.date() + dt.timedelta(days=7)
    hours = int((end - start).days + 1) * 24
    times = [(dt.datetime.combine(start, dt.time(), UTC) + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
             for h in range(hours)]
    columns = params["hourly"].split(",")
    speed = OBSERVED if "archive" in url else CURRENT_RUN

    def value(column):
        if column.startswith("wind_speed_10m_previous_day"):
            return RUN_DAYS_BEFORE[int(column.rsplit("day", 1)[1])]
        if column.startswith("wind_speed_10m"):
            return speed
        if column.startswith("wind_direction_10m"):
            return 270.0
        return 10.0

    hourly = {"time": times, **{c: [value(c)] * hours for c in columns}}
    return type("Response", (), {"json": lambda self: {"hourly": hourly}})()


@pytest.fixture
def forecast(monkeypatch, tmp_path):
    """The real weather tiers over a stubbed model and network; records what was used."""
    for name in ("NOAA_GRIB_ENABLED", "NOAA_GRID_CACHE_REQUIRED", "NOAA_GRID_CACHE_DIR",
                 "NOAA_GRID_CACHE_TEMPLATE", "CALIBRATION_PATH", "CALIBRATION_REQUIRED",
                 "HYBRID_SPREAD_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HRRR_ON_DEMAND", "1")
    dynamic_builder._weather_cache.clear()
    hrrr_files, wind_seen = [], []

    def hrrr(_lat, _lon, cycle, forecast_hour, **_kwargs):
        hrrr_files.append((cycle, forecast_hour))
        return {name: np.zeros((SIZE, SIZE), dtype=np.float32) for name in WEATHER}

    u, v = app.MODEL_DYNAMIC_ORDER.index("u"), app.MODEL_DYNAMIC_ORDER.index("v")

    def model(dyn, _stat, **_kwargs):
        # The wind of the newest frame each day was run on.
        wind_seen.append(float(np.hypot(dyn[-1, u], dyn[-1, v]).mean()))
        return np.zeros((SIZE, SIZE), dtype=np.float32)

    monkeypatch.setattr(dynamic_builder, "_utcnow", lambda: NOW)
    monkeypatch.setattr(dynamic_builder, "fetch_hrrr_tile_grids", hrrr)
    monkeypatch.setattr(dynamic_builder.requests, "get", _open_meteo)
    monkeypatch.setattr(dynamic_builder, "_load_firms_points_from_snapshots", lambda *a, **k: [])
    monkeypatch.setattr(app, "_ensure_model_metadata_loaded", lambda *a, **k: None)
    monkeypatch.setattr(app, "_load_static_tensor_with_summary",
                        lambda _tile: (np.zeros((app.MODEL_CS, SIZE, SIZE), dtype=np.float32), {}))
    monkeypatch.setattr(app, "_predict_probability_from_inputs", model)
    monkeypatch.setattr(app, "file_sha256", lambda _path: "0" * 64)

    def run(issued=ISSUED, steps=3):
        hrrr_files.clear(); wind_seen.clear()
        # Called directly, so every Query default is passed explicitly.
        app.predict_multistep(
            lat=LAT, lon=LON, model="ignis", Tseq=3, steps=steps, step_hours=24, thr=None,
            display_floor=None, crop_frac=0.5, ignition=True, date=issued.isoformat(), debug=None,
            burned_area=None,
        )
        return list(hrrr_files), list(wind_seen)

    yield run
    dynamic_builder._weather_cache.clear()


def test_the_next_day_comes_from_an_hrrr_run_published_by_the_issue_time(forecast):
    files, _ = forecast()

    # +24 h: the newest 48-hour run out by 12Z is 06Z, at f30.
    assert (dt.datetime(2026, 9, 11, 6, tzinfo=UTC), 30) in files
    # No file valid after the issue time came from a run that started after it.
    for cycle, lead in files:
        if cycle + dt.timedelta(hours=lead) > ISSUED:
            assert cycle <= ISSUED - dt.timedelta(hours=2), (cycle, lead)


def test_hours_no_run_reached_use_open_meteos_forecast_from_the_issue_time(forecast):
    _, wind = forecast()

    # Day 3 runs on the +48 h hour: HRRR's runs out by 12Z stop short of it, so
    # Open-Meteo serves it - as forecast two days ahead, not as observed.
    assert wind[2] == pytest.approx(RUN_DAYS_BEFORE[2])


def test_observed_weather_on_disk_is_not_served_for_an_hour_after_issue(forecast, monkeypatch, tmp_path):
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "1")
    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    # A later live forecast stored the weather that happened at +24 h.
    later = ISSUED + dt.timedelta(hours=24)
    observed = {name: np.full((SIZE, SIZE), 30.0, np.float32) for name in ("u", "v", "gust", "tempC", "q", "precip")}
    np.savez_compressed(dynamic_builder._noaa_cache_path(LAT, LON, later), source=np.array("noaa_hrrr"), **observed)

    _, wind = forecast()

    assert wind[1] == pytest.approx(0.0), "day 2 must run on the forecast from 06Z, not the stored observation"


def test_forecasts_issued_at_different_times_do_not_share_an_hour(forecast):
    shared_hour = ISSUED + dt.timedelta(hours=24)
    first, _ = forecast(issued=ISSUED, steps=2)
    # Issued a day after that hour, so its input day there is long published.
    second, _ = forecast(issued=shared_hour + dt.timedelta(hours=24), steps=1)

    # For the first it was a forecast; by the second's issue time it had happened.
    assert (dt.datetime(2026, 9, 11, 6, tzinfo=UTC), 30) in first
    assert (shared_hour - dt.timedelta(hours=1), 1) in second
