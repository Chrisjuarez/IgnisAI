import datetime as dt

import pytest

from services.tilesvc import dynamic_builder


SERIES_HOURS = 24 * 7


def _hourly_series(start: dt.datetime, hours: int) -> dict:
    """Series whose wind turns steadily, so each hour is distinguishable."""
    times, speeds, directions = [], [], []
    for offset in range(hours):
        times.append((start + dt.timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M"))
        speeds.append(1.0 + offset)
        directions.append((offset * 15) % 360)

    return {
        "time": times,
        "temperature_2m": [20.0] * hours,
        "relative_humidity_2m": [30.0] * hours,
        "wind_speed_10m": speeds,
        "wind_direction_10m": directions,
        "wind_gusts_10m": [speed * 1.5 for speed in speeds],
        "precipitation": [0.0] * hours,
    }


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def open_meteo(monkeypatch):
    monkeypatch.setenv("NOAA_GRIB_ENABLED", "0")
    dynamic_builder._weather_cache.clear()

    start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    requested = []

    def stub_get(url, params=None, timeout=None):
        requested.append(params)
        return _StubResponse({"hourly": _hourly_series(start, SERIES_HOURS)})

    monkeypatch.setattr(dynamic_builder.requests, "get", stub_get)
    return requested, start


def test_rollout_shares_one_series_and_varies_with_lead_time(open_meteo):
    requested, start = open_meteo

    winds = []
    for lead_hours in (0, 24, 48, 72, 96, 120):
        grids = dynamic_builder.fetch_weather_grids(
            34.0, -118.0, ref_time=start + dt.timedelta(hours=lead_hours)
        )
        winds.append((float(grids["u"][0, 0]), float(grids["v"][0, 0])))

    assert len(requested) == 1, "a rollout should cost one hourly fetch, not one per step"
    assert len(set(winds)) == len(winds), "each forecast step needs its own lead-time weather"


def test_forecast_request_spans_the_rollout_horizon(open_meteo):
    requested, start = open_meteo

    dynamic_builder.fetch_weather_grids(34.0, -118.0, ref_time=start)

    params = requested[0]
    assert "current" not in params, "the current endpoint cannot express lead time"
    assert params["forecast_days"] >= 6, "must cover a six-step, 24h-per-step rollout"


def test_lead_time_beyond_the_series_clamps_to_its_last_hour(open_meteo):
    _, start = open_meteo

    last = dynamic_builder.fetch_weather_grids(
        34.0, -118.0, ref_time=start + dt.timedelta(hours=SERIES_HOURS - 1)
    )
    beyond = dynamic_builder.fetch_weather_grids(
        34.0, -118.0, ref_time=start + dt.timedelta(days=30)
    )

    assert float(beyond["u"][0, 0]) == float(last["u"][0, 0])
    assert float(beyond["gust"][0, 0]) == float(last["gust"][0, 0])
