"""The fire picture ends at the request, not at the top of the hour.

Bouquet, 2026-10-04: NOAA-21 passed at 1:33 PM PDT and its two detections were
on the map within minutes. A forecast run at 1:54 PM left them out, because the
window was cut at 1:00 PM - so the freshest data any forecast could have was
missing from every forecast run between a pass and the next hour.
"""
import datetime as dt

import numpy as np
import pytest

from services.tilesvc import dynamic_builder as db
from services.tilesvc.grid import lonlat_to_tile, lonlat_to_xy_m, tile_affine

LAT, LON = 34.561835, -118.40183
UTC = dt.timezone.utc
REQUEST = dt.datetime(2026, 10, 4, 20, 54, tzinfo=UTC)          # 1:54 PM PDT
NOAA21_PASS = dt.datetime(2026, 10, 4, 20, 33, tzinfo=UTC)      # 1:33 PM PDT
WEATHER = ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")


def _detection(ts):
    return {"lon": LON, "lat": LAT, "ts": ts, "frp": 9.4}


@pytest.fixture
def builder(monkeypatch):
    """build_dynamic_for_tile over given detections, with the clock pinned."""
    weather_times = []

    def weather(lat, lon, ref_time=None):
        weather_times.append(ref_time)
        return {name: np.zeros((db.SIZE, db.SIZE), dtype=np.float32) for name in WEATHER}

    def build(detections, ref_time=None):
        monkeypatch.setattr(db, "_load_firms_points_from_snapshots",
                            lambda bbox, start, end: [p for p in detections if start <= p["ts"] <= end])
        return db.build_dynamic_for_tile(LAT, LON, T_seq=3, hours_step=24, ref_time=ref_time,
                                         channel_order=["fire_t", "u", "v", "gust", "tempC", "q", "precip"])

    monkeypatch.setattr(db, "_utcnow", lambda: REQUEST)
    monkeypatch.setattr(db, "fetch_weather_grids", weather)
    build.weather_times = weather_times
    return build


def _alight_at_detection(frame):
    col, row = ~tile_affine(lonlat_to_tile(LON, LAT)) * lonlat_to_xy_m(LON, LAT)
    return frame[int(row), int(col)] > 0


def test_a_pass_after_the_top_of_the_hour_is_in_the_forecast(builder):
    x = builder([_detection(NOAA21_PASS)])

    assert _alight_at_detection(x[-1, 0])


def test_weather_stays_on_the_hour(builder):
    builder([_detection(NOAA21_PASS)])

    on_the_hour = REQUEST.replace(minute=0)
    assert builder.weather_times == [on_the_hour - dt.timedelta(days=2),
                                     on_the_hour - dt.timedelta(days=1),
                                     on_the_hour]


def test_a_dated_request_uses_its_own_time_for_both(builder):
    asked_for = dt.datetime(2025, 1, 7, 18, 30, tzinfo=UTC)
    before, after = asked_for - dt.timedelta(minutes=20), asked_for + dt.timedelta(minutes=10)

    seen = builder([_detection(before)], ref_time=asked_for)
    unseen = builder([_detection(after)], ref_time=asked_for)

    assert _alight_at_detection(seen[-1, 0])
    assert not _alight_at_detection(unseen[-1, 0])
    assert builder.weather_times[-1] == asked_for
