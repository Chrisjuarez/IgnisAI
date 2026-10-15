"""Where /predict_multistep draws its day bands.

A viewer reads a band as "fire here by day N". The bands follow the decision
threshold - the line the contour and the next-fire layer use - not the
calibrated display floor, which drew several times the area that burned.
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from services.tilesvc import app  # noqa: E402
from services.tilesvc.grid import SIZE, lonlat_to_tile, tile_bounds_lonlat  # noqa: E402

LON, LAT = -118.40183, 34.561835


@pytest.fixture
def forecast_bands(monkeypatch):
    """Day bands for a two-day forecast that scores one block of cells at `score`."""
    monkeypatch.delenv("CALIBRATION_PATH", raising=False)
    monkeypatch.delenv("CALIBRATION_REQUIRED", raising=False)
    monkeypatch.setattr(app, "file_sha256", lambda _path: "0" * 64)

    def run(score, thr=0.5):
        prob = np.zeros((SIZE, SIZE), dtype=np.float32)
        prob[24:40, 24:40] = score

        def rollout(lat, lon, *, debug_sink, crop_frac, **_kwargs):
            debug_sink["dyn"] = np.zeros((3, len(app.MODEL_DYNAMIC_ORDER), SIZE, SIZE), dtype=np.float32)
            debug_sink["stat"] = np.zeros((app.MODEL_CS, SIZE, SIZE), dtype=np.float32)
            debug_sink["base_time"] = dt.datetime(2026, 10, 4, 3, tzinfo=dt.timezone.utc)
            debug_sink["static_summary"] = {}
            bounds = tile_bounds_lonlat(lonlat_to_tile(lon, lat))
            steps = [{"index": i, "lead_hours": 24 * (i + 1), "label": f"{i + 1} day", "prob": prob.copy()}
                     for i in range(2)]
            return bounds, app._build_crop_window(prob.shape, bounds, lat, lon, crop_frac), steps

        monkeypatch.setattr(app, "_rollout_multistep_predictions", rollout)
        response = TestClient(app.app).get(
            "/predict_multistep", params={"lat": LAT, "lon": LON, "steps": 2, "thr": thr})
        assert response.status_code == 200, response.text
        return response.json()["scene"]["forecast"]["features"]

    return run


def test_cells_below_the_decision_threshold_get_no_band(forecast_bands):
    # 0.3 clears the 0.10 display floor but not the 0.5 decision threshold.
    assert forecast_bands(0.3) == []


def test_cells_above_the_decision_threshold_are_banded(forecast_bands):
    bands = forecast_bands(0.7)

    assert bands
    assert {band["properties"]["threshold"] for band in bands} == {0.5}


def test_bands_follow_a_threshold_override(forecast_bands):
    assert forecast_bands(0.3, thr=0.2)
