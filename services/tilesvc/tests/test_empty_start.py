"""A forecast with no fire to start from.

With no detection in the latest day and no ignition seed, the model has
nothing to grow and draws artifacts along the tile edge instead (McCully: 36,000
ac of bands from an empty input). /predict_multistep answers "no forecast".
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("fastapi")

from services.tilesvc import app  # noqa: E402
from services.tilesvc.grid import SIZE, lonlat_to_tile, tile_bounds_lonlat  # noqa: E402
from services.tilesvc.static_catalog import InputUnavailable  # noqa: E402

LON, LAT = -118.40183, 34.561835
FIRE = list(app.MODEL_DYNAMIC_ORDER).index("fire_t")
OK_WEATHER = {"status": "ok", "source": "hrrr_on_demand", "reason": ""}


def _inputs(*fire_cells):
    """Three input days; `fire_cells` are (day, row, col) with a detection."""
    dyn = np.zeros((3, len(app.MODEL_DYNAMIC_ORDER), SIZE, SIZE), dtype=np.float32)
    for day, row, col in fire_cells:
        dyn[day, FIRE, row, col] = 0.4
    return dyn


@pytest.fixture
def forecast(monkeypatch):
    """Runs /predict_multistep on given inputs; records which engines ran."""
    monkeypatch.delenv("CALIBRATION_PATH", raising=False)
    monkeypatch.delenv("CALIBRATION_REQUIRED", raising=False)
    monkeypatch.setattr(app, "file_sha256", lambda _path: "0" * 64)
    ran = []

    def prepare_with(dyn):
        def prepare(lat, lon, Tseq, *, return_weather=False, **_kwargs):
            prepared = (
                dyn,
                np.zeros((app.MODEL_CS, SIZE, SIZE), dtype=np.float32),
                tile_bounds_lonlat(lonlat_to_tile(lon, lat)),
                dt.datetime(2026, 10, 4, 12, tzinfo=dt.timezone.utc),
                {},
            )
            return prepared + ([OK_WEATHER] * len(dyn),) if return_weather else prepared
        return prepare

    def predict(dyn, stat, **_kwargs):
        ran.append("ignis")
        return np.zeros((SIZE, SIZE), dtype=np.float32)

    def baseline(observed, **kwargs):
        ran.append("downwind")
        return [{"index": i, "lead_hours": 24 * (i + 1), "label": f"day {i + 1}",
                 "prob": np.zeros((SIZE, SIZE), dtype=np.float32)} for i in range(kwargs["steps"])]

    weather = {name: np.zeros((SIZE, SIZE), dtype=np.float32)
               for name in ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")}
    monkeypatch.setattr(app, "_predict_probability_from_inputs", predict)
    monkeypatch.setattr(app, "baseline_rollout", baseline)
    monkeypatch.setattr(app, "fetch_weather_with_quality", lambda *a, **k: (dict(weather), OK_WEATHER))

    def run(dyn, model="ignis", debug=None):
        monkeypatch.setattr(app, "_prepare_prediction_inputs_with_summary", prepare_with(dyn))
        # Called directly, so every Query default is passed explicitly.
        return app.predict_multistep(
            lat=LAT, lon=LON, model=model, Tseq=3, steps=2, step_hours=24, thr=None,
            display_floor=None, crop_frac=0.5, ignition=False, date=None, debug=debug,
            burned_area=None,
        )

    run.ran = ran
    return run


@pytest.mark.parametrize("model", ["ignis", "downwind"])
def test_no_fire_in_the_latest_day_is_no_forecast(forecast, model):
    with pytest.raises(InputUnavailable) as raised:
        forecast(_inputs(), model=model)

    assert raised.value.reason == "no_recent_fire"
    assert forecast.ran == [], "neither engine should run without a fire to grow"


def test_fire_seen_only_on_earlier_days_is_no_forecast(forecast):
    # The rollout grows from the latest day; a fire last seen two days ago has
    # no current position to grow from.
    with pytest.raises(InputUnavailable):
        forecast(_inputs((0, 30, 30), (1, 31, 31)))
    assert forecast.ran == []


def test_one_detection_in_the_latest_day_is_enough(forecast):
    response = forecast(_inputs((2, 32, 32)))

    assert forecast.ran == ["ignis", "ignis"]
    assert len(response["steps"]) == 2


def test_the_plumbing_test_still_draws_without_a_fire(forecast):
    response = forecast(_inputs(), debug="solid")

    assert len(response["steps"]) == 2
    assert forecast.ran == []
