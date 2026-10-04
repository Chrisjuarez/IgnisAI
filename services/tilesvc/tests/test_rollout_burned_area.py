"""The rollout's side of the burned area: no new fire there, and none fed back.

Each forecast day is fed into the next as its fire. A burned interior predicted
as "new fire" would therefore not just be drawn wrongly - it would seed the
following day from ground that has nothing left to burn.
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("fastapi")

from services.tilesvc import app  # noqa: E402
from services.tilesvc.grid import SIZE, lonlat_to_tile, tile_bounds_lonlat  # noqa: E402

LON, LAT = -118.40183, 34.561835
MODEL_SCORE = 0.4


@pytest.fixture
def stubbed_model(monkeypatch):
    """The real rollout loop over a model that scores every cell the same."""
    order = list(app.MODEL_DYNAMIC_ORDER)
    dyn = np.zeros((3, len(order), SIZE, SIZE), dtype=np.float32)
    stat = np.zeros((app.MODEL_CS, SIZE, SIZE), dtype=np.float32)
    bounds = tile_bounds_lonlat(lonlat_to_tile(LON, LAT))
    base_time = dt.datetime(2026, 10, 4, 3, tzinfo=dt.timezone.utc)
    fire_given = []

    def predict(current_dyn, _stat, log_label=""):
        fire_given.append(current_dyn[-1, order.index("fire_t")].copy())
        return np.full((SIZE, SIZE), MODEL_SCORE, dtype=np.float32)

    monkeypatch.setattr(app, "MODEL_TARGET_MODE", "delta")
    monkeypatch.setattr(app, "MODEL_AR_FEEDBACK_MODE", "soft")
    monkeypatch.setattr(app, "_prepare_prediction_inputs_with_summary",
                        lambda *args, **kwargs: (dyn, stat, bounds, base_time, {}))
    monkeypatch.setattr(app, "_predict_probability_from_inputs", predict)
    monkeypatch.setattr(app, "fetch_weather_grids", lambda *args, **kwargs: {
        name: np.zeros((SIZE, SIZE), dtype=np.float32)
        for name in ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")
    })
    return fire_given


def _rollout(burned):
    _, _, rollout = app._rollout_multistep_predictions(
        LAT, LON, Tseq=3, steps=3, step_hours=24, crop_frac=0.5,
        ignition=False, ref_time=None, threshold=0.5, burned=burned,
    )
    return rollout


def _burned_block():
    burned = np.zeros((SIZE, SIZE), dtype=bool)
    burned[20:30, 20:30] = True
    return burned


def test_burned_cells_get_no_new_fire_on_any_day(stubbed_model):
    burned = _burned_block()

    for step in _rollout(burned):
        assert (step["prob"][burned] == 0).all()
        assert np.allclose(step["prob"][~burned], MODEL_SCORE)


def test_a_burned_interior_seeds_nothing_into_the_next_day(stubbed_model):
    burned = _burned_block()

    _rollout(burned)

    next_day_fire = stubbed_model[1]
    assert (next_day_fire[burned] == 0).all()
    assert (next_day_fire[~burned] > 0).all()


def test_without_a_burned_area_the_forecast_is_unchanged(stubbed_model):
    for step in _rollout(None):
        assert np.allclose(step["prob"], MODEL_SCORE)
