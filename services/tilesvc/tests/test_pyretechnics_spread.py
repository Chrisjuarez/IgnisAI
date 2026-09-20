"""Tests for the pyretechnics level-set adapter.

Two conventions in this integration are easy to get wrong and silent when
wrong, so they are pinned here: the level set stores phi (negative inside the
fire, so a mask of ones marks everything UNBURNED), and pyretechnics indexes
from the lower-left corner while our rasters are north-up.

Both were wrong in the first version. Neither showed up on a symmetric test
blob - the direction sweep is what caught them.
"""
import math

import numpy as np
import pytest

pytest.importorskip("pyretechnics")

from services.tilesvc.grid import SIZE
from services.tilesvc.pyretechnics_spread import (
    pyretechnics_rollout,
    slope_and_aspect,
    wind_to_pyretechnics,
)

CENTRE = SIZE // 2
GRASS = 102


def burning_blob():
    obs = np.zeros((SIZE, SIZE), dtype=np.float32)
    obs[CENTRE - 1:CENTRE + 2, CENTRE - 1:CENTRE + 2] = 1.0
    return obs


def spread_bearing(step):
    mask = step["prob"] > 0.5
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (math.degrees(math.atan2(cols.mean() - CENTRE, -(rows.mean() - CENTRE))) + 360) % 360


@pytest.mark.parametrize("name,u,v,expected", [
    ("north", 0.0, 10.0, 0.0),
    ("east", 10.0, 0.0, 90.0),
    ("south", 0.0, -10.0, 180.0),
    ("west", -10.0, 0.0, 270.0),
    ("southwest", -7.0, -7.0, 225.0),
])
def test_fire_runs_the_way_the_wind_blows(name, u, v, expected):
    roll = pyretechnics_rollout(burning_blob(), fuel_codes=np.full((SIZE, SIZE), GRASS, np.int16),
                                wind_series=[(u, v)], steps=1, step_hours=12)

    bearing = spread_bearing(roll[0])

    assert bearing is not None, f"{name}: nothing burned"
    assert abs((bearing - expected + 180) % 360 - 180) < 20, f"{name}: spread {bearing:.0f}"


def test_upwind_direction_is_the_reverse_of_our_toward_convention():
    # Ours is the vector the wind blows toward; pyretechnics wants where it
    # comes from.
    speed, upwind = wind_to_pyretechnics(0.0, 10.0)      # blowing toward north

    assert speed == pytest.approx(36.0)                  # 10 m/s -> km/h
    assert upwind == pytest.approx(180.0)                # comes from the south


def test_non_burnable_fuel_stops_the_fire():
    # What is already alight stays alight; the point is that it spreads
    # nowhere, so the burned area never exceeds the seed.
    seed = burning_blob()
    roll = pyretechnics_rollout(seed, fuel_codes=np.full((SIZE, SIZE), 91, np.int16),
                                wind_series=[(-7.0, -7.0)], steps=1, step_hours=24)

    assert roll[0]["prob"].sum() <= seed.sum()


def test_nothing_burning_yields_empty_steps_rather_than_an_error():
    roll = pyretechnics_rollout(np.zeros((SIZE, SIZE), np.float32),
                                fuel_codes=np.full((SIZE, SIZE), GRASS, np.int16),
                                wind_series=[(-7.0, -7.0)], steps=3, step_hours=24)

    assert len(roll) == 3
    assert all(step["prob"].sum() == 0.0 for step in roll)


def test_burned_area_grows_monotonically_with_lead_time():
    roll = pyretechnics_rollout(burning_blob(), fuel_codes=np.full((SIZE, SIZE), GRASS, np.int16),
                                wind_series=[(-5.0, -5.0)], steps=3, step_hours=24)

    areas = [float(step["prob"].sum()) for step in roll]

    assert areas[0] > 0
    assert areas[0] <= areas[1] <= areas[2], "a fire cannot un-burn"


def test_wind_can_turn_across_the_forecast():
    # The reason wind is a series: holding it constant is defensible for one
    # extreme event and wrong for most multi-day fires.
    turning = pyretechnics_rollout(
        burning_blob(), fuel_codes=np.full((SIZE, SIZE), GRASS, np.int16),
        wind_series=[(10.0, 0.0), (0.0, 10.0), (0.0, 10.0)], steps=3, step_hours=24)
    steady = pyretechnics_rollout(
        burning_blob(), fuel_codes=np.full((SIZE, SIZE), GRASS, np.int16),
        wind_series=[(10.0, 0.0)], steps=3, step_hours=24)

    assert spread_bearing(turning[2]) != pytest.approx(spread_bearing(steady[2]), abs=5)


def test_aspect_points_downslope_clockwise_from_north():
    # Elevation rising toward the north: the downslope direction is south.
    rows = np.arange(SIZE, dtype=np.float32)[:, None]
    elevation = np.repeat((SIZE - rows) * 10.0, SIZE, axis=1)

    slope, aspect = slope_and_aspect(elevation)

    assert slope.mean() > 0
    assert abs((float(np.median(aspect)) - 180.0 + 180) % 360 - 180) < 20
