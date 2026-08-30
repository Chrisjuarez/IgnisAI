"""Checks on the Rothermel implementation.

A spread model that compiles but is wrong is worse than none, because its
output looks authoritative. These assert behaviour a fire behaviour analyst
would recognise, plus the magnitudes published for standard fuels.
"""
import math

import pytest

from services.tilesvc.rothermel import (
    length_to_breadth,
    midflame_wind,
    spread_rate_grid,
    spread_rate_m_per_h,
)

# Fire-season conditions: dry fine dead fuel, cured grass, live woody in
# drought. Curing matters - without the transfer the grass models are damped by
# live moisture and come out slower than conifer litter.
DRY = dict(dead_moisture=0.06, live_moisture=0.60, herb_moisture=0.30)


def ros(code, **kw):
    return spread_rate_m_per_h(code, **{**DRY, **kw})


def test_non_burnable_fuels_never_spread():
    for code in (91, 93, 98, 99):          # urban, agriculture, water, barren
        assert ros(code, midflame_wind_ms=20.0, slope_fraction=0.5) == 0.0


def test_unknown_fuel_code_is_treated_as_non_burnable():
    assert ros(9999, midflame_wind_ms=10.0) == 0.0


def test_spread_increases_with_wind():
    calm = ros(102, midflame_wind_ms=0.0)
    breezy = ros(102, midflame_wind_ms=2.0)
    gale = ros(102, midflame_wind_ms=8.0)

    assert 0 < calm < breezy < gale


def test_spread_increases_up_a_steeper_slope():
    flat = ros(102, midflame_wind_ms=1.0, slope_fraction=0.0)
    moderate = ros(102, midflame_wind_ms=1.0, slope_fraction=0.3)
    steep = ros(102, midflame_wind_ms=1.0, slope_fraction=0.7)

    assert flat < moderate < steep


def test_wetter_dead_fuel_slows_the_fire():
    dry = spread_rate_m_per_h(102, dead_moisture=0.04, live_moisture=0.6, midflame_wind_ms=3.0)
    damp = spread_rate_m_per_h(102, dead_moisture=0.15, live_moisture=0.6, midflame_wind_ms=3.0)

    assert damp < dry


def test_fire_goes_out_above_the_moisture_of_extinction():
    # GR2's dead fuel moisture of extinction is 15%. Past it the damping
    # polynomial is zero, and a fire that is out must report exactly no spread
    # rather than the 1e-13 that rounding leaves behind.
    assert spread_rate_m_per_h(102, dead_moisture=0.30, live_moisture=0.60,
                               midflame_wind_ms=3.0) == 0.0


def test_grass_outruns_timber_litter_in_the_same_weather():
    grass = ros(102, midflame_wind_ms=4.0)          # GR2
    litter = ros(183, midflame_wind_ms=4.0)         # TL3 conifer litter

    assert grass > litter * 2, "grass fires are far faster than litter fires"


def test_chaparral_is_an_order_of_magnitude_faster_than_litter():
    # SH5 chaparral is what carried Palisades; TL3 is conifer litter. The gap
    # between them is the clearest qualitative signature of the model, and it
    # survives whatever the absolute calibration turns out to be.
    chaparral = ros(145, midflame_wind_ms=2.0)
    litter = ros(183, midflame_wind_ms=2.0)

    assert chaparral > 10 * litter


def test_spread_rates_are_within_an_order_of_magnitude_of_field_behaviour():
    # Deliberately loose. These bounds catch a model that is broken by orders
    # of magnitude - the failure mode that matters - and make no claim of
    # agreement with BehavePlus, which this has not yet been checked against.
    # See MODULE NOTE in rothermel.py: quantitative validation is outstanding.
    at_4mph = {code: ros(code, midflame_wind_ms=1.79) for code in (102, 122, 145, 183)}

    assert 100 < at_4mph[102] < 5_000, f"grass {at_4mph[102]:.0f} m/h"
    assert 100 < at_4mph[122] < 6_000, f"grass-shrub {at_4mph[122]:.0f} m/h"
    assert 200 < at_4mph[145] < 12_000, f"chaparral {at_4mph[145]:.0f} m/h"
    assert 10 < at_4mph[183] < 500, f"litter {at_4mph[183]:.0f} m/h"


def test_curing_transfers_grass_from_live_to_dead():
    from services.tilesvc.rothermel import curing_fraction

    assert curing_fraction(0.30) == 1.0        # fully cured
    assert curing_fraction(1.20) == 0.0        # fully green
    assert 0.0 < curing_fraction(0.75) < 1.0

    green = spread_rate_m_per_h(102, dead_moisture=0.06, live_moisture=0.60,
                                herb_moisture=1.20, midflame_wind_ms=2.0)
    cured = spread_rate_m_per_h(102, dead_moisture=0.06, live_moisture=0.60,
                                herb_moisture=0.30, midflame_wind_ms=2.0)

    assert cured > green * 2, "cured grass must carry fire far better than green"


def test_heavy_fuels_are_not_dried_to_the_fine_fuel_value():
    # 100-hour fuels hold more water than 1-hour fuels. Applying the fine
    # moisture to everything made the heavy shrub models far too fast.
    from services.tilesvc.rothermel import MOISTURE_STEP_100H, MOISTURE_STEP_10H

    assert MOISTURE_STEP_10H > 0 and MOISTURE_STEP_100H > MOISTURE_STEP_10H


def test_midflame_wind_is_reduced_from_open_wind():
    # Rothermel takes the wind at the flame, not the 10 m observation.
    assert midflame_wind(10.0, 102) == pytest.approx(3.6)     # grass
    assert midflame_wind(10.0, 183) == pytest.approx(2.0)     # timber litter
    assert midflame_wind(10.0, 91) == 0.0                     # non-burnable


def test_ellipse_elongates_with_wind_and_is_bounded():
    assert length_to_breadth(0.0) == pytest.approx(1.0)
    assert length_to_breadth(5.0) > length_to_breadth(1.0)
    assert length_to_breadth(100.0) <= 8.0


def test_grid_evaluates_each_fuel_and_zeroes_non_burnable():
    import numpy as np

    codes = np.array([[102, 102, 91], [183, 122, 98]], dtype=np.int32)
    out = spread_rate_grid(codes, dead_moisture=0.06, live_moisture=0.6, wind_ms=5.0)

    assert out[0, 2] == 0.0 and out[1, 2] == 0.0, "non-burnable cells must not spread"
    assert out[0, 0] > 0 and out[1, 0] > 0
    assert out[0, 0] > out[1, 0], "grass should outrun litter in the grid path too"


def test_characteristic_sav_matches_published_fuel_model_values():
    """The fuel bed maths must reproduce Scott & Burgan's published SAV.

    Independent of weather, so it isolates a mis-transcribed load or depth from
    an error in the spread equations. Three models hitting their published
    values is the strongest check available without a reference implementation.
    """
    from tools.validate_rothermel import characteristic_sav

    assert characteristic_sav(101) == pytest.approx(2054, rel=0.02)   # GR1
    assert characteristic_sav(102) == pytest.approx(1820, rel=0.02)   # GR2
    assert characteristic_sav(142) == pytest.approx(1672, rel=0.02)   # SH2


def test_effective_wind_limit_stops_runaway_spread():
    """Rothermel's wind factor is unbounded; Andrews (2013) caps it near 0.9*I_R.

    Without the cap a Santa Ana produces spread rates that are not physical.
    Doubling an already-strong wind must not double the rate.
    """
    strong = ros(102, midflame_wind_ms=8.0)
    stronger = ros(102, midflame_wind_ms=16.0)

    assert stronger < strong * 1.5, "wind factor is running away past the limit"
