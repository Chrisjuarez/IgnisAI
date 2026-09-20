"""Tests for the WUI fuel treatment.

This module invents burnable fuel where a published model says there is none.
That is a real modelling claim, so the conditions under which it fires - and
the conditions under which it declines to - need to be pinned down.
"""
import numpy as np
import pytest

from services.tilesvc.fuel_models import lookup
from services.tilesvc.wui_fuels import (
    CORE_URBAN_MIN,
    DEVELOPED_CODES,
    apply_wui_fuels,
    wui_summary,
)


def grid(code, imp):
    return np.full((4, 4), code, dtype=np.int16), np.full((4, 4), imp, dtype=np.float32)


def test_developed_land_becomes_burnable():
    codes, imp = grid(91, 0.5)      # suburban density

    out = apply_wui_fuels(codes, imp)

    assert lookup(int(out[0, 0])) is not None, "a suburb must be able to burn"


def test_density_selects_a_heavier_fuel():
    sparse = apply_wui_fuels(*grid(91, 0.10))[0, 0]
    moderate = apply_wui_fuels(*grid(91, 0.50))[0, 0]
    dense = apply_wui_fuels(*grid(91, 0.80))[0, 0]

    assert len({int(sparse), int(moderate), int(dense)}) == 3


def test_continuous_urban_core_is_left_alone():
    # A surface-fire model has nothing useful to say about a city centre, and
    # guessing is worse than declining.
    codes, imp = grid(91, CORE_URBAN_MIN + 0.05)

    out = apply_wui_fuels(codes, imp)

    assert int(out[0, 0]) in DEVELOPED_CODES
    assert lookup(int(out[0, 0])) is None


def test_water_and_barren_are_never_made_burnable():
    for code in (98, 99):
        out = apply_wui_fuels(*grid(code, 0.5))
        assert int(out[0, 0]) == code, "water and rock are genuinely non-flammable"


def test_wildland_fuels_pass_through_untouched():
    codes, imp = grid(145, 0.4)      # chaparral

    assert np.array_equal(apply_wui_fuels(codes, imp), codes)


def test_treatment_is_opt_in():
    codes, imp = grid(91, 0.5)

    assert np.array_equal(apply_wui_fuels(codes, imp, enable=False), codes)


def test_input_is_not_modified():
    codes, imp = grid(91, 0.5)
    before = codes.copy()

    apply_wui_fuels(codes, imp)

    assert np.array_equal(codes, before), "the wildland answer must stay available"


def test_summary_reports_what_changed_and_carries_the_caveat():
    codes, imp = grid(91, 0.5)
    out = apply_wui_fuels(codes, imp)

    s = wui_summary(codes, out)

    assert s["applied"] is True
    assert s["cells_reassigned"] == 16
    assert s["burnable_fraction_before"] == 0.0
    assert s["burnable_fraction_after"] == 1.0
    assert "not a per-building ignition probability" in s["caveat"]
