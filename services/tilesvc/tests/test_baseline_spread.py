import math

import numpy as np
import pytest

from services.tilesvc import baseline_spread as bs
from services.tilesvc.grid import SIZE

CENTRE = (SIZE // 2, SIZE // 2)


def _source_at_centre():
    m = np.zeros((SIZE, SIZE), dtype=np.float32)
    m[CENTRE] = 1.0
    return m


def _centroid(field, threshold=0.1):
    idx = np.argwhere(field >= threshold)
    return idx.mean(axis=0) if idx.size else None


def test_fire_grows_downwind_not_upwind():
    # The whole reason this baseline exists: the learned model does the opposite.
    field = bs.spread_field(_source_at_centre(), u_ms=8.0, v_ms=0.0, hours=24)
    r, c = _centroid(field)

    assert c > CENTRE[1], "an easterly wind must push the fire east"
    assert abs(r - CENTRE[0]) < 2, "and not meaningfully north or south"


def test_a_northward_wind_pushes_north():
    field = bs.spread_field(_source_at_centre(), u_ms=0.0, v_ms=8.0, hours=24)
    r, _ = _centroid(field)

    # Rows increase southward, so north is a smaller row index.
    assert r < CENTRE[0]


def test_stronger_wind_reaches_further():
    weak = bs.spread_field(_source_at_centre(), u_ms=2.0, v_ms=0.0, hours=24)
    strong = bs.spread_field(_source_at_centre(), u_ms=12.0, v_ms=0.0, hours=24)

    assert (strong >= 0.1).sum() > (weak >= 0.1).sum()


def test_the_fire_stretches_along_the_wind():
    field = bs.spread_field(_source_at_centre(), u_ms=10.0, v_ms=0.0, hours=24)
    burning = np.argwhere(field >= 0.1)
    along = burning[:, 1].ptp()
    across = burning[:, 0].ptp()

    assert along > across, "a wind-driven fire is longer than it is wide"


def test_it_backs_into_the_wind_only_slightly():
    field = bs.spread_field(_source_at_centre(), u_ms=10.0, v_ms=0.0, hours=24)
    burning = np.argwhere(field >= 0.1)
    downwind = burning[:, 1].max() - CENTRE[1]
    upwind = CENTRE[1] - burning[:, 1].min()

    assert upwind < downwind, "a backing fire must not outrun the head"
    assert upwind >= 0, "but some backing is expected"


def test_calm_air_grows_a_circle_rather_than_inventing_a_direction():
    field = bs.spread_field(_source_at_centre(), u_ms=0.05, v_ms=-0.05, hours=24)
    burning = np.argwhere(field >= 0.1)

    assert abs(burning[:, 0].ptp() - burning[:, 1].ptp()) <= 2


def test_growth_accumulates_over_the_horizon():
    m = _source_at_centre()
    day1 = bs.spread_field(m, u_ms=6.0, v_ms=0.0, hours=24)
    day3 = bs.spread_field(m, u_ms=6.0, v_ms=0.0, hours=72)

    assert (day3 >= 0.1).sum() > (day1 >= 0.1).sum()


def test_a_wide_front_is_not_collapsed_to_a_point():
    front = np.zeros((SIZE, SIZE), dtype=np.float32)
    front[CENTRE[0] - 6:CENTRE[0] + 6, CENTRE[1]] = 1.0

    field = bs.spread_field(front, u_ms=8.0, v_ms=0.0, hours=24)
    burning = np.argwhere(field >= 0.1)

    assert burning[:, 0].ptp() >= 12, "the whole front should advance, not its centre"


def test_no_source_produces_no_fire_rather_than_a_guess():
    assert not (bs.spread_field(None, u_ms=8.0, v_ms=0.0, hours=24) > 0).any()
    empty = np.zeros((SIZE, SIZE), dtype=np.float32)
    assert not (bs.spread_field(empty, u_ms=8.0, v_ms=0.0, hours=24) > 0).any()


def test_an_ignition_point_seeds_growth_when_nothing_has_burned():
    field = bs.spread_field(None, u_ms=8.0, v_ms=0.0, hours=24, ignition_rc=CENTRE)

    assert (field >= 0.1).any()


def test_rollout_matches_the_model_rollout_shape():
    rollout = bs.baseline_rollout(_source_at_centre(), u_ms=6.0, v_ms=2.0,
                                  steps=3, step_hours=24)

    assert [s["lead_hours"] for s in rollout] == [24, 48, 72]
    assert all(s["prob"].shape == (SIZE, SIZE) for s in rollout)
    assert all({"index", "lead_hours", "label", "prob"} <= set(s) for s in rollout)


def test_rate_and_shape_respond_to_wind_monotonically():
    assert bs.rate_of_spread_m_per_h(10) > bs.rate_of_spread_m_per_h(2)
    assert bs.length_to_breadth(10) > bs.length_to_breadth(2)
    assert bs.length_to_breadth(100) <= bs.LB_MAX, "elongation must stay bounded"
