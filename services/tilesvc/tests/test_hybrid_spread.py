"""
The hybrid exists because the engines do not share a scale: the learned model's
useful range sits near 1e-3 and rothermel's near 1e-1, so averaging their
probabilities is decided entirely by rothermel. These pin the properties that
make rank averaging the right combiner.
"""

import numpy as np
import pytest

from services.tilesvc.hybrid_spread import (
    HYBRID_THRESHOLD,
    combine_fields,
    hybrid_rollout,
    percentile_ranks,
)


def test_ranks_span_zero_to_one_in_score_order():
    field = np.array([[0.4, 0.1], [0.3, 0.2]], dtype=np.float32)
    ranks = percentile_ranks(field)
    assert ranks.min() == pytest.approx(0.0)
    assert ranks.max() == pytest.approx(1.0)
    assert ranks[0, 1] < ranks[1, 1] < ranks[1, 0] < ranks[0, 0]


def test_ties_share_the_average_position():
    """A field that is constant over a region must not have an arbitrary order
    imposed on it - physics engines produce large flat zero regions."""
    ranks = percentile_ranks(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert ranks[0] == ranks[1] == ranks[2]
    assert ranks[3] == pytest.approx(1.0)
    assert ranks[0] == pytest.approx(1.0 / 3.0)


def test_cells_outside_the_mask_are_not_ranked():
    field = np.array([[9.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    mask = np.array([[False, True], [True, True]])
    ranks = percentile_ranks(field, mask=mask)
    assert ranks[0, 0] == 0.0, "masked cell must not win the ranking"
    assert ranks[1, 1] == pytest.approx(1.0), "highest unmasked score takes the top rank"


def test_combining_is_invariant_to_each_engine_s_scale():
    """The reason for ranking at all. Scaling one engine by 1000x must not
    change the combined ordering, or the blend is just that engine."""
    a = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
    b = np.array([[0.4, 0.3], [0.2, 0.1]], dtype=np.float32)
    baseline = combine_fields([a, b])
    rescaled = combine_fields([a * 1000.0, b])
    np.testing.assert_allclose(baseline, rescaled, atol=1e-6)


def test_a_weak_engine_cannot_be_drowned_out_by_a_confident_one():
    """Averaging raw probabilities would let the larger-magnitude engine
    decide every cell; averaging ranks gives each an equal vote."""
    timid = np.array([0.001, 0.002, 0.003], dtype=np.float32)   # ascending
    loud = np.array([0.9, 0.5, 0.1], dtype=np.float32)          # descending
    combined = combine_fields([timid, loud])
    # Opposite orderings and equal weight means the result is flat, not a copy
    # of the loud engine's descending order.
    assert combined[0] == pytest.approx(combined[2], abs=1e-6)
    assert not np.all(np.diff(combined) < 0)


def test_a_single_engine_passes_through_as_its_own_ranks():
    field = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    np.testing.assert_allclose(combine_fields([field]), percentile_ranks(field))


def test_no_engines_is_an_error_not_an_empty_field():
    with pytest.raises(ValueError):
        combine_fields([])
    with pytest.raises(ValueError):
        hybrid_rollout({})


def test_engines_that_ran_different_step_counts_are_refused():
    """Step i of one engine is only comparable with step i of another."""
    short = [{"prob": np.zeros((2, 2), np.float32)}]
    long = [{"prob": np.zeros((2, 2), np.float32)}] * 3
    with pytest.raises(ValueError, match="step count"):
        hybrid_rollout({"a": short, "b": long})


def test_rollout_combines_each_step_and_records_its_members():
    steps = 3
    rollouts = {
        "learned": [{"prob": np.full((4, 4), 0.001 * (i + 1), np.float32)} for i in range(steps)],
        "rothermel": [{"prob": np.full((4, 4), 0.2 * (i + 1), np.float32)} for i in range(steps)],
    }
    combined = hybrid_rollout(rollouts)
    assert len(combined) == steps
    assert combined[0]["engines"] == ["learned", "rothermel"]
    assert combined[0]["prob"].shape == (4, 4)
    assert combined[0]["prob"].dtype == np.float32


def test_the_documented_operating_point_is_a_rank_not_a_probability():
    """0.72 is a percentile in the combined ranking. A value outside [0, 1]
    would mean someone had reinterpreted the field as a probability."""
    assert 0.0 < HYBRID_THRESHOLD < 1.0
