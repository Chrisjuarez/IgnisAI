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


class TestServingIntegration:
    """The hybrid is an optional extra layer. Nothing about it may fail a
    prediction — that is the whole contract with the request path."""

    @staticmethod
    def _learned(steps=3, size=8):
        return [{"prob": np.random.default_rng(i).random((size, size)).astype(np.float32)}
                for i in range(steps)]

    def test_a_missing_fuel_raster_returns_no_layer_and_says_why(self, tmp_path, monkeypatch):
        from services.tilesvc.grid import lonlat_to_tile
        from services.tilesvc.hybrid_spread import build_hybrid_rollout

        monkeypatch.delenv("FBFM40_PATH", raising=False)
        monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path / "empty"))
        monkeypatch.chdir(tmp_path)

        rollout, status = build_hybrid_rollout(
            self._learned(),
            tile=lonlat_to_tile(-118.555, 34.078),
            observed_fire=np.zeros((64, 64), np.float32),
            wind_series=[(3.0, -2.0)] * 3,
            steps=3,
            step_hours=24,
        )
        assert rollout is None, "no raster means no layer, not a wrong layer"
        assert status["available"] is False
        assert status["reason"] == "fuel_raster_missing"
        assert "fetch-fuel" in status["detail"]

    def test_an_engine_that_raises_is_contained(self, monkeypatch):
        """A physics engine blowing up must cost the hybrid layer, not the
        forecast the user asked for."""
        from services.tilesvc import hybrid_spread
        from services.tilesvc.grid import lonlat_to_tile

        monkeypatch.setattr(
            "services.tilesvc.fuel_raster.fuel_codes_for_tile",
            lambda tile: np.full((64, 64), 142, np.int16),
        )
        monkeypatch.setattr(
            "services.tilesvc.physics_spread.physics_rollout",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine exploded")),
        )
        rollout, status = hybrid_spread.build_hybrid_rollout(
            self._learned(),
            tile=lonlat_to_tile(-118.555, 34.078),
            observed_fire=np.zeros((64, 64), np.float32),
            wind_series=[(3.0, -2.0)] * 3,
            steps=3,
            step_hours=24,
        )
        assert rollout is None
        assert status["available"] is False
        assert status["reason"] == "hybrid_engine_error"
        assert "engine exploded" in status["detail"]

    def test_the_status_declares_the_field_is_an_uncalibrated_rank(self, monkeypatch):
        """Downstream must never feed this through the isotonic calibration,
        which was fitted on this model's probabilities."""
        from services.tilesvc import hybrid_spread
        from services.tilesvc.grid import lonlat_to_tile

        monkeypatch.setattr(
            "services.tilesvc.fuel_raster.fuel_codes_for_tile",
            lambda tile: np.full((64, 64), 142, np.int16),
        )
        rollout, status = hybrid_spread.build_hybrid_rollout(
            self._learned(steps=2, size=64),
            tile=lonlat_to_tile(-118.555, 34.078),
            observed_fire=np.zeros((64, 64), np.float32),
            wind_series=[(3.0, -2.0)] * 3,
            steps=2,
            step_hours=24,
        )
        assert status["available"] is True
        assert status["units"] == "percentile_rank"
        assert status["calibrated"] is False
        assert status["threshold"] == HYBRID_THRESHOLD
        assert len(rollout) == 2
        assert rollout[0]["prob"].max() <= 1.0

    def test_wind_series_reads_the_last_frames_in_channel_order(self):
        from services.tilesvc.hybrid_spread import wind_series_from_dyn

        order = ["fire_t", "u", "v", "gust"]
        dyn = np.zeros((6, 4, 2, 2), np.float32)
        for frame in range(6):
            dyn[frame, 1] = frame          # u
            dyn[frame, 2] = -frame         # v
        series = wind_series_from_dyn(dyn, order, frames=3)
        assert series == [(3.0, -3.0), (4.0, -4.0), (5.0, -5.0)]

    def test_wind_series_handles_fewer_frames_than_requested(self):
        from services.tilesvc.hybrid_spread import wind_series_from_dyn

        dyn = np.zeros((2, 4, 2, 2), np.float32)
        assert len(wind_series_from_dyn(dyn, ["fire_t", "u", "v", "gust"], frames=3)) == 2
