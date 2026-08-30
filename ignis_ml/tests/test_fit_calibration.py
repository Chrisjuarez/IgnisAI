"""Tests for the calibration fit.

The point of a calibration is that a displayed 0.7 means seven in ten. These
check the maths that claim can rest on, using data whose answer is known.
"""
import numpy as np
import pytest

from ignis_ml.scripts.fit_calibration import (
    brier,
    brier_skill_score,
    curve_from_pairs,
    expected_calibration_error,
    isotonic_fit,
    reliability_bins,
)


def test_isotonic_fit_is_monotone():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < x).astype(float)

    _, fitted = isotonic_fit(x, y)

    assert np.all(np.diff(fitted) >= -1e-9), "isotonic output must never decrease"


def test_calibration_corrects_a_model_that_doubles_its_probabilities():
    # Truth is p; the model reports 2p capped at 1. A correct calibration maps
    # the reported value back toward the frequency actually observed.
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 0.5, 40_000)
    label = (rng.uniform(0, 1, truth.size) < truth).astype(float)
    reported = np.clip(truth * 2, 0, 1)

    points = curve_from_pairs(reported, label)
    raw = np.array([p[0] for p in points])
    mapped_curve = np.array([p[1] for p in points])
    mapped = np.interp(reported, raw, mapped_curve)

    before = expected_calibration_error(reliability_bins(reported, label))
    after = expected_calibration_error(reliability_bins(mapped, label))

    assert after < before / 2, f"calibration should shrink ECE: {before:.4f} -> {after:.4f}"


def test_brier_skill_is_zero_for_predicting_the_base_rate():
    label = np.array([1.0] * 30 + [0.0] * 70)
    always_base = np.full(label.size, label.mean())

    assert brier_skill_score(always_base, label) == pytest.approx(0.0, abs=1e-9)


def test_brier_skill_is_positive_for_a_model_that_knows_something():
    label = np.array([1.0] * 50 + [0.0] * 50)
    informed = np.array([0.9] * 50 + [0.1] * 50)

    assert brier_skill_score(informed, label) > 0.5


def test_brier_skill_is_negative_for_a_model_that_is_confidently_wrong():
    label = np.array([1.0] * 50 + [0.0] * 50)
    backwards = np.array([0.1] * 50 + [0.9] * 50)

    assert brier_skill_score(backwards, label) < 0


def test_perfect_forecast_scores_zero_brier():
    label = np.array([1.0, 0.0, 1.0, 0.0])

    assert brier(label, label) == pytest.approx(0.0)


def test_reliability_bins_report_the_observed_frequency():
    prob = np.array([0.05, 0.05, 0.95, 0.95])
    label = np.array([0.0, 0.0, 1.0, 1.0])

    bins = [b for b in reliability_bins(prob, label) if b["n"]]

    assert bins[0]["observed"] == pytest.approx(0.0)
    assert bins[-1]["observed"] == pytest.approx(1.0)


def test_curve_spans_the_whole_range_and_never_decreases():
    rng = np.random.default_rng(2)
    prob = rng.uniform(0, 1, 5_000)
    label = (rng.uniform(0, 1, prob.size) < prob).astype(float)

    points = curve_from_pairs(prob, label)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    assert xs[0] == 0.0 and xs[-1] == 1.0, "curve must cover the full input range"
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])), "curve must be monotone"
    assert all(0.0 <= y <= 1.0 for y in ys)
