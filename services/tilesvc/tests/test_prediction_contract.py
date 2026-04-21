import numpy as np

from services.tilesvc.prediction_contract import next_fire_from_delta, risk_class_summary


def test_delta_reconstructs_next_fire_from_observed_or_thresholded_new_burn():
    observed = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    prob_new = np.asarray([[0.0, 0.7], [0.2, 0.9]], dtype=np.float32)

    out = next_fire_from_delta(prob_new, observed, 0.5, target_mode="delta")

    assert np.array_equal(out, np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32))


def test_mask_mode_uses_thresholded_probability_without_persistence_union():
    observed = np.ones((2, 2), dtype=np.float32)
    prob_next = np.asarray([[0.1, 0.6], [0.4, 0.9]], dtype=np.float32)

    out = next_fire_from_delta(prob_next, observed, 0.5, target_mode="mask")

    assert np.array_equal(out, np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))


def test_risk_class_summary_reports_class_fractions():
    risk = np.asarray([["low", "medium"], ["high", "extreme"]], dtype=object)

    assert risk_class_summary(risk) == {
        "low": 0.25,
        "medium": 0.25,
        "high": 0.25,
        "extreme": 0.25,
    }
