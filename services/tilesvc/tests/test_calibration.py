import json

import numpy as np
import pytest

from services.tilesvc import calibration
from services.tilesvc.static_catalog import InputUnavailable


def reset_calibration_cache():
    calibration._calibration_cache = None


def test_missing_required_calibration_refuses_production_display(monkeypatch):
    reset_calibration_cache()
    monkeypatch.delenv("CALIBRATION_PATH", raising=False)
    monkeypatch.setenv("CALIBRATION_REQUIRED", "1")

    with pytest.raises(InputUnavailable, match="CALIBRATION_PATH"):
        calibration.load_calibration(model_sha256="abc")


def test_isotonic_calibration_maps_score_and_risk_classes(tmp_path, monkeypatch):
    reset_calibration_cache()
    artifact = {
        "method": "isotonic",
        "model_sha256": "model-hash",
        "points": [[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]],
        "risk_breaks": [
            {"class": "low", "upper": 0.1},
            {"class": "medium", "upper": 0.35},
            {"class": "high", "upper": 0.7},
            {"class": "extreme", "upper": 1.01},
        ],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setenv("CALIBRATION_PATH", str(path))
    monkeypatch.setenv("CALIBRATION_REQUIRED", "1")

    score, risk, meta = calibration.calibrate_probability(
        np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
        model_sha256="model-hash",
    )

    assert np.allclose(score, [[0.0, 0.2, 1.0]])
    assert risk.tolist() == [["low", "medium", "extreme"]]
    assert meta["method"] == "isotonic"


def test_calibration_hash_mismatch_is_unavailable(tmp_path, monkeypatch):
    reset_calibration_cache()
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"model_sha256": "expected", "points": [[0, 0], [1, 1]]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CALIBRATION_PATH", str(path))

    with pytest.raises(InputUnavailable, match="does not match"):
        calibration.load_calibration(model_sha256="actual")


def test_calibration_hash_accepts_sha256_prefix(tmp_path, monkeypatch):
    reset_calibration_cache()
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"model_sha256": "sha256:ABC123", "points": [[0, 0], [1, 1]]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CALIBRATION_PATH", str(path))

    cal = calibration.load_calibration(model_sha256="abc123")

    assert cal["model_sha256"] == "sha256:ABC123"
