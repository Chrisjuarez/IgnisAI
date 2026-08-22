import hashlib
from pathlib import Path

import numpy as np

from services.tilesvc import ml_runtime


def test_runtime_imports_bundled_derived_feature_helpers(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("IGNIS_ML_SOURCE_PATH", str(repo_root / "ignis_ml"))

    names, append_derived_features, expected_channel_count, module_name = ml_runtime.import_feature_helpers()

    assert module_name == "src.data.features"
    assert names == (
        "wind_speed",
        "wind_dir_cos",
        "wind_dir_sin",
        "temp_delta",
        "q_delta",
        "days_since_fire",
    )
    assert expected_channel_count(["fire_t", "u", "v", "gust", "tempC", "q", "precip"], None) == 13

    x_dyn = np.zeros((6, 7, 2, 2), dtype=np.float32)
    x_dyn[:, 1] = 0.5
    x_dyn[:, 2] = 0.5
    out, order = append_derived_features(
        x_dyn,
        dyn_order=["fire_t", "u", "v", "gust", "tempC", "q", "precip"],
    )

    assert out.shape == (6, 13, 2, 2)
    assert order[-6:] == list(names)


def test_file_sha256_reuses_digest_until_file_changes(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    target.write_bytes(b"first")
    ml_runtime._digest_by_path.clear()

    hash_calls = 0
    real_sha256 = hashlib.sha256

    def counting_sha256(*args, **kwargs):
        nonlocal hash_calls
        hash_calls += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(ml_runtime.hashlib, "sha256", counting_sha256)

    first = ml_runtime.file_sha256(target)
    assert hash_calls == 1

    assert ml_runtime.file_sha256(target) == first
    assert hash_calls == 1, "unchanged file should not be re-hashed"

    target.write_bytes(b"second content")
    assert ml_runtime.file_sha256(target) != first
    assert hash_calls == 2, "changed file must be re-hashed"


def test_file_sha256_returns_none_for_missing_file(tmp_path):
    assert ml_runtime.file_sha256(tmp_path / "absent.pt") is None
