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
