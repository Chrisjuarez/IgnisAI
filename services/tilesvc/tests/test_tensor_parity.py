import os
import sys
from pathlib import Path

import numpy as np
import pytest


BASE_DYNAMIC_ORDER = ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]
STATIC_ORDER = [
    "elev",
    "slope",
    "aspect_cos",
    "aspect_sin",
    "ndvi",
    "bi",
    "erc",
    "pdsi",
    "chili",
    "impervious",
    "water",
    "population",
    "fuel1",
    "fuel2",
    "fuel3",
]


def test_ndws_dataset_and_live_audit_tensor_parity():
    """
    CI gate for the migration hazard that matters most: channel/order/scaling drift.

    Set IGNIS_PARITY_NDWS_TILE to a known raw NDWS .npz and IGNIS_PARITY_AUDIT_NPZ
    to the audit NPZ emitted by /input_audit?include_npz=true for the same tile.
    Local runs skip unless IGNIS_STRICT_PARITY=1 is set.
    """
    strict = os.getenv("IGNIS_STRICT_PARITY", "0").strip().lower() in {"1", "true", "yes", "on"}
    ndws_tile = os.getenv("IGNIS_PARITY_NDWS_TILE")
    audit_npz = os.getenv("IGNIS_PARITY_AUDIT_NPZ")
    if not ndws_tile or not audit_npz:
        if strict:
            pytest.fail("IGNIS_PARITY_NDWS_TILE and IGNIS_PARITY_AUDIT_NPZ are required")
        pytest.skip("parity fixture env vars not configured")

    ml_source = os.getenv("IGNIS_ML_SOURCE_PATH") or "/Users/chrisjuarez/Downloads/ignis_ml_nautilus"
    if Path(ml_source).exists() and ml_source not in sys.path:
        sys.path.insert(0, ml_source)

    from src.data.dataset import NpzTileDataset

    ds = NpzTileDataset(
        [ndws_tile],
        expected_dyn_order=BASE_DYNAMIC_ORDER,
        expected_stat_order=STATIC_ORDER,
        seq_len=6,
        normalize=True,
        compute_slope_aspect=True,
        derived_features_enable=True,
        target_mode="delta",
    )
    x_dyn_train, x_stat_train, *_ = ds[0]
    with np.load(audit_npz, allow_pickle=True) as audit:
        x_dyn_live = audit["x_dyn"].astype(np.float32)
        x_stat_live = audit["x_stat"].astype(np.float32)

    assert tuple(x_dyn_train.shape) == (6, 13, 64, 64)
    assert tuple(x_stat_train.shape) == (15, 64, 64)
    assert x_dyn_live.shape == x_dyn_train.shape
    assert x_stat_live.shape == x_stat_train.shape
    assert np.allclose(x_dyn_live, x_dyn_train.numpy(), atol=1e-6)
    assert np.allclose(x_stat_live, x_stat_train.numpy(), atol=1e-6)
