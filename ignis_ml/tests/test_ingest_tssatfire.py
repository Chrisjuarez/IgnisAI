"""Phase 1 smoke tests — channel assembly, schema, no NaN/Inf, eval metrics.

These run with no external dataset (synthetic FireStack) so they're safe in CI.
Run:  python -m pytest ignis_ml/tests/test_ingest_tssatfire.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ignis_ml.scripts.ingest_ts_satfire import (  # noqa: E402
    V4_DYNAMIC_ORDER,
    V4_STATIC_ORDER,
    FireStack,
    _assemble_dynamic,
    _assemble_static,
    _mean_u_for_window,
)
from ignis_ml.scripts import eval_historical as ev  # noqa: E402


def _synthetic_fire(days: int = 8, h: int = 64, w: int = 64) -> FireStack:
    rng = np.random.default_rng(0)
    # Provide a subset of dynamic channels; gust must be derived, viirs zero-filled.
    dynamic = {
        "fire_t": (rng.random((days, h, w)) > 0.9).astype(np.float32),
        "u": rng.normal(-7.0, 1.0, (days, h, w)).astype(np.float32),  # Santa-Ana
        "v": rng.normal(1.0, 1.0, (days, h, w)).astype(np.float32),
        "tempC": rng.normal(25.0, 3.0, (days, h, w)).astype(np.float32),
        "q": rng.random((days, h, w)).astype(np.float32) * 0.01,
        "precip": np.zeros((days, h, w), np.float32),
        "erc": rng.random((days, h, w)).astype(np.float32) * 100,
        "bi": rng.random((days, h, w)).astype(np.float32) * 200,
        "ndvi": rng.random((days, h, w)).astype(np.float32),
    }
    static = {
        "elev": rng.random((h, w)).astype(np.float32) * 2000,
        "pdsi": rng.normal(0, 3, (h, w)).astype(np.float32),
        "chili": rng.random((h, w)).astype(np.float32) * 255,
        "water": np.zeros((h, w), np.float32),
        "fuel1": rng.normal(0, 1, (h, w)).astype(np.float32),
        "fuel2": rng.normal(0, 1, (h, w)).astype(np.float32),
    }
    return FireStack(
        name="synthetic", dynamic=dynamic, static=static,
        transform=None, height=h, width=w, ignition_lonlat=(-118.5, 34.0),
    )


def test_dynamic_assembly_shape_and_order():
    fire = _synthetic_fire()
    win = slice(0, 6)
    x_dyn = _assemble_dynamic(fire, win)
    assert x_dyn.shape == (6, len(V4_DYNAMIC_ORDER), 64, 64)
    assert len(V4_DYNAMIC_ORDER) == 12
    assert np.isfinite(x_dyn).all()
    # viirs channels absent -> zero-filled
    for name in ("viirs_i4", "viirs_i5"):
        idx = V4_DYNAMIC_ORDER.index(name)
        assert np.allclose(x_dyn[:, idx], 0.0)
    # gust derived (non-zero where wind present)
    g = x_dyn[:, V4_DYNAMIC_ORDER.index("gust")]
    assert g.mean() > 0.0


def test_static_assembly_shape_and_finite():
    fire = _synthetic_fire()
    x_stat = _assemble_static(fire)
    assert x_stat.shape == (len(V4_STATIC_ORDER), 64, 64)
    assert len(V4_STATIC_ORDER) == 9
    assert np.isfinite(x_stat).all()


def test_santa_ana_detection():
    fire = _synthetic_fire()
    x_dyn = _assemble_dynamic(fire, slice(0, 6))
    assert _mean_u_for_window(x_dyn) < -5.0  # synthetic winds are Santa-Ana


def test_eval_metrics_basic():
    a = np.zeros((64, 64), np.float32)
    b = np.zeros((64, 64), np.float32)
    a[10:20, 10:20] = 1
    b[10:20, 10:20] = 1
    assert ev.iou(a, b) == pytest.approx(1.0)
    assert ev.dice(a, b) == pytest.approx(1.0)
    assert ev.csi(a, b) == pytest.approx(1.0)
    assert ev.hausdorff_km(a, b) == pytest.approx(0.0)

    c = np.zeros((64, 64), np.float32)
    c[30:40, 30:40] = 1
    assert ev.iou(a, c) == 0.0
    assert ev.hausdorff_km(a, c) > 0.0
