from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .static_catalog import InputUnavailable


DEFAULT_RISK_BREAKS = (
    ("low", 0.05),
    ("medium", 0.20),
    ("high", 0.50),
    ("extreme", 1.01),
)


_calibration_cache: Dict[str, Any] | None = None


def _identity_calibration(reason: str | None = None) -> Dict[str, Any]:
    cal: Dict[str, Any] = {
        "method": "identity",
        "points": [[0.0, 0.0], [1.0, 1.0]],
        "risk_breaks": DEFAULT_RISK_BREAKS,
    }
    if reason:
        cal["placeholder_or_missing"] = True
        cal["reason"] = reason
    return cal


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:") :]
    return normalized


def _calibration_path() -> Path | None:
    raw = os.getenv("CALIBRATION_PATH")
    return Path(raw) if raw else None


def load_calibration(*, model_sha256: str | None = None) -> Dict[str, Any]:
    global _calibration_cache
    if _calibration_cache is not None:
        return _calibration_cache
    path = _calibration_path()
    required = os.getenv("CALIBRATION_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not path:
        if required:
            raise InputUnavailable(
                "CALIBRATION_PATH is required for calibrated production display",
                reason="calibration_missing",
            )
        _calibration_cache = _identity_calibration("calibration_missing")
        return _calibration_cache
    if not path.exists():
        if not required:
            _calibration_cache = _identity_calibration("calibration_missing")
            return _calibration_cache
        raise InputUnavailable(
            f"Calibration artifact does not exist: {path}",
            reason="calibration_missing",
            details={"path": str(path)},
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    expected_hash = data.get("model_sha256")
    expected_norm = _normalize_sha256(expected_hash)
    actual_norm = _normalize_sha256(model_sha256)
    if expected_norm and actual_norm and expected_norm != actual_norm:
        raise InputUnavailable(
            "Calibration artifact does not match loaded model hash",
            reason="calibration_model_mismatch",
            details={"expected_model_sha256": expected_hash, "actual_model_sha256": model_sha256},
        )
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise InputUnavailable(
            "Calibration artifact must contain at least two [raw, calibrated] points",
            reason="calibration_invalid",
            details={"path": str(path)},
        )
    _calibration_cache = data
    return data


def calibration_status(model_sha256: str | None = None) -> Dict[str, Any]:
    try:
        cal = load_calibration(model_sha256=model_sha256)
        return {
            "ok": True,
            "method": cal.get("method", "isotonic"),
            "path": str(_calibration_path()) if _calibration_path() else None,
            "model_sha256": cal.get("model_sha256"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": str(_calibration_path()) if _calibration_path() else None,
            "error": str(exc),
        }


def _risk_breaks(cal: Dict[str, Any]):
    raw = cal.get("risk_breaks")
    if isinstance(raw, list) and raw:
        parsed = []
        for item in raw:
            if isinstance(item, dict):
                parsed.append((str(item["class"]), float(item["upper"])))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.append((str(item[0]), float(item[1])))
        if parsed:
            return tuple(parsed)
    return DEFAULT_RISK_BREAKS


def calibrate_probability(prob: np.ndarray, *, model_sha256: str | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    cal = load_calibration(model_sha256=model_sha256)
    pts = np.asarray(cal.get("points"), dtype=np.float32)
    raw = np.clip(pts[:, 0], 0.0, 1.0)
    mapped = np.clip(pts[:, 1], 0.0, 1.0)
    order = np.argsort(raw)
    raw = raw[order]
    mapped = mapped[order]
    score = np.interp(np.clip(prob, 0.0, 1.0), raw, mapped).astype(np.float32)

    risk = np.full(score.shape, "low", dtype=object)
    lower = 0.0
    for klass, upper in _risk_breaks(cal):
        mask = (score >= lower) & (score < float(upper))
        risk[mask] = klass
        lower = float(upper)
    meta = {
        "method": cal.get("method", "isotonic"),
        "path": str(_calibration_path()) if _calibration_path() else None,
        "risk_breaks": [{"class": name, "upper": upper} for name, upper in _risk_breaks(cal)],
    }
    return score, risk, meta
