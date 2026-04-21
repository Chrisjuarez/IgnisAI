from __future__ import annotations

from typing import Dict

import numpy as np


def next_fire_from_delta(
    prob_new_burn: np.ndarray,
    observed_fire: np.ndarray,
    threshold: float,
    *,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "delta":
        return np.maximum(
            observed_fire.astype(np.float32),
            (prob_new_burn >= float(threshold)).astype(np.float32),
        )
    return (prob_new_burn >= float(threshold)).astype(np.float32)


def risk_class_summary(risk: np.ndarray) -> Dict[str, float]:
    total = float(risk.size or 1)
    return {
        klass: float((risk == klass).sum()) / total
        for klass in ("low", "medium", "high", "extreme")
    }
