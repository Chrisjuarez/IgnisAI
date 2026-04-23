from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def is_static_placeholder_or_missing(name: str, arr: np.ndarray) -> bool:
    if name == "water":
        return False
    critical_static = {
        "NDVI", "BI", "ERC", "PDSI", "CHILI",
        "ndvi", "bi", "erc", "pdsi", "chili",
        "fuel1", "fuel2", "fuel3", "impervious", "population",
    }
    return name in critical_static and float((np.asarray(arr) == 0).mean()) > 0.999


def display_mask_from_static(
    stat: Optional[np.ndarray],
    static_order: List[str],
    *,
    water_threshold: float = 0.5,
    impervious_threshold: float = 0.8,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if stat is None:
        return None, {"available": False, "reason": "static_tensor_missing"}

    burnable = np.ones(tuple(stat.shape[-2:]), dtype=np.float32)
    masked: Dict[str, float] = {}

    def channel(name: str) -> Optional[np.ndarray]:
        try:
            idx = list(static_order).index(name)
        except ValueError:
            return None
        return np.asarray(stat[idx], dtype=np.float32)

    water = channel("water")
    if water is not None:
        water_mask = water >= float(water_threshold)
        burnable[water_mask] = 0.0
        masked["water"] = float(water_mask.mean())

    impervious = channel("impervious")
    if impervious is not None:
        impervious_mask = impervious >= float(impervious_threshold)
        burnable[impervious_mask] = 0.0
        masked["impervious"] = float(impervious_mask.mean())

    return burnable, {
        "available": True,
        "description": "Display-only nonburnable mask; model inference remains unchanged",
        "masked_fraction": float((burnable <= 0.0).mean()),
        "masked_by_channel": masked,
        "thresholds": {
            "water": float(water_threshold),
            "impervious": float(impervious_threshold),
        },
    }
