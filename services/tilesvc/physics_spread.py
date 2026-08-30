"""Fire growth from Rothermel rates, in the shape the service already returns.

The deterministic baseline in baseline_spread invents its rate of spread from
two constants fitted to one fire. This does the same geometry from published
fuel physics instead: every cell gets its own rate from its own fuel model,
moisture, wind and slope, so the front slows in timber, stops at a road or a
lake, and runs in cured grass.

Same interface as baseline_rollout, so /predict_multistep can serve it and the
direction audit can score it against the learned model without special cases.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .grid import PIX, SIZE
from .rothermel import length_to_breadth, spread_rate_grid

CALM_MS = 0.5
BACKING_FRACTION = 0.12


def _source_cells(observed: Optional[np.ndarray]) -> np.ndarray:
    if observed is None:
        return np.empty((0, 2), dtype=int)
    return np.argwhere(np.asarray(observed) >= 0.5)


def slope_fraction_from_elevation(elevation_m: Optional[np.ndarray]) -> np.ndarray:
    """Rise over run per cell, from the elevation grid the model already loads."""
    if elevation_m is None:
        return np.zeros((SIZE, SIZE), dtype=np.float32)
    dz_dy, dz_dx = np.gradient(np.asarray(elevation_m, dtype=np.float32), PIX)
    return np.hypot(dz_dx, dz_dy).astype(np.float32)


def physics_spread_field(
    observed: Optional[np.ndarray],
    *,
    fuel_codes: np.ndarray,
    u_ms: float,
    v_ms: float,
    hours: float,
    dead_moisture: float = 0.06,
    live_moisture: float = 0.60,
    herb_moisture: float = 0.30,
    elevation_m: Optional[np.ndarray] = None,
    ignition_rc: Optional[tuple] = None,
) -> np.ndarray:
    """Burn confidence after `hours`, grown from the observed fire."""
    field = np.zeros((SIZE, SIZE), dtype=np.float32)
    sources = _source_cells(observed)
    if sources.size == 0 and ignition_rc is not None:
        sources = np.array([ignition_rc], dtype=int)
    if sources.size == 0:
        return field

    speed = math.hypot(float(u_ms), float(v_ms))
    slope = slope_fraction_from_elevation(elevation_m)
    ros = spread_rate_grid(
        fuel_codes, dead_moisture=dead_moisture, live_moisture=live_moisture,
        wind_ms=speed, slope_fraction=slope,
    )
    if not np.any(ros > 0):
        return field

    # One representative head rate for the geometry, but the per-cell field
    # still gates growth: a front cannot enter a cell whose fuel cannot burn.
    burnable = ros > 0
    head_m = float(np.mean(ros[burnable])) * max(0.0, float(hours))
    if head_m <= 0:
        return field

    if speed < CALM_MS:
        along_hat = (0.0, 0.0)
        breadth_m = back_m = head_m
    else:
        along_hat = (float(u_ms) / speed, float(v_ms) / speed)   # (east, north)
        breadth_m = head_m / length_to_breadth(speed)
        back_m = head_m * BACKING_FRACTION

    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    for r0, c0 in sources:
        east_m = (cols - c0) * PIX
        north_m = (r0 - rows) * PIX          # rows increase southward
        if speed < CALM_MS:
            norm = np.hypot(east_m, north_m) / max(head_m, 1e-6)
        else:
            along = east_m * along_hat[0] + north_m * along_hat[1]
            cross = east_m * (-along_hat[1]) + north_m * along_hat[0]
            reach = np.where(along >= 0, head_m, back_m)
            norm = np.sqrt((along / np.maximum(reach, 1e-6)) ** 2
                           + (cross / max(breadth_m, 1e-6)) ** 2)
        np.maximum(field, np.clip(1.0 - norm, 0.0, 1.0).astype(np.float32), out=field)

    # Non-burnable cells are barriers, not slow fuel. This is what makes the
    # physics run visibly different from an ellipse: fronts stop at water,
    # rock and pavement instead of drawing through them.
    field[~burnable] = 0.0
    return field


def physics_rollout(
    observed: Optional[np.ndarray],
    *,
    fuel_codes: np.ndarray,
    u_ms: float,
    v_ms: float,
    steps: int,
    step_hours: int,
    elevation_m: Optional[np.ndarray] = None,
    dead_moisture: float = 0.06,
    live_moisture: float = 0.60,
    ignition_rc: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index in range(max(1, int(steps))):
        lead = (index + 1) * int(step_hours)
        out.append({
            "index": index,
            "lead_hours": lead,
            "label": f"{lead // 24} day" if lead % 24 == 0 else f"{lead}h",
            "prob": physics_spread_field(
                observed, fuel_codes=fuel_codes, u_ms=u_ms, v_ms=v_ms, hours=lead,
                dead_moisture=dead_moisture, live_moisture=live_moisture,
                elevation_m=elevation_m, ignition_rc=ignition_rc,
            ),
        })
    return out
