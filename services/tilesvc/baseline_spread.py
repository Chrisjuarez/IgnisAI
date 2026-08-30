"""A deterministic downwind spread baseline.

Every complex model owes an answer to "does it beat the obvious thing". For
wind-driven fire the obvious thing is: the fire runs downwind, faster in
stronger wind, and stretches into an ellipse as it goes. That is not a physical
model - it has no fuel, no slope, no moisture - and it is not meant to be. It
is the bar the learned model has to clear to be worth its complexity.

Right now it clears it easily, because the learned model runs upwind
(tools/audit_direction.py). Until that is fixed this baseline is also the more
defensible thing to show anyone, which is why it lives in the service rather
than in a notebook.

Shape of the output deliberately matches the model rollout - a list of
{prob, lead_hours, label} - so every consumer downstream, and the audit, treats
the two identically and comparisons stay honest.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .grid import PIX, SIZE

#: Rate of spread, metres per hour, as base + slope * wind.
#:
#: Anchored to an observed event rather than guessed. NASA put the Palisades
#: fire at ~14,500 acres burned on its worst day - 59 km2 - which for an
#: ellipse with the length-to-breadth ratio below implies a head advance of
#: roughly 7-8 km in 24 hours, so a few hundred metres per hour under a strong
#: Santa Ana. An earlier draft of these constants gave 7 km/h, which would run
#: a fire 170 km in a day and simply filled the tile.
#:
#: These are the only free parameters, and they are the two numbers to change
#: if the baseline is obviously too fast or slow.
ROS_BASE_M_PER_H = 60.0
ROS_PER_MS_M_PER_H = 40.0

#: How much longer the fire is along the wind than across it. Real
#: length-to-breadth grows with wind; this is a bounded linear stand-in.
LB_BASE = 1.6
LB_PER_MS = 0.55
LB_MAX = 6.0

#: A backing fire creeps against the wind at a small fraction of the head rate.
BACKING_FRACTION = 0.12

#: Wind below this gives no direction, so growth is treated as circular.
CALM_MS = 0.5


def rate_of_spread_m_per_h(wind_ms: float) -> float:
    return ROS_BASE_M_PER_H + ROS_PER_MS_M_PER_H * max(0.0, float(wind_ms))


def length_to_breadth(wind_ms: float) -> float:
    return min(LB_MAX, LB_BASE + LB_PER_MS * max(0.0, float(wind_ms)))


def _source_cells(observed: Optional[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Cells the fire is currently in, as (row, col) pairs."""
    if observed is None:
        return np.empty((0, 2), dtype=int)
    return np.argwhere(np.asarray(observed, dtype=np.float32) >= threshold)


def spread_field(
    observed: Optional[np.ndarray],
    *,
    u_ms: float,
    v_ms: float,
    hours: float,
    ignition_rc: Optional[tuple] = None,
) -> np.ndarray:
    """Probability of burn after `hours`, grown downwind from the source.

    Each source cell contributes an ellipse: long downwind, narrower across,
    and barely backing into the wind. A cell's value is set by the source that
    reaches it most strongly, which keeps a wide fire front from being modelled
    as a single point.
    """
    field = np.zeros((SIZE, SIZE), dtype=np.float32)
    sources = _source_cells(observed)
    if sources.size == 0 and ignition_rc is not None:
        sources = np.array([ignition_rc], dtype=int)
    if sources.size == 0:
        return field

    speed = math.hypot(float(u_ms), float(v_ms))
    head_m = rate_of_spread_m_per_h(speed) * max(0.0, float(hours))
    if head_m <= 0:
        return field

    if speed < CALM_MS:
        # No steer: grow a circle rather than invent a direction.
        along_hat = (0.0, 0.0)
        breadth_m = head_m
        back_m = head_m
    else:
        # Grid rows increase southward, so north is -row.
        along_hat = (float(u_ms) / speed, float(v_ms) / speed)  # (east, north)
        breadth_m = head_m / length_to_breadth(speed)
        back_m = head_m * BACKING_FRACTION

    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    for r0, c0 in sources:
        east_m = (cols - c0) * PIX
        north_m = (r0 - rows) * PIX

        if speed < CALM_MS:
            norm = np.hypot(east_m, north_m) / max(head_m, 1e-6)
        else:
            along = east_m * along_hat[0] + north_m * along_hat[1]
            cross = east_m * (-along_hat[1]) + north_m * along_hat[0]
            # Downwind reach is the head distance; upwind only the backing one.
            reach = np.where(along >= 0, head_m, back_m)
            norm = np.sqrt((along / np.maximum(reach, 1e-6)) ** 2
                           + (cross / max(breadth_m, 1e-6)) ** 2)

        # 1 at the source, tapering to 0 at the ellipse edge, so the result
        # reads as confidence that decays with distance rather than a hard mask.
        contribution = np.clip(1.0 - norm, 0.0, 1.0).astype(np.float32)
        np.maximum(field, contribution, out=field)

    return field


def baseline_rollout(
    observed: Optional[np.ndarray],
    *,
    u_ms: float,
    v_ms: float,
    steps: int,
    step_hours: int,
    ignition_rc: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """A rollout shaped exactly like the model's, so the two are comparable."""
    rollout: List[Dict[str, Any]] = []
    for index in range(max(1, int(steps))):
        lead_hours = (index + 1) * int(step_hours)
        rollout.append({
            "index": index,
            "lead_hours": lead_hours,
            "label": f"day {index + 1}" if step_hours == 24 else f"+{lead_hours}h",
            "prob": spread_field(observed, u_ms=u_ms, v_ms=v_ms,
                                 hours=lead_hours, ignition_rc=ignition_rc),
        })
    return rollout
