"""Fire growth via pyretechnics' Eulerian level-set solver.

The same spread algorithm ELMFIRE uses, from Spatial Informatics Group under
EPL 2.0 - usable commercially, unlike ELMFIRE itself, whose Commons Clause
forbids selling a service built on it.

Preferred over baseline_spread and physics_spread because it carries what
those do not: crown fire, ember spotting, a real level-set front rather than
an ellipse pasted onto each source cell, and weather that varies over the
forecast. It also spreads from whatever is already burning, which is the
normal case - a live incident has an irregular multi-day footprint, not a
point ignition.

Wind is fed as a time series, one band per forecast step. Holding wind constant
across three days is defensible for a single extreme event like a Santa Ana and
wrong for most fires, where the direction turns.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid import PIX, SIZE

#: Bands are one forecast step long, so wind can change per step.
MINUTES_PER_HOUR = 60.0

#: Fuel moisture. Live values follow the driest standard scenario; dead values
#: are the fine-fuel figure with the usual spacing between size classes. These
#: are inputs the pipeline does not yet measure - ERC and BI are proxies for
#: them - so they are named here rather than buried.
DEFAULT_MOISTURE = {
    "fuel_moisture_dead_1hr": 0.03,
    "fuel_moisture_dead_10hr": 0.04,
    "fuel_moisture_dead_100hr": 0.05,
    "fuel_moisture_live_herbaceous": 0.30,
    "fuel_moisture_live_woody": 0.60,
    "foliar_moisture": 0.90,
}

#: Canopy inputs drive crown fire. Absent a canopy layer in the catalog these
#: stay zero, which disables crowning rather than inventing a forest.
DEFAULT_CANOPY = {
    "canopy_cover": 0.0,
    "canopy_height": 0.0,
    "canopy_base_height": 0.0,
    "canopy_bulk_density": 0.0,
}


def wind_to_pyretechnics(u_ms: float, v_ms: float) -> Tuple[float, float]:
    """(speed km/h at 10 m, upwind direction in degrees clockwise from North).

    Our convention is a vector the wind blows TOWARD, in m/s. Pyretechnics
    wants the direction it comes FROM, which is the opposite bearing.
    """
    speed_kmh = math.hypot(float(u_ms), float(v_ms)) * 3.6
    toward = (math.degrees(math.atan2(float(u_ms), float(v_ms))) + 360.0) % 360.0
    return speed_kmh, (toward + 180.0) % 360.0


def slope_and_aspect(elevation_m: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Rise/run and downslope azimuth, in the units the solver expects."""
    if elevation_m is None:
        return np.zeros((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    dz_dy, dz_dx = np.gradient(np.asarray(elevation_m, dtype=np.float32), PIX)
    slope = np.hypot(dz_dx, dz_dy).astype(np.float32)
    # Aspect is the downslope direction, clockwise from North. Rows increase
    # southward, so the northward gradient is -dz_dy.
    aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0
    return slope, aspect.astype(np.float32)


def _cube(value, bands: int, stc):
    """Wrap a scalar or [T, H, W] array as a SpaceTimeCube."""
    shape = (bands, SIZE, SIZE)
    if np.ndim(value) == 0:
        return stc.SpaceTimeCube(shape, float(value))
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, ...], bands, axis=0)
    return stc.SpaceTimeCube(shape, arr)


def pyretechnics_rollout(
    observed: Optional[np.ndarray],
    *,
    fuel_codes: np.ndarray,
    wind_series: Sequence[Tuple[float, float]],
    steps: int,
    step_hours: int,
    elevation_m: Optional[np.ndarray] = None,
    moisture: Optional[Dict[str, float]] = None,
    use_wind_limit: bool = True,
) -> List[Dict[str, Any]]:
    """Cumulative burn probability per forecast step.

    wind_series carries one (u, v) in m/s per step. A single pair is broadcast,
    which is the right behaviour for a short forecast under a steady event and
    the wrong one for anything longer - hence the sequence.

    The level set gives a time of arrival per cell, which is a cleaner answer
    than a per-step probability: a cell either has burned by step k or has not.
    Reported as 1.0/0.0 so it slots into the same rollout contract as the
    learned model without pretending to a confidence it does not produce.
    """
    import pyretechnics.eulerian_level_set as els
    import pyretechnics.space_time_cube as stc

    n = max(1, int(steps))
    winds = list(wind_series) or [(0.0, 0.0)]
    if len(winds) < n:
        winds = winds + [winds[-1]] * (n - len(winds))

    speeds = np.zeros((n, SIZE, SIZE), np.float32)
    upwinds = np.zeros((n, SIZE, SIZE), np.float32)
    for i, (u, v) in enumerate(winds[:n]):
        speed_kmh, upwind_deg = wind_to_pyretechnics(u, v)
        speeds[i, :, :] = speed_kmh
        upwinds[i, :, :] = upwind_deg

    slope, aspect = slope_and_aspect(elevation_m)
    moist = {**DEFAULT_MOISTURE, **(moisture or {})}

    # Row 0 is SOUTH here - ignite_cells takes a lower_left_corner - while our
    # rasters are north-up with row 0 at the top. Every spatial input is
    # flipped on the way in and the result is flipped back. A symmetric test
    # blob hides this; a real fire footprint does not, and the spread comes out
    # mirrored north-south.
    flip = np.flipud

    cubes = {
        "slope": _cube(flip(slope), n, stc),
        "aspect": _cube(flip(aspect), n, stc),
        "fuel_model": _cube(flip(np.asarray(fuel_codes, dtype=np.float32)), n, stc),
        "wind_speed_10m": _cube(speeds, n, stc),
        "upwind_direction": _cube(upwinds, n, stc),
        "temperature": _cube(20.0, n, stc),
        **{k: _cube(v, n, stc) for k, v in DEFAULT_CANOPY.items()},
        **{k: _cube(v, n, stc) for k, v in moist.items()},
    }

    # The level set stores phi, negative inside the fire and positive outside,
    # and ignite_cells copies those values in directly - handing it a mask of
    # 1.0 marks every cell UNBURNED and the solver stops immediately with "no
    # burnable cells".
    flip = np.flipud
    already_burning = (np.asarray(observed) >= 0.5) if observed is not None else None
    if already_burning is None or not already_burning.any():
        return [_empty_step(i, step_hours) for i in range(n)]

    state = els.SpreadState(cube_shape=(n, SIZE, SIZE))
    seed = np.ones((SIZE, SIZE), dtype=np.float32)
    seed[flip(already_burning)] = -1.0
    state.ignite_cells(lower_left_corner=(0, 0), ignition_matrix=seed)

    band_minutes = float(step_hours) * MINUTES_PER_HOUR
    result = els.spread_fire_with_phi_field(
        cubes, state,
        cube_resolution=(band_minutes, float(PIX), float(PIX)),
        start_time=0.0,
        max_duration=band_minutes * n,
        use_wind_limit=use_wind_limit,
    )
    arrival = flip(np.asarray(result["spread_state"].get_full_matrices()["time_of_arrival"],
                              dtype=np.float32))

    out: List[Dict[str, Any]] = []
    for i in range(n):
        lead = (i + 1) * int(step_hours)
        burned = (arrival >= 0.0) & (arrival <= band_minutes * (i + 1))
        out.append({
            "index": i,
            "lead_hours": lead,
            "label": f"{lead // 24} day" if lead % 24 == 0 else f"{lead}h",
            "prob": burned.astype(np.float32),
        })
    return out


def _empty_step(index: int, step_hours: int) -> Dict[str, Any]:
    lead = (index + 1) * int(step_hours)
    return {
        "index": index,
        "lead_hours": lead,
        "label": f"{lead // 24} day" if lead % 24 == 0 else f"{lead}h",
        "prob": np.zeros((SIZE, SIZE), dtype=np.float32),
    }
