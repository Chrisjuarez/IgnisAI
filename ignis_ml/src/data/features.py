# src/data/features.py
"""
Shared derived-feature computation.

Training and real-time inference must produce identical derived channels from
the same raw inputs, or the model will silently degrade when deployed against
NOAA forecast feeds. The functions operate on numpy arrays in normalized [0,1]
space, matching the convention in src/data/dataset.py.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


DERIVED_FEATURE_NAMES: Tuple[str, ...] = (
    "wind_speed",
    "wind_dir_cos",
    "wind_dir_sin",
    "temp_delta",
    "q_delta",
    "days_since_fire",
)

_U_LO, _U_HI = -15.0, 15.0
_V_LO, _V_HI = -15.0, 15.0
_GUST_HI = 25.0


def _denorm_uv(u_n: np.ndarray, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map normalized [0,1] u,v back to m/s for polar-form math."""
    u = u_n * (_U_HI - _U_LO) + _U_LO
    v = v_n * (_V_HI - _V_LO) + _V_LO
    return u.astype(np.float32), v.astype(np.float32)


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def compute_wind_polar(u_n: np.ndarray, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wind speed and direction from normalized u,v.

    Returns speed_n, dir_cos_n, and dir_sin_n, all in [0,1] normalized space.
    """
    u_ms, v_ms = _denorm_uv(u_n, v_n)
    speed = np.sqrt(u_ms * u_ms + v_ms * v_ms)
    speed_n = _clip01(speed / _GUST_HI)

    eps = 1e-3
    safe = speed >= (eps * _GUST_HI)
    cos_t = np.where(safe, u_ms / np.maximum(speed, eps), 0.0)
    sin_t = np.where(safe, v_ms / np.maximum(speed, eps), 0.0)
    dir_cos_n = _clip01((cos_t + 1.0) * 0.5)
    dir_sin_n = _clip01((sin_t + 1.0) * 0.5)
    return speed_n, dir_cos_n, dir_sin_n


def compute_temporal_delta(channel_seq: np.ndarray) -> np.ndarray:
    """
    Per-timestep first difference of a [T,H,W] channel, returned in [0,1].

    The first timestep gets 0 because no prior frame is available. Later frames
    use (x[t] - x[t-1] + 1) / 2, so measured zero change is represented as 0.5.
    """
    if channel_seq.ndim != 3:
        raise ValueError(f"expected [T,H,W], got shape {channel_seq.shape}")
    out = np.zeros_like(channel_seq, dtype=np.float32)
    if channel_seq.shape[0] >= 2:
        diff = channel_seq[1:] - channel_seq[:-1]
        out[1:] = _clip01((diff + 1.0) * 0.5)
    return out


def compute_days_since_fire(fire_seq: np.ndarray, max_days: int | None = None) -> np.ndarray:
    """
    Per-pixel frames-since-fire channel.

    Output is [T,H,W] in [0,1], where 0 means currently burning and 1 means
    never burned, or burned longer than the configured cap.
    """
    if fire_seq.ndim != 3:
        raise ValueError(f"expected [T,H,W], got shape {fire_seq.shape}")
    t_count, height, width = fire_seq.shape
    cap = float(max_days if max_days is not None else t_count)

    burning = fire_seq >= 0.5
    counter = np.full((height, width), cap, dtype=np.float32)
    out = np.empty((t_count, height, width), dtype=np.float32)
    for t_idx in range(t_count):
        counter = np.where(burning[t_idx], 0.0, np.minimum(counter + 1.0, cap))
        out[t_idx] = counter
    return _clip01(out / cap)


def append_derived_features(
    x_dyn_norm: np.ndarray,
    dyn_order: Sequence[str],
    include: Sequence[str] | None = None,
    days_since_fire_cap: int | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Append derived features to a normalized [T,Cd,H,W] dynamic tensor.

    Returns the expanded tensor and its output channel order.
    """
    if x_dyn_norm.ndim != 4:
        raise ValueError(f"expected [T,Cd,H,W], got {x_dyn_norm.shape}")
    if include is None:
        include = list(DERIVED_FEATURE_NAMES)
    else:
        include = list(include)
        unknown = [name for name in include if name not in DERIVED_FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown derived features: {unknown}")

    name_to_idx = {name: idx for idx, name in enumerate(dyn_order)}

    def _require(name: str) -> np.ndarray:
        if name not in name_to_idx:
            raise KeyError(
                f"derived features need base channel '{name}' but dyn_order={list(dyn_order)}"
            )
        return x_dyn_norm[:, name_to_idx[name]]

    extras: List[np.ndarray] = []
    new_names: List[str] = []

    if any(name in include for name in ("wind_speed", "wind_dir_cos", "wind_dir_sin")):
        speed_n, dir_cos_n, dir_sin_n = compute_wind_polar(_require("u"), _require("v"))
        if "wind_speed" in include:
            extras.append(speed_n)
            new_names.append("wind_speed")
        if "wind_dir_cos" in include:
            extras.append(dir_cos_n)
            new_names.append("wind_dir_cos")
        if "wind_dir_sin" in include:
            extras.append(dir_sin_n)
            new_names.append("wind_dir_sin")

    if "temp_delta" in include:
        extras.append(compute_temporal_delta(_require("tempC")))
        new_names.append("temp_delta")

    if "q_delta" in include:
        extras.append(compute_temporal_delta(_require("q")))
        new_names.append("q_delta")

    if "days_since_fire" in include:
        extras.append(compute_days_since_fire(_require("fire_t"), max_days=days_since_fire_cap))
        new_names.append("days_since_fire")

    if not extras:
        return x_dyn_norm.astype(np.float32, copy=False), list(dyn_order)

    extras_stacked = np.stack(extras, axis=1).astype(np.float32)
    x_out = np.concatenate([x_dyn_norm.astype(np.float32, copy=False), extras_stacked], axis=1)
    return x_out, list(dyn_order) + new_names


def expected_channel_count(
    base_order: Sequence[str],
    derived_include: Sequence[str] | None,
) -> int:
    """Return the expected dynamic channel count after derived features."""
    if derived_include is None:
        return len(base_order) + len(DERIVED_FEATURE_NAMES)
    return len(base_order) + len(derived_include)
