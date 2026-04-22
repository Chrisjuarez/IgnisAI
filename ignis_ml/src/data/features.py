# src/data/features.py
"""
Shared derived-feature computation.

WHY THIS MODULE EXISTS
----------------------
Training and real-time inference must produce IDENTICAL derived channels from
the same raw inputs, or the model will silently degrade when deployed against
NOAA forecast feeds. Putting the math in one module enforces that.

The functions operate on numpy arrays in normalized [0,1] space, matching the
convention in src/data/dataset.py (_to01 / _normalize_dynamic_by_name). They
expect the dataset's standard dynamic channel order:

    ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]

and append derived channels in a fixed order:

    ["wind_speed", "wind_dir_cos", "wind_dir_sin",
     "temp_delta", "q_delta", "days_since_fire"]

DESIGN NOTES
------------
* wind_speed, wind_dir_cos, wind_dir_sin are computed from u,v in
  PHYSICAL space (m/s) because the normalized [0,1] u,v values map 0.5 -> 0 m/s,
  which makes angle extraction unstable near the origin. We denormalize, compute
  polar form, then re-normalize the outputs to [0,1].
* temp_delta and q_delta are per-timestep first differences. The first
  timestep gets a zero delta (no prior frame to diff against).
* days_since_fire is a per-pixel "how many frames since fire_t last showed
  >= 0.5 at this location" channel. It is clipped to [0, T] / T so it lives
  in [0,1]. On inference rollout this channel gets updated after each day.

ALL FUNCTIONS ARE PURE (no shared state, no file I/O), so the same calls
can happen in a DataLoader worker process or inside a real-time inference
loop with no surprises.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# --- Feature set metadata --------------------------------------------------

# Canonical order of derived features. Do not reorder without also updating
# any config.yaml feature include list that references these names.
DERIVED_FEATURE_NAMES: Tuple[str, ...] = (
    "wind_speed",
    "wind_dir_cos",
    "wind_dir_sin",
    "temp_delta",
    "q_delta",
    "days_since_fire",
)

# Physical-space ranges used to denormalize/renormalize. These MUST match the
# ranges in NpzTileDataset._dyn_ranges or the derived features will be off.
_U_LO, _U_HI = -15.0, 15.0      # m/s
_V_LO, _V_HI = -15.0, 15.0      # m/s
_GUST_HI = 25.0                 # m/s (cap for wind_speed normalization)


# --- Internal helpers ------------------------------------------------------


def _denorm_uv(u_n: np.ndarray, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map normalized [0,1] u,v back to m/s for polar-form math."""
    u = u_n * (_U_HI - _U_LO) + _U_LO
    v = v_n * (_V_HI - _V_LO) + _V_LO
    return u.astype(np.float32), v.astype(np.float32)


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0).astype(np.float32)


# --- Public API ------------------------------------------------------------


def compute_wind_polar(u_n: np.ndarray, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wind speed and direction (cos/sin) from normalized u,v.

    Returns speed_n, dir_cos_n, dir_sin_n, all in [0,1] normalized space.
      * speed_n       ~ wind magnitude / _GUST_HI   (clipped to [0,1])
      * dir_cos_n     = (cos(theta) + 1) / 2
      * dir_sin_n     = (sin(theta) + 1) / 2

    Shapes are preserved. u_n and v_n may be 3D ([T,H,W]) or 4D ([T,C,H,W] slice).
    """
    u_ms, v_ms = _denorm_uv(u_n, v_n)
    speed = np.sqrt(u_ms * u_ms + v_ms * v_ms)
    speed_n = _clip01(speed / _GUST_HI)
    # Where speed is near zero, angle is ambiguous — mid-scale both components
    # to the neutral 0.5 (cos=sin=0 after denorm) so the model sees a stable
    # "no wind" signal rather than rapidly-flipping junk.
    eps = 1e-3
    safe = speed >= (eps * _GUST_HI)
    cos_t = np.where(safe, u_ms / np.maximum(speed, eps), 0.0)
    sin_t = np.where(safe, v_ms / np.maximum(speed, eps), 0.0)
    dir_cos_n = _clip01((cos_t + 1.0) * 0.5)
    dir_sin_n = _clip01((sin_t + 1.0) * 0.5)
    return speed_n, dir_cos_n, dir_sin_n


def compute_temporal_delta(channel_seq: np.ndarray) -> np.ndarray:
    """
    Per-timestep first difference of a [T, H, W] channel, returned in [0,1].

    delta[0] = 0 (no prior frame). delta[t>0] = (x[t] - x[t-1] + 1) / 2 so
    a neutral (no change) appears at 0.5 and negatives are preserved.

    Input is assumed already normalized to [0,1]. Output is clipped to [0,1].
    """
    if channel_seq.ndim != 3:
        raise ValueError(f"expected [T,H,W], got shape {channel_seq.shape}")
    T = channel_seq.shape[0]
    out = np.zeros_like(channel_seq, dtype=np.float32)
    if T >= 2:
        diff = channel_seq[1:] - channel_seq[:-1]         # in [-1, 1]
        out[1:] = _clip01((diff + 1.0) * 0.5)
    # first timestep stays at 0 -> that's what "no delta available" means.
    # We use 0 (not 0.5) because the model's temporal backbone handles
    # missing-context implicitly, and 0 distinguishes it from a measured
    # "zero change" (which comes out as 0.5).
    return out


def compute_days_since_fire(fire_seq: np.ndarray, max_days: int | None = None) -> np.ndarray:
    """
    Per-pixel "frames since fire" channel.

    Input:  fire_seq [T, H, W] normalized fire mask (~{0,1}, after dataset
            fire_boost+clip it's effectively binary but may have some [0,1]
            bleed — we threshold at 0.5).
    Output: [T, H, W] in [0, 1], where 0 = currently burning, 1 = never
            burned (or burned longer than max_days ago).

    If max_days is None, it defaults to T (the sequence length), meaning
    "at most T frames of history".
    """
    if fire_seq.ndim != 3:
        raise ValueError(f"expected [T,H,W], got shape {fire_seq.shape}")
    T, H, W = fire_seq.shape
    cap = float(max_days if max_days is not None else T)

    burning = fire_seq >= 0.5                             # [T,H,W] bool
    # Initialize "days since fire" as cap everywhere (never burned). As we
    # scan forward in time, reset to 0 at burning pixels and +1 otherwise.
    counter = np.full((H, W), cap, dtype=np.float32)
    out = np.empty((T, H, W), dtype=np.float32)
    for t in range(T):
        counter = np.where(burning[t], 0.0, np.minimum(counter + 1.0, cap))
        out[t] = counter
    return _clip01(out / cap)


def append_derived_features(
    x_dyn_norm: np.ndarray,
    dyn_order: Sequence[str],
    include: Sequence[str] | None = None,
    days_since_fire_cap: int | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Given a normalized [T, Cd, H, W] dynamic tensor and its channel name order,
    append the configured derived features along the channel axis.

    Parameters
    ----------
    x_dyn_norm : np.ndarray
        Shape [T, Cd, H, W], values in [0,1] (post-normalization).
    dyn_order : Sequence[str]
        Channel names matching x_dyn_norm's C axis. Must include at least the
        names referenced by the derived features the caller requests.
    include : Sequence[str] | None
        Subset of DERIVED_FEATURE_NAMES to append. None means "append all".
    days_since_fire_cap : int | None
        Cap for the days_since_fire channel. None defaults to T.

    Returns
    -------
    x_dyn_out : np.ndarray  [T, Cd + k, H, W]
    new_order : List[str]   channel names in the output tensor
    """
    if x_dyn_norm.ndim != 4:
        raise ValueError(f"expected [T,Cd,H,W], got {x_dyn_norm.shape}")
    if include is None:
        include = list(DERIVED_FEATURE_NAMES)
    else:
        include = list(include)
        unknown = [n for n in include if n not in DERIVED_FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown derived features: {unknown}")

    name_to_idx = {n: i for i, n in enumerate(dyn_order)}

    def _require(name: str) -> np.ndarray:
        if name not in name_to_idx:
            raise KeyError(
                f"derived features need base channel '{name}' but dyn_order={list(dyn_order)}"
            )
        return x_dyn_norm[:, name_to_idx[name]]

    extras: List[np.ndarray] = []
    new_names: List[str] = []

    # Compute wind polar once if any wind derivative is requested.
    if any(n in include for n in ("wind_speed", "wind_dir_cos", "wind_dir_sin")):
        u_seq = _require("u")
        v_seq = _require("v")
        speed_n, dir_cos_n, dir_sin_n = compute_wind_polar(u_seq, v_seq)
        if "wind_speed" in include:
            extras.append(speed_n); new_names.append("wind_speed")
        if "wind_dir_cos" in include:
            extras.append(dir_cos_n); new_names.append("wind_dir_cos")
        if "wind_dir_sin" in include:
            extras.append(dir_sin_n); new_names.append("wind_dir_sin")

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

    # Stack extras along a new channel axis: each extra is [T,H,W] -> [T,1,H,W]
    extras_stacked = np.stack(extras, axis=1).astype(np.float32)  # [T, k, H, W]
    x_out = np.concatenate([x_dyn_norm.astype(np.float32, copy=False), extras_stacked], axis=1)
    return x_out, list(dyn_order) + new_names


def expected_channel_count(
    base_order: Sequence[str],
    derived_include: Sequence[str] | None,
) -> int:
    """Convenience: given base dyn_order + derived include list, return Cd."""
    if derived_include is None:
        return len(base_order) + len(DERIVED_FEATURE_NAMES)
    return len(base_order) + len(derived_include)
