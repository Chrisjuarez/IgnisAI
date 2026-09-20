"""Combine several spread engines into one field by averaging percentile ranks.

Measured on the five validation fires, each engine fails in a different way.
The learned model ranks cells better than any physics engine (pooled AUC 0.673
against rothermel's 0.554) but its probabilities are so compressed that at the
deployed threshold of 0.5 it emits almost nothing - 5% recall, a tenth of the
area that actually burned. Rothermel has the opposite problem: it finds most of
the real growth and over-predicts it 2.3x, because it models no suppression, no
fuel exhaustion and no multi-day moisture recovery.

Averaging their probabilities does not work, because the engines do not share a
scale: the learned model's useful range sits near 1e-3 and rothermel's near
1e-1, so any direct blend is decided entirely by rothermel. Averaging percentile
ranks makes them commensurate without needing any of them to be calibrated,
which matters because we know the learned one is not.

Leave-one-fire-out, threshold fitted on four fires and scored on the fifth:

    engine            held-out IoU (pooled)
    pyretechnics                      0.055
    rothermel                         0.063
    downwind                          0.074
    learned                           0.103
    rank-mean of the three            0.218

The ensemble beat every single engine on every one of the five held-out fires,
which is a stronger signal than the pooled figure on its own. Downwind is left
out deliberately: it scores barely above chance (AUC 0.565), is largely
redundant with rothermel, and including it drops the held-out figure to 0.175.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

#: Operating point for the combined field. Leave-one-out threshold selection
#: chose 0.690-0.753 across the five folds; this is the middle of that band.
#: The narrow spread is the reason to trust it - a method whose best threshold
#: swings with the fold would not survive a sixth fire.
HYBRID_THRESHOLD = 0.72


def percentile_ranks(field: np.ndarray, *, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rank cells in [0, 1] by their score, within `mask`.

    Ties take the average of the positions they span, so a field that is
    constant over a region does not impose an arbitrary order on it. Cells
    outside the mask are zero: they are not competing for rank.
    """
    out = np.zeros(field.shape, dtype=np.float32)
    selected = np.ones(field.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    values = np.asarray(field, dtype=np.float64)[selected]
    if values.size == 0:
        return out
    if values.size == 1:
        out[selected] = 1.0
        return out

    order = np.argsort(values, kind="mergesort")
    positions = np.empty(values.size, dtype=np.float64)
    positions[order] = np.arange(values.size, dtype=np.float64)

    # Average the positions within each run of equal values.
    ordered = values[order]
    boundaries = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1], [True])))
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        positions[order[start:stop]] = (start + stop - 1) / 2.0

    out[selected] = (positions / (values.size - 1)).astype(np.float32)
    return out


def combine_fields(fields: Sequence[np.ndarray], *, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean percentile rank across engines. Unweighted on purpose: weighting
    the physics engines higher scored 0.222 against 0.218, which on five fires
    is not a difference, and it would add two constants nothing justifies."""
    if not fields:
        raise ValueError("hybrid needs at least one engine field")
    stacked = np.stack([percentile_ranks(f, mask=mask) for f in fields])
    return stacked.mean(axis=0).astype(np.float32)


def hybrid_rollout(
    rollouts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Combine per-step rollouts from several engines into one rollout.

    `rollouts` maps engine name to that engine's list of steps, each carrying a
    "prob" field. Every engine must have run the same number of steps, because
    step i of one is only comparable with step i of another.
    """
    if not rollouts:
        raise ValueError("hybrid needs at least one engine rollout")

    lengths = {name: len(steps) for name, steps in rollouts.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"engines disagree on step count: {lengths}")

    steps = next(iter(lengths.values()))
    combined: List[Dict[str, Any]] = []
    for index in range(steps):
        fields = [np.asarray(rollouts[name][index]["prob"], dtype=np.float32) for name in rollouts]
        combined.append({"prob": combine_fields(fields, mask=mask), "engines": list(rollouts)})
    return combined


def wind_series_from_dyn(dyn: np.ndarray, channel_order: Sequence[str], frames: int = 3) -> List[tuple]:
    """Per-frame mean (u, v) over the last `frames` input frames."""
    order = list(channel_order)
    u_idx, v_idx = order.index("u"), order.index("v")
    count = min(frames, dyn.shape[0])
    return [
        (float(dyn[i, u_idx].mean()), float(dyn[i, v_idx].mean()))
        for i in range(-count, 0)
    ]


#: Fewer engines than this and the combination is not an ensemble. Measured on
#: the five fires, every pair scores 0.218-0.233 at HYBRID_THRESHOLD against the
#: full trio's 0.235, and every leave-one-out band brackets that threshold, so
#: losing one engine costs little and needs no different operating point.
MIN_ENGINES = 2


def build_hybrid_rollout(
    learned: Sequence[Mapping[str, Any]],
    *,
    tile: Any,
    observed_fire: np.ndarray,
    wind_series: Sequence[tuple],
    steps: int,
    step_hours: int,
) -> tuple:
    """
    Run the physics engines beside a learned rollout and combine what succeeds.

    Returns (rollout, status). The rollout is None whenever the hybrid cannot
    be produced, and status always names the engines used and any that failed -
    a prediction must not fail because an optional extra layer could not be
    built, and it must never be ambiguous whether the layer is absent, degraded
    or merely empty. An engine missing from the image looks exactly like an
    engine that crashed unless the status distinguishes them.

    Ranks are taken over cells that are not already alight, matching the
    population the combiner was validated against.
    """
    from .fuel_raster import fuel_codes_for_tile
    from .grid import SIZE
    from .static_catalog import InputUnavailable

    try:
        codes = fuel_codes_for_tile(tile)
    except InputUnavailable as exc:
        return None, {"available": False, "reason": exc.reason, "detail": str(exc)}

    prior = np.asarray(observed_fire, dtype=np.float32)
    u_ms, v_ms = wind_series[-1] if wind_series else (0.0, 0.0)
    ignition_rc = (int(SIZE // 2), int(SIZE // 2))

    def run_rothermel():
        from .physics_spread import physics_rollout
        return physics_rollout(
            prior, fuel_codes=codes, u_ms=u_ms, v_ms=v_ms,
            steps=steps, step_hours=step_hours, ignition_rc=ignition_rc,
        )

    def run_pyretechnics():
        from .pyretechnics_spread import pyretechnics_rollout
        return pyretechnics_rollout(
            prior, fuel_codes=codes, wind_series=list(wind_series),
            steps=steps, step_hours=step_hours,
        )

    members: Dict[str, Any] = {"learned": list(learned)}
    failed: Dict[str, str] = {}
    for name, run in (("rothermel", run_rothermel), ("pyretechnics", run_pyretechnics)):
        try:
            members[name] = run()
        except Exception as exc:  # noqa: BLE001 - one engine down is not an outage
            failed[name] = f"{type(exc).__name__}: {exc}"

    if len(members) < MIN_ENGINES:
        return None, {
            "available": False,
            "reason": "insufficient_engines",
            "engines": list(members),
            "failed": failed,
        }

    try:
        rollout = hybrid_rollout(members, mask=prior < 0.5)
    except Exception as exc:  # noqa: BLE001
        return None, {"available": False, "reason": "hybrid_combine_error", "detail": repr(exc)}

    return rollout, {
        "available": True,
        "engines": list(members),
        "failed": failed,
        "degraded": bool(failed),
        "threshold": HYBRID_THRESHOLD,
        "units": "percentile_rank",
        "calibrated": False,
    }
