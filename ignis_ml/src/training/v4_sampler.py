"""v4 Santa-Ana-aware training sampler.

v3's build_train_sampler balances by fire fraction only. Santa-Ana fires are
<2% of the merged dataset but are exactly the events the model fails on. This
sampler up-weights tiles whose source fire is flagged `is_santa_ana` (computed
during ingestion from mean u over the fire's lifecycle) so they appear in
~10% of batches.

It composes with the fire-fraction weighting: final weight = fire_frac_weight *
santa_ana_multiplier. Pass it the same `train_files` list and the per-fire meta
directory written by ingest_ts_satfire.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def _load_santa_ana_tilekeys(meta_dir: Path) -> Dict[str, bool]:
    """Map tile-key stem -> is_santa_ana, reading per-fire meta JSON.

    Each meta JSON is expected to contain {"is_santa_ana": bool, "tile_keys":
    [...]} as written by ingest_ts_satfire.py. Returns {} if meta_dir missing.
    """
    flags: Dict[str, bool] = {}
    if not meta_dir or not Path(meta_dir).is_dir():
        return flags
    for jf in Path(meta_dir).glob("*.json"):
        try:
            meta = json.loads(jf.read_text())
        except Exception:
            continue
        is_sa = bool(meta.get("is_santa_ana", False))
        for key in meta.get("tile_keys", []):
            flags[str(key)] = is_sa
    return flags


def _fire_fraction(npz_path: Path) -> float:
    """Mean positive fraction of the delta/target in a tile (cheap, mmap'd)."""
    try:
        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as d:
            y = d["y"]
            return float((np.asarray(y) > 0.5).mean())
    except Exception:
        return 0.0


#: How hard to rebalance the fire-fraction bins.
#:
#:   "inverse"      weight = 1/bin_count  -> every bin gets EQUAL total mass.
#:                  Maximum rebalancing. On mNDWS_500m_T3 this oversamples the
#:                  rarest bin ~133x and shifts the delta positive prior from
#:                  0.035 (natural) to 0.185 (as sampled) — a 5.26x prior shift
#:                  that the model then bakes into its output probabilities.
#:                  See ignis_ml/scripts/diagnose_prior_shift.py.
#:   "sqrt_inverse" weight = 1/sqrt(bin_count) -> partial rebalancing. Keeps
#:                  fire-heavy tiles well represented without the extreme prior
#:                  shift, so raw probabilities stay closer to deployment-real.
#:   "uniform"      no rebalancing; natural class distribution.
WEIGHT_MODES = ("inverse", "sqrt_inverse", "uniform")


def build_santa_ana_sampler(
    train_files: Sequence[Path],
    meta_dir: Optional[Path] = None,
    *,
    santa_ana_boost: float = 5.0,
    num_bins: int = 6,
    floor_weight: float = 0.05,
    seed: int = 42,
    weight_mode: str = "inverse",
) -> Tuple[WeightedRandomSampler, dict]:
    """Return (sampler, stats).

    Weight per tile = (binned fire-fraction weight) * (boost if santa_ana).
    Mirrors the intent of train_nautilus.build_train_sampler and multiplies in
    the Santa-Ana boost on top.

    `weight_mode` controls how aggressively the fire-fraction bins are
    rebalanced; see WEIGHT_MODES. Default "inverse" preserves the historical
    v3/v4 behavior so existing runs stay reproducible.

    The returned stats include `prior_shift`, the ratio of the sampled mean
    fire fraction to the natural one. That number is the factor by which the
    model's learned probabilities will be inflated relative to deployment, so
    it belongs in every training log.
    """
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}")

    files = [Path(f) for f in train_files]

    sa_flags = _load_santa_ana_tilekeys(Path(meta_dir)) if meta_dir else {}

    fracs = np.array([_fire_fraction(f) for f in files], dtype=np.float64)

    # Bin by fire fraction so empty/near-empty tiles don't dominate, matching
    # the v3 fire-fraction-balanced intent.
    if fracs.max() > 0 and weight_mode != "uniform":
        edges = np.linspace(0.0, fracs.max() + 1e-9, num_bins + 1)
        bins = np.clip(np.digitize(fracs, edges) - 1, 0, num_bins - 1)
        bin_counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
        if weight_mode == "inverse":
            inv = np.where(bin_counts > 0, 1.0 / bin_counts, 0.0)
        else:  # sqrt_inverse
            inv = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0)
        base_w = inv[bins]
    else:
        base_w = np.ones(len(files), dtype=np.float64)

    base_w = np.maximum(base_w / (base_w.mean() + 1e-12), floor_weight)

    boost = np.array(
        [santa_ana_boost if sa_flags.get(f.stem, False) else 1.0 for f in files],
        dtype=np.float64,
    )
    weights = base_w * boost

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights_t, num_samples=len(weights_t), replacement=True
    )

    # Prior shift: what the sampler does to the class balance the model sees.
    p_draw = weights / (weights.sum() + 1e-12)
    natural = float(fracs.mean())
    sampled = float((p_draw * fracs).sum())

    stats = {
        "n_tiles": len(files),
        "n_santa_ana": int((boost > 1.0).sum()),
        "santa_ana_fraction_raw": float((boost > 1.0).mean()),
        "mean_fire_fraction": natural,
        "effective_santa_ana_sampling_share": float(
            weights[boost > 1.0].sum() / (weights.sum() + 1e-12)
        ),
        "weight_mode": weight_mode,
        "sampled_fire_fraction": sampled,
        "prior_shift": float(sampled / max(natural, 1e-12)),
    }
    return sampler, stats
