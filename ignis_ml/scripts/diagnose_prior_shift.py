#!/usr/bin/env python3
"""Diagnose the train/val positive-prior gap and its downstream effects.

Why this exists
---------------
The Phase A run logged `delta pos prior=0.20594` (-> focal alpha=0.794,
pos_weight=3.86), but a full-range re-sweep of the same checkpoint measured the
val set's delta prevalence at 0.036 — a 5.7x gap.

The cause is visible in `src/training/v4_sampler.py`: tile weights are the
INVERSE of their fire-fraction bin count, so every bin receives equal total
sampling mass. With `num_bins=6` and a long-tailed fire-fraction distribution,
the rare fire-heavy bins get sampled as often as the huge near-empty bin. The
training loader therefore sees a very different class balance than the val
loader, and `estimate_pos_prior` (which reads the *weighted train* loader)
faithfully reports that shifted number.

This is prior shift, and it has three consequences worth measuring:
  1. focal `alpha` and `pos_weight` are set from the shifted prior, not the
     deployment prior.
  2. the model learns P(fire | oversampled) instead of P(fire | real), which
     shows up as systematic over-prediction — exactly the shape of the isotonic
     curve (raw 0.50 -> 0.137 observed).
  3. any threshold tuned on val is compensating for a bias introduced in train.

This script quantifies all three from the actual tiles. Read-only; changes
nothing.

Usage
-----
    python -m ignis_ml.scripts.diagnose_prior_shift --config IgnisAI/ignis_ml/config.phase_a.yaml
    python -m ignis_ml.scripts.diagnose_prior_shift --config ... --max-tiles 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_yaml(p: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(p.read_text())


def collect_tiles(cfg: Dict[str, Any]) -> List[Path]:
    from ignis_ml.src.utils.paths import resolve_data_path
    files: List[Path] = []
    for name, ds in (cfg.get("datasets") or {}).items():
        td = ds.get("tiles_dir")
        if not td:
            continue
        p = resolve_data_path(td)
        if p.is_dir():
            found = sorted(p.glob("*.npz"))
            print(f"[data] {name}: {len(found)} tiles in {p}")
            files += found
    return files


def val_split(files: List[Path], seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * 0.15))
    return [files[i] for i in idx[n_val:]], [files[i] for i in idx[:n_val]]


def tile_stats(p: Path) -> Tuple[float, float]:
    """Return (full_mask_fraction, delta_fraction) for one tile.

    delta = y AND NOT last-frame-fire, matching NpzTileDataset's delta target.
    fire_t is dynamic channel 0 by convention in every IgnisAI schema.
    """
    try:
        with np.load(p, mmap_mode="r", allow_pickle=False) as d:
            y = np.asarray(d["y"]) > 0.5
            fire_last = np.asarray(d["x_dyn"][-1, 0]) > 0.5
        return float(y.mean()), float((y & ~fire_last).mean())
    except Exception:
        return 0.0, 0.0


def sampler_weights(fracs: np.ndarray, num_bins: int = 6,
                    floor_weight: float = 0.05) -> np.ndarray:
    """Reproduce build_santa_ana_sampler's weighting (no Santa-Ana boost:
    n_santa_ana was 0 on this corpus)."""
    if fracs.max() <= 0:
        return np.ones_like(fracs)
    edges = np.linspace(0.0, fracs.max() + 1e-9, num_bins + 1)
    bins = np.clip(np.digitize(fracs, edges) - 1, 0, num_bins - 1)
    counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / counts, 0.0)
    w = inv[bins]
    return np.maximum(w / (w.mean() + 1e-12), floor_weight), bins, counts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose train/val prior shift")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="subsample for speed (0 = all)")
    args = ap.parse_args(argv)

    from ignis_ml.src.utils.paths import describe
    print(f"[diag] {describe()}")

    cfg = load_yaml(args.config)
    seed = int(cfg_get(cfg, "training.seed", 42))
    files = collect_tiles(cfg)
    if not files:
        print("[diag] no tiles found")
        return 1

    train_files, val_files = val_split(files, seed)
    print(f"[diag] train={len(train_files)} val={len(val_files)} (seed={seed})")

    if args.max_tiles:
        rng = np.random.default_rng(0)
        if len(train_files) > args.max_tiles:
            train_files = [train_files[i] for i in
                           rng.choice(len(train_files), args.max_tiles, replace=False)]
        if len(val_files) > args.max_tiles:
            val_files = [val_files[i] for i in
                         rng.choice(len(val_files), args.max_tiles, replace=False)]
        print(f"[diag] subsampled to train={len(train_files)} val={len(val_files)}")

    print("[diag] scanning tiles (mmap, y + last fire frame only)...")
    tr = np.array([tile_stats(p) for p in train_files])     # [N,2]
    va = np.array([tile_stats(p) for p in val_files])
    tr_full, tr_delta = tr[:, 0], tr[:, 1]
    va_full, va_delta = va[:, 0], va[:, 1]

    # ---- what the sampler does -------------------------------------------
    w, bins, counts = sampler_weights(tr_full)
    p = w / w.sum()
    weighted_delta = float((p * tr_delta).sum())
    weighted_full = float((p * tr_full).sum())

    print("\n=== fire-fraction binning (what the sampler keys on) ===")
    print(f"  {'bin':>4} {'tiles':>8} {'share':>8} {'sampled share':>14} {'oversample':>11}")
    for b in range(len(counts)):
        if counts[b] == 0:
            continue
        share = counts[b] / counts.sum()
        samp = float(p[bins == b].sum())
        print(f"  {b:>4} {int(counts[b]):>8} {share:>8.4f} {samp:>14.4f} "
              f"{samp / max(share, 1e-12):>10.1f}x")

    print("\n=== positive prevalence (delta target) ===")
    print(f"  train, UNWEIGHTED        : {tr_delta.mean():.4f}")
    print(f"  train, AS SAMPLED        : {weighted_delta:.4f}   <- what training sees")
    print(f"  val   (deployment-like)  : {va_delta.mean():.4f}")
    print(f"  shift (sampled / val)    : {weighted_delta / max(va_delta.mean(), 1e-9):.2f}x")

    print("\n=== positive prevalence (full mask) ===")
    print(f"  train, UNWEIGHTED        : {tr_full.mean():.4f}")
    print(f"  train, AS SAMPLED        : {weighted_full:.4f}")
    print(f"  val                      : {va_full.mean():.4f}")

    # ---- downstream consequences -----------------------------------------
    pi_tr, pi_va = weighted_delta, float(va_delta.mean())
    print("\n=== consequences ===")
    print(f"  focal alpha from sampled prior : {1 - pi_tr:.4f}   (what Phase A used)")
    print(f"  focal alpha from val prior     : {1 - pi_va:.4f}   (deployment-matched)")
    print(f"  pos_weight from sampled prior  : {(1 - pi_tr) / max(pi_tr, 1e-9):.2f}")
    print(f"  pos_weight from val prior      : {(1 - pi_va) / max(pi_va, 1e-9):.2f}")

    if pi_tr > 0 and pi_va > 0:
        shift_logit = np.log((pi_tr / (1 - pi_tr)) / (pi_va / (1 - pi_va)))
        print(f"\n  implied logit bias        : {shift_logit:+.4f}")
        print(f"  prior-corrected inference : logit_adj = logit - {shift_logit:.4f}")
        print(f"  a raw p=0.50 becomes      : "
              f"{1 / (1 + np.exp(-(0.0 - shift_logit))):.4f} after correction")
        print("  (compare against the isotonic curve's observed rate at raw 0.50 —")
        print("   if they agree, prior shift explains the over-prediction outright.)")

    print("\n=== what to do ===")
    print("  The sampler gives every fire-fraction bin equal mass, so rare")
    print("  fire-heavy tiles are drawn as often as the huge near-empty bin.")
    print("  Options, cheapest first:")
    print("   1. Keep the sampler, but compute focal alpha / pos_weight from the")
    print("      UNWEIGHTED val prior so the loss is not double-correcting for")
    print("      an imbalance the sampler already fixed.")
    print("   2. Soften the sampler: weight by 1/sqrt(bin_count) instead of")
    print("      1/bin_count, or raise num_bins, to reduce the shift.")
    print("   3. Apply the logit bias above at inference (principled prior")
    print("      correction) instead of leaning entirely on isotonic calibration.")
    print("  These are alternatives, not a stack — 1 and 3 together would")
    print("  over-correct. Pick one and re-measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
