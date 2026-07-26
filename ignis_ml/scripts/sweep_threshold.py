#!/usr/bin/env python3
"""Re-score a trained checkpoint across a FULL threshold range.

Motivation: `train_v4.py` picks `best_threshold` from a sweep whose bounds come
from `training.metrics.threshold_sweep` in the config. If the optimum sits at a
bound, the reported CSI is an underestimate — the true optimum was clipped. This
already happened once in v3 (config.yaml comments: "run #1 best threshold was
pegged at 0.70, top of sweep"), and the Phase A run came back with
best_threshold == 0.50 exactly, which is the ceiling of the v4-style
`lo: 0.05, hi: 0.50` sweep.

This script re-evaluates a SAVED checkpoint — no retraining — over 0..1 and
reports the full precision/recall/CSI/F1 curve, so you can see where the real
optimum is and whether the training sweep clipped it.

Method: one forward pass over the val split, accumulating a probability
histogram split by label. Every threshold metric is then computed exactly from
that histogram, so memory is O(bins) rather than O(pixels) and the sweep
resolution is free.

Reproduces the training val split exactly (same seed, same glob order, same
0.15 fraction as train_v4.py), so numbers are directly comparable.

Usage
-----
    python -m ignis_ml.scripts.sweep_threshold \
        --ckpt "$IGNIS_DATA_ROOT/models/convlstm_unet_phase_a_delta_Cd13_Cs15_H64_T3.pt" \
        --config config.phase_a.yaml

    # apply the isotonic calibration before thresholding
    python -m ignis_ml.scripts.sweep_threshold --ckpt ... --config ... \
        --calibration "$IGNIS_DATA_ROOT/models/calibration_phase_a.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

NBINS = 2001          # probability histogram resolution (~0.0005 steps)


# ---------------------------------------------------------------------------
# config / split reproduction (mirrors train_v4.py exactly)
# ---------------------------------------------------------------------------
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


def val_split(files: List[Path], seed: int) -> List[Path]:
    """Byte-identical to train_v4.py's split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * 0.15))
    return [files[i] for i in idx[:n_val]]


# ---------------------------------------------------------------------------
# metric computation from a label-split probability histogram
# ---------------------------------------------------------------------------
def metrics_from_hist(pos: np.ndarray, neg: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact P/R/CSI/F1 at every bin edge, computed from the histogram.

    Threshold at bin b means "predict positive if prob >= edge[b]".
    """
    # reverse-cumulative: counts at or above each bin
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    total_pos = pos.sum()
    fn = total_pos - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(total_pos > 0, tp / total_pos, 0.0)
        csi = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return {
        "edges": np.linspace(0.0, 1.0, len(pos)),
        "precision": precision, "recall": recall, "csi": csi, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    """Step-wise AP from the histogram (same definition sklearn uses)."""
    m = metrics_from_hist(pos, neg)
    r, p = m["recall"], m["precision"]
    # recall is non-increasing as threshold rises; walk from high->low threshold
    # so recall increases, and sum precision * delta-recall.
    r_asc = r[::-1]
    p_asc = p[::-1]
    dr = np.diff(np.concatenate([[0.0], r_asc]))
    return float((p_asc * dr).sum())


def apply_calibration(p: np.ndarray, points: List[List[float]]) -> np.ndarray:
    xs = np.array([a for a, _ in points], dtype=np.float32)
    ys = np.array([b for _, b in points], dtype=np.float32)
    return np.interp(p, xs, ys).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    from ignis_ml.src.data.dataset import NpzTileDataset
    from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet
    from ignis_ml.src.utils.paths import describe

    ap = argparse.ArgumentParser(description="Full-range threshold re-sweep of a checkpoint")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.v4.yaml")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="apply this isotonic calibration JSON before thresholding")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit-batches", type=int, default=0, help="0 = all (quick check: 50)")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--out", type=Path, default=None, help="write full curve as CSV")
    args = ap.parse_args(argv)

    print(f"[sweep] {describe()}")
    cfg = load_yaml(args.config)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[sweep] device={device}")

    # ---- checkpoint drives the architecture, config drives the data ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    Cd, Cs = int(ck["cd"]), int(ck["cs"])
    hidden = int(ck.get("hidden", 64))
    dyn_order = ck.get("dyn_order") or cfg_get(cfg, "channels.dynamic_order")
    stat_order = ck.get("stat_order") or cfg_get(cfg, "channels.static_order")
    print(f"[sweep] ckpt arch={ck.get('arch_version')} Cd={Cd} Cs={Cs} hidden={hidden}")
    print(f"[sweep] ckpt reported: val_ap={ck.get('val_ap')} val_csi={ck.get('val_csi')} "
          f"best_threshold={ck.get('best_threshold')}")

    seq_len = int(cfg_get(cfg, "training.seq_len", 6))
    res_m = float(cfg_get(cfg, "grid.res_m", 500))
    seed = int(cfg_get(cfg, "training.seed", 42))

    files = collect_tiles(cfg)
    if not files:
        print("[sweep] no tiles found — check datasets.*.tiles_dir and IGNIS_DATA_ROOT")
        return 1
    vfiles = val_split(files, seed)
    print(f"[sweep] val tiles={len(vfiles)} (seed={seed}, 15% split — matches training)")

    ds = NpzTileDataset(
        vfiles, augment=False,
        expected_dyn_order=dyn_order, expected_stat_order=stat_order,
        seq_len=seq_len, res_m=res_m, normalize=True, fire_boost=5.0,
        compute_slope_aspect=True,
        derived_features_enable=bool(cfg_get(cfg, "features.derived.enable", True)),
        derived_features_include=cfg_get(cfg, "features.derived.include", None),
        days_since_fire_cap=cfg_get(cfg, "features.days_since_fire_cap", None),
        target_mode="delta",
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=hidden, drop=0.0, drop_decoder=0.0).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    calib = None
    if args.calibration and args.calibration.exists():
        calib = json.loads(args.calibration.read_text())["points"]
        print(f"[sweep] applying calibration from {args.calibration.name}")

    # ---- single pass: accumulate label-split probability histograms --------
    # delta: the trained target.  full: delta OR persistence (operational mask).
    pos_d = np.zeros(NBINS, dtype=np.float64); neg_d = np.zeros(NBINS, dtype=np.float64)
    pos_f = np.zeros(NBINS, dtype=np.float64); neg_f = np.zeros(NBINS, dtype=np.float64)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            x_d, x_s, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            persist = batch[3].to(device) if len(batch) > 3 else torch.zeros_like(y)

            logits = model(x_d, x_s)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:],
                                       mode="bilinear", align_corners=False)
            p = torch.sigmoid(logits).flatten().cpu().numpy().astype(np.float32)
            yb = (y >= 0.5).flatten().cpu().numpy()
            pb = (persist >= 0.5).flatten().cpu().numpy()

            if calib is not None:
                p = apply_calibration(p, calib)

            bins = np.clip((p * (NBINS - 1)).astype(np.int32), 0, NBINS - 1)
            pos_d += np.bincount(bins[yb], minlength=NBINS)
            neg_d += np.bincount(bins[~yb], minlength=NBINS)

            # full-mask view: a pixel already burning is predicted burning at
            # any threshold, and is positive in truth.
            y_full = yb | pb
            p_full_bins = np.where(pb, NBINS - 1, bins)
            pos_f += np.bincount(p_full_bins[y_full], minlength=NBINS)
            neg_f += np.bincount(p_full_bins[~y_full], minlength=NBINS)

            if args.limit_batches and i + 1 >= args.limit_batches:
                break
            if (i + 1) % 50 == 0:
                print(f"  ..{i + 1} batches")

    # ---- report ------------------------------------------------------------
    def report(tag: str, pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float]:
        m = metrics_from_hist(pos, neg)
        apv = average_precision(pos, neg)
        prevalence = pos.sum() / max(pos.sum() + neg.sum(), 1)
        best = int(np.argmax(m["csi"]))
        best_t, best_csi = float(m["edges"][best]), float(m["csi"][best])
        bf = int(np.argmax(m["f1"]))

        print(f"\n=== {tag} ===")
        print(f"  positive prevalence : {prevalence:.4f}   (AP of a random ranker)")
        print(f"  Average Precision   : {apv:.4f}   "
              f"({'ABOVE' if apv > prevalence else 'BELOW'} trivial, "
              f"lift x{apv / max(prevalence, 1e-9):.2f})")
        print(f"  best CSI            : {best_csi:.4f} @ threshold {best_t:.3f}")
        print(f"  best F1             : {float(m['f1'][bf]):.4f} @ threshold {float(m['edges'][bf]):.3f}")
        if best_t <= 0.02 or best_t >= 0.98:
            print(f"  ⚠  optimum at the edge of [0,1] — model probabilities are degenerate")
        print(f"\n  {'thresh':>7} {'prec':>7} {'recall':>7} {'CSI':>7} {'F1':>7}")
        for t in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            j = int(round(t * (NBINS - 1)))
            print(f"  {t:>7.2f} {m['precision'][j]:>7.4f} {m['recall'][j]:>7.4f} "
                  f"{m['csi'][j]:>7.4f} {m['f1'][j]:>7.4f}")
        return best_t, best_csi

    best_t, best_csi = report("DELTA target (what the model was trained on)", pos_d, neg_d)
    report("FULL mask (delta OR persistence — operational view)", pos_f, neg_f)

    # ---- did the training sweep clip the optimum? --------------------------
    lo = float(cfg_get(cfg, "training.metrics.threshold_sweep.lo", 0.05))
    hi = float(cfg_get(cfg, "training.metrics.threshold_sweep.hi", 0.50))
    print(f"\n=== sweep-ceiling check ===")
    print(f"  config sweep range   : [{lo:.2f}, {hi:.2f}]")
    print(f"  true optimum (CSI)   : {best_t:.3f}")
    reported = ck.get("val_csi")
    if best_t >= hi - 1e-6:
        print(f"  ⚠  CLIPPED: the optimum sits at/above the sweep ceiling ({hi:.2f}).")
        print(f"     Training reported CSI={reported} — that is an UNDERESTIMATE.")
        print(f"     Widen training.metrics.threshold_sweep.hi to ~0.95 and re-run eval.")
    elif best_t <= lo + 1e-6:
        print(f"  ⚠  CLIPPED at the sweep FLOOR ({lo:.2f}). Lower training.metrics."
              f"threshold_sweep.lo.")
    else:
        print(f"  ✓ optimum is interior to the sweep — the reported threshold is real.")
    if reported is not None:
        print(f"  reported CSI {float(reported):.4f} -> true best CSI {best_csi:.4f} "
              f"(delta {best_csi - float(reported):+.4f})")

    # ---- optional CSV ------------------------------------------------------
    if args.out:
        m = metrics_from_hist(pos_d, neg_d)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            fh.write("threshold,precision,recall,csi,f1,tp,fp,fn\n")
            for j in range(NBINS):
                fh.write(f"{m['edges'][j]:.5f},{m['precision'][j]:.6f},"
                         f"{m['recall'][j]:.6f},{m['csi'][j]:.6f},{m['f1'][j]:.6f},"
                         f"{m['tp'][j]:.0f},{m['fp'][j]:.0f},{m['fn'][j]:.0f}\n")
        print(f"\n[sweep] full curve -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
