#!/usr/bin/env python3
"""train_v4.py — self-contained v4 trainer (Santa-Ana retrain).

Why a new file instead of editing train_nautilus.py:
  * train_nautilus.py imports `src.utils.config` which is not in this repo
    (the real Nautilus trainer lived in an unmounted folder), so it can't run
    here as-is, AND
  * it is mask-mode only — no delta target, no derived features, no Tversky.
This trainer is the complete v4 path and depends only on modules that exist:
  src/data/dataset.py, src/data/features.py, src/models/convlstm_unet.py,
  src/training/v4_losses.py, src/training/v4_sampler.py, src/utils/paths.py.

PRESS GO (after ingestion):  cd ignis_ml && python train_v4.py
Data/tiles/models follow $IGNIS_DATA_ROOT (external drive) automatically.

NOTE: requires a real torch + CUDA GPU + the ingested tiles to actually run.
It is import/syntax-clean here; it has not been executed end-to-end (no GPU/data
in this environment). Use --limit-files for a fast first sanity pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import NpzTileDataset
from src.models.convlstm_unet import ConvLSTMUNet, ARCH_VERSION
from src.training.v4_losses import v4_total_loss
from src.training.v4_sampler import build_santa_ana_sampler
from src.utils.paths import data_root, models_root, resolve_data_path, describe


# ----------------------------- config ------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def cfg_get(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------- helpers -----------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_tiles(cfg: Dict[str, Any]) -> List[Path]:
    """Glob npz tiles from every configured dataset, rebased to the data root."""
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
        else:
            print(f"[data] {name}: tiles dir not found ({p}) — skipping")
    return files


def _dilate(y: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return y
    k = 2 * px + 1
    return F.max_pool2d(y, kernel_size=k, stride=1, padding=px)


def base_bce_dice(logits, y, *, pos_weight, dice_lambda, dilate_px) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    yd = _dilate(y, int(dilate_px))
    p = torch.sigmoid(logits)
    inter = (p * yd).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1e-6) / (p.sum(dim=(1, 2, 3)) + yd.sum(dim=(1, 2, 3)) + 1e-6)
    return bce + dice_lambda * dice.mean()


def resize_to_target(logits, y):
    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def estimate_pos_prior(loader, max_batches=30) -> float:
    pos = tot = 0.0
    for i, batch in enumerate(loader):
        y = batch[2]  # delta target
        pos += float((y > 0.5).sum()); tot += float(y.numel())
        if i + 1 >= max_batches:
            break
    return pos / max(tot, 1.0)


@torch.no_grad()
def eval_csi(model, loader, device, thresholds: List[float]) -> Tuple[float, float]:
    """Return (best_threshold, best_CSI) over the val set."""
    model.eval()
    tp = {t: 0.0 for t in thresholds}; fp = dict(tp); fn = dict(tp)
    for batch in loader:
        x_d, x_s, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits = resize_to_target(model(x_d, x_s), y)
        p = torch.sigmoid(logits)
        for t in thresholds:
            pb = (p >= t)
            yb = (y >= 0.5)
            tp[t] += float((pb & yb).sum())
            fp[t] += float((pb & ~yb).sum())
            fn[t] += float((~pb & yb).sum())
    best_t, best_csi = thresholds[0], -1.0
    for t in thresholds:
        denom = tp[t] + fp[t] + fn[t]
        csi = tp[t] / denom if denom > 0 else 0.0
        if csi > best_csi:
            best_csi, best_t = csi, t
    return best_t, best_csi


def export_calibration(model, loader, device, ckpt_path: Path, out_path: Path):
    """Fit isotonic calibration on (raw_prob, observed) and save JSON.

    Skips gracefully if scikit-learn is unavailable.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        print("[calib] scikit-learn not installed — skipping calibration export")
        return
    probs, obs = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_d, x_s, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            p = torch.sigmoid(resize_to_target(model(x_d, x_s), y))
            probs.append(p.flatten().cpu().numpy())
            obs.append((y >= 0.5).flatten().cpu().numpy().astype(np.float32))
            if i + 1 >= 40:
                break
    if not probs:
        return
    pr = np.concatenate(probs); ob = np.concatenate(obs)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pr, ob)
    xs = np.linspace(0, 1, 101)
    ys = iso.predict(xs)
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else None
    out_path.write_text(json.dumps({
        "method": "isotonic",
        "model_sha256": sha,
        "points": [[float(a), float(b)] for a, b in zip(xs, ys)],
        "risk_breaks": [["low", 0.05], ["medium", 0.20], ["high", 0.50], ["extreme", 1.01]],
    }, indent=2))
    print(f"[calib] wrote {out_path}")


# ------------------------------- main ------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="IgnisAI v4 trainer")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.v4.yaml"))
    ap.add_argument("--limit-files", type=int, default=0, help="cap tiles for a fast sanity run")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    print(f"[paths] {describe()}")
    set_seed(int(cfg_get(cfg, "training.seed", 42)))
    device = get_device()
    use_amp = (device == "cuda") and bool(cfg_get(cfg, "training.mixed_precision", True))
    print(f"[v4] device={device} amp={use_amp}")

    dyn_order = cfg_get(cfg, "channels.dynamic_order")
    stat_order = cfg_get(cfg, "channels.static_order")
    seq_len = int(cfg_get(cfg, "training.seq_len", 6))
    res_m = float(cfg_get(cfg, "grid.res_m", 500))

    files = collect_tiles(cfg)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        print("[v4] no tiles found. Run ingestion first "
              "(python -m ignis_ml.scripts.ingest_ts_satfire).")
        return 1

    rng = np.random.default_rng(int(cfg_get(cfg, "training.seed", 42)))
    idx = np.arange(len(files)); rng.shuffle(idx)
    n_val = max(1, int(len(files) * 0.15))
    val_files = [files[i] for i in idx[:n_val]]
    train_files = [files[i] for i in idx[n_val:]]
    print(f"[v4] train={len(train_files)} val={len(val_files)}")

    aug = cfg.get("augmentation", {})
    ds_common = dict(
        expected_dyn_order=dyn_order, expected_stat_order=stat_order,
        seq_len=seq_len, res_m=res_m, normalize=True, fire_boost=5.0,
        compute_slope_aspect=True,
        derived_features_enable=bool(cfg_get(cfg, "features.derived.enable", True)),
        derived_features_include=cfg_get(cfg, "features.derived.include", None),
        days_since_fire_cap=cfg_get(cfg, "features.days_since_fire_cap", None),
        target_mode="delta",
    )
    ds_tr = NpzTileDataset(
        train_files, augment=True,
        channel_dropout_p=float(aug.get("channel_dropout_p", 0.0)),
        channel_dropout_exclude=aug.get("channel_dropout_exclude"),
        gaussian_noise_std=float(aug.get("gaussian_noise_std", 0.0)),
        temporal_dropout_p=float(aug.get("temporal_dropout_p", 0.0)),
        **ds_common,
    )
    ds_va = NpzTileDataset(val_files, augment=False, **ds_common)

    meta_dir = cfg_get(cfg, "datasets.tssatfire.meta_dir")
    meta_dir = resolve_data_path(meta_dir) if meta_dir else None
    sampler, sa_stats = build_santa_ana_sampler(
        train_files, meta_dir=meta_dir,
        santa_ana_boost=float(cfg_get(cfg, "training.sampling.santa_ana_boost", 5.0)),
        seed=int(cfg_get(cfg, "training.seed", 42)),
    )
    print(f"[v4] sampler: {sa_stats}")

    bs = int(cfg_get(cfg, "training.batch_size", 8))
    nw = int(cfg_get(cfg, "training.num_workers", 4))
    pin = (device == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=bs, sampler=sampler, num_workers=nw,
                       pin_memory=pin, persistent_workers=(nw > 0))
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw,
                       pin_memory=pin, persistent_workers=(nw > 0))

    # Infer channel counts from a real batch (derived features included).
    first = next(iter(dl_tr))
    x_d0, x_s0 = first[0], first[1]
    Cd, Cs, tile = x_d0.shape[2], x_s0.shape[1], x_d0.shape[-1]
    print(f"[v4] Cd={Cd} Cs={Cs} tile={tile} (arch={ARCH_VERSION})")

    hidden = int(cfg_get(cfg, "training.hidden_channels", 64))
    model = ConvLSTMUNet(
        Cd=Cd, Cs=Cs, hidden=hidden,
        drop=float(cfg_get(cfg, "training.drop_encoder", 0.10)),
        drop_decoder=float(cfg_get(cfg, "training.drop_decoder", 0.15)),
    ).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    prior = estimate_pos_prior(dl_tr)
    pw = min(15.0, (1 - prior) / max(prior, 1e-5))
    pos_weight = torch.tensor([pw], device=device)
    print(f"[v4] delta pos prior={prior:.5f} pos_weight={pw:.2f}")

    epochs = args.epochs or int(cfg_get(cfg, "training.epochs", 30))
    warmup = int(cfg_get(cfg, "training.warmup_epochs", 5))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_get(cfg, "training.lr", 8e-4)),
                            weight_decay=float(cfg_get(cfg, "training.weight_decay", 5e-4)))

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    dice_lambda = float(cfg_get(cfg, "training.loss.dice_weight", 0.6))
    dilate_px = int(cfg_get(cfg, "training.loss.dice_dilate_px", 2))
    tversky_cfg = cfg_get(cfg, "training.loss.tversky")
    wind_cfg = cfg_get(cfg, "training.loss.wind_align")
    fire_idx = dyn_order.index("fire_t")
    u_idx, v_idx = dyn_order.index("u"), dyn_order.index("v")

    sweep = cfg_get(cfg, "training.metrics.threshold_sweep", {"lo": 0.05, "hi": 0.5, "n": 46})
    thresholds = list(np.linspace(sweep["lo"], sweep["hi"], int(sweep["n"])))

    grad_clip = float(cfg_get(cfg, "training.grad_clip", 1.0))
    early_stop = int(cfg_get(cfg, "training.early_stop", 6))
    best_csi, best_t, patience = -1.0, thresholds[0], 0
    out_dir = models_root(); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"convlstm_unet_{ARCH_VERSION}_delta_Cd{Cd}_Cs{Cs}_H{hidden}_T{seq_len}.pt"

    print(f"[v4] training {epochs} epochs -> {ckpt_path}")
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for batch in dl_tr:
            x_d = batch[0].to(device, non_blocking=True)
            x_s = batch[1].to(device, non_blocking=True)
            y = batch[2].to(device, non_blocking=True)            # delta target
            # de-normalize wind ([0,1] from [-15,15]) -> m/s for wind-align
            u_mean = x_d[:, :, u_idx].mean(dim=(1, 2, 3)) * 30.0 - 15.0
            v_mean = x_d[:, :, v_idx].mean(dim=(1, 2, 3)) * 30.0 - 15.0
            fire_last = x_d[:, -1, fire_idx:fire_idx + 1]         # [B,1,H,W]

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = resize_to_target(model(x_d, x_s), y)
                base = base_bce_dice(logits, y, pos_weight=pos_weight,
                                     dice_lambda=dice_lambda, dilate_px=dilate_px)
                loss = v4_total_loss(
                    logits, y, base_loss=base,
                    tversky_cfg=tversky_cfg, wind_cfg=wind_cfg,
                    fire_last=fire_last, u_mean=u_mean, v_mean=v_mean,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            run += float(loss.item())
        sch.step()

        t, csi = eval_csi(model, dl_va, device, thresholds)
        print(f"[v4] epoch {ep}/{epochs} loss={run/max(len(dl_tr),1):.4f} "
              f"valCSI={csi:.3f}@{t:.2f} ({time.time()-t0:.0f}s)")

        if csi > best_csi + 1e-4:
            best_csi, best_t, patience = csi, t, 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch_version": ARCH_VERSION,
                "cd": Cd, "cs": Cs, "tile_size": tile, "hidden": hidden,
                "dyn_order": dyn_order, "stat_order": stat_order,
                "target_mode": "delta",
                "best_threshold": best_t, "val_csi": best_csi,
                "santa_ana_stats": sa_stats,
            }, ckpt_path)
            print(f"  ✓ saved {ckpt_path.name} (CSI={best_csi:.3f}, th={best_t:.2f})")
        else:
            patience += 1
            if patience >= early_stop:
                print(f"[v4] early stop at epoch {ep} (best CSI={best_csi:.3f})")
                break

    # Calibration JSON keyed to the best checkpoint.
    export_calibration(model, dl_va, device, ckpt_path,
                       out_dir / f"calibration_{ARCH_VERSION}.json")
    print(f"[v4] done. best CSI={best_csi:.3f}@{best_t:.2f}  ckpt={ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
