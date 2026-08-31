"""Fine-tune a checkpoint with rollout supervision on the events corpus.

The deployed model is trained on a single next-day transition and served as
three autoregressive steps, and it collapses across that gap: recall 0.284 at
one day, 0.132 at two, 0.020 at three. Each step feeds its own error back in
and nothing in training ever penalised that.

Fixing it needs ground truth for consecutive days of the same fire, which NDWS
does not have - its samples are independent single days at different places.
The events corpus does, because it is built from real multi-day sequences, so
this trains the rollout the service actually runs.

Two other things the corpus makes possible. Weather varies frame to frame here,
so wind_align_loss can push the forecast downwind instead of leaving direction
unconstrained by the objective. And splitting on fire_id keeps every day of a
fire on one side of the split, so validation cannot score a model on a fire it
memorised.

    python -m ignis_ml.scripts.finetune_rollout \
        --ckpt convlstm_unet_control60_delta_Cd13_Cs15_H64_T3.pt \
        --tiles /Volumes/T9/ignis-data/events_500m_T3 --out finetuned.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: Autoregressive steps to supervise. Three matches the served horizon.
ROLLOUT_STEPS = 3

#: Later steps carry less weight: their inputs are partly the model's own
#: output, so an error there is compounded rather than newly made.
STEP_WEIGHTS = (1.0, 0.6, 0.4)

#: Fraction of FIRES (not samples) held out. Splitting on samples would put
#: consecutive days of one fire on both sides, which is the leakage that made
#: v3 look better than it was.
VAL_FRACTION = 0.25


def npz_files(tiles: Path) -> List[Path]:
    """Sample files, excluding macOS AppleDouble sidecars.

    An external volume formatted for cross-platform use gets a ._name companion
    per file holding extended attributes. They match *.npz and are not npz.
    """
    return sorted(p for p in tiles.glob("*.npz") if not p.name.startswith("._"))


def load_corpus(tiles: Path) -> Dict[str, List[Path]]:
    """Sample paths grouped by fire, ordered by day."""
    by_fire: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for path in npz_files(tiles):
        with np.load(path, allow_pickle=True) as d:
            by_fire[str(d["fire_id"])].append((int(d["day_index"]), path))
    return {fire: [p for _, p in sorted(days)] for fire, days in by_fire.items()}


def rollout_groups(paths: Sequence[Path], steps: int) -> List[List[Path]]:
    """Consecutive runs of `steps` samples, for supervising an autoregression."""
    groups = []
    for i in range(len(paths) - steps + 1):
        groups.append(list(paths[i:i + steps]))
    return groups


def expand_statics(x_stat: np.ndarray, stat_order: Sequence[str]) -> np.ndarray:
    """Insert slope and aspect after elev, as the training dataset does.

    The corpus stores the 12 measured channels; the checkpoint expects 15,
    the extra three derived from elevation. Deriving them here rather than
    storing them keeps the corpus to what was actually observed.
    """
    from ignis_ml.src.data.dataset import _compute_slope_aspect

    order = list(stat_order)
    if "slope" not in order or x_stat.shape[0] == len(order):
        return x_stat
    slope, cos_a, sin_a = _compute_slope_aspect(x_stat[0], 500.0)
    return np.concatenate([x_stat[:1], slope[None], cos_a[None], sin_a[None],
                           x_stat[1:]], axis=0).astype(np.float32)


def prepare(path: Path, dyn_order: Sequence[str], cd: int,
            stat_order: Sequence[str] = ()):
    from ignis_ml.scripts.validate_checkpoint import normalize_dynamic
    from ignis_ml.src.data.features import append_derived_features

    with np.load(path, allow_pickle=True) as d:
        x_dyn = np.asarray(d["x_dyn"], dtype=np.float32)
        x_stat = expand_statics(np.asarray(d["x_stat"], dtype=np.float32), stat_order)
        y = np.asarray(d["y"], dtype=np.float32)
    raw_wind = (float(x_dyn[-1, 1].mean()), float(x_dyn[-1, 2].mean()))
    xn = normalize_dynamic(x_dyn, list(dyn_order)[:7])
    if xn.shape[1] < cd:
        xn, _ = append_derived_features(xn, dyn_order=list(dyn_order)[:7],
                                        include=None, days_since_fire_cap=None)
    return xn, x_stat, y, raw_wind


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch
    import torch.nn.functional as F

    from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet
    from ignis_ml.src.training.v4_losses import tversky_loss, wind_align_loss

    ap = argparse.ArgumentParser(prog="python -m ignis_ml.scripts.finetune_rollout")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wind-weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cd, cs = int(ck["cd"]), int(ck["cs"])
    dyn_order = list(ck.get("dyn_order") or
                     ["fire_t", "u", "v", "gust", "tempC", "q", "precip"])
    stat_order = list(ck.get("stat_order") or [])
    model = ConvLSTMUNet(Cd=cd, Cs=cs, hidden=int(ck.get("hidden", 64)),
                         drop=0.0, drop_decoder=0.0)
    model.load_state_dict(ck["state_dict"])

    by_fire = load_corpus(args.tiles)
    fires = sorted(by_fire)
    rng.shuffle(fires)
    n_val = max(1, int(len(fires) * VAL_FRACTION))
    val_fires, train_fires = set(fires[:n_val]), set(fires[n_val:])

    train_groups = [g for f in train_fires for g in rollout_groups(by_fire[f], ROLLOUT_STEPS)]
    val_groups = [g for f in val_fires for g in rollout_groups(by_fire[f], ROLLOUT_STEPS)]
    print(f"  {len(fires)} fires -> {len(train_fires)} train / {len(val_fires)} val")
    print(f"  rollout groups: {len(train_groups)} train / {len(val_groups)} val "
          f"({ROLLOUT_STEPS} steps each)")
    if not train_groups or not val_groups:
        print("  not enough consecutive days to supervise a rollout", file=sys.stderr)
        return 1

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    def run_group(group: Sequence[Path], train: bool) -> torch.Tensor:
        xn, stat, _, wind = prepare(group[0], dyn_order, cd, stat_order)
        x = torch.from_numpy(xn[None]).float()
        s = torch.from_numpy(stat[None]).float()
        total = torch.zeros(())
        for step, path in enumerate(group):
            _, _, y_np, wind = prepare(path, dyn_order, cd, stat_order)
            y = torch.from_numpy(y_np[None, None]).float()
            logits = model(x, s)
            base = F.binary_cross_entropy_with_logits(logits, y)
            loss = base + 0.6 * tversky_loss(logits, y, alpha=0.3, beta=0.7)
            if args.wind_weight > 0:
                fire_last = x[:, -1, 0:1]
                loss = loss + args.wind_weight * wind_align_loss(
                    logits, fire_last,
                    torch.tensor([wind[0]], dtype=torch.float32),
                    torch.tensor([wind[1]], dtype=torch.float32))
            total = total + STEP_WEIGHTS[min(step, len(STEP_WEIGHTS) - 1)] * loss
            # Autoregressive feedback: the prediction becomes the next fire
            # channel, which is exactly what the service does at serve time.
            nxt = x[:, -1].clone()
            nxt[:, 0] = torch.maximum(nxt[:, 0], torch.sigmoid(logits.detach())[:, 0])
            x = torch.cat([x[:, 1:], nxt[:, None]], dim=1)
        return total

    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_groups))
        train_loss = 0.0
        for idx in order:
            opt.zero_grad()
            loss = run_group(train_groups[idx], True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss)
        model.eval()
        with torch.no_grad():
            val_loss = sum(float(run_group(g, False)) for g in val_groups) / len(val_groups)
        train_loss /= len(train_groups)
        marker = ""
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"  epoch {epoch + 1:2d}  train {train_loss:.4f}  val {val_loss:.4f}{marker}")

    out = dict(ck)
    out["state_dict"] = best_state or model.state_dict()
    out["seq_len"] = int(ck.get("seq_len") or 3)
    out["finetune"] = {
        "base_checkpoint": args.ckpt.name,
        "base_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "corpus": str(args.tiles),
        "rollout_steps": ROLLOUT_STEPS,
        "step_weights": list(STEP_WEIGHTS),
        "wind_weight": args.wind_weight,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_loss": best,
        "fires_train": len(train_fires),
        "fires_val": len(val_fires),
    }
    out["split_stats"] = {"split": "group_aware", "grouped_on": "fire_id",
                          "val_fraction": VAL_FRACTION, "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\n  best val loss {best:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
