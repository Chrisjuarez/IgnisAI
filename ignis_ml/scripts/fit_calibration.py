"""Fit a probability calibration for a checkpoint, and report how good it is.

A raw sigmoid is not a probability. The network is trained to rank cells, and
nothing in that objective makes "0.7" mean "seven of ten such cells burn".
Serving currently ships no calibration at all, so every probability the app
displays - and anything an underwriter would price off - is an uncalibrated
score.

This fits the mapping that fixes that, using isotonic regression: monotone, so
it never reorders the model's own ranking, and non-parametric, so it does not
impose a shape the residuals do not have.

It also reports the numbers that say whether the result can be trusted -
Brier score, its skill against the base rate, and expected calibration error -
because a calibration fitted on too little data is worse than none: it looks
authoritative and is not.

    python -m ignis_ml.scripts.fit_calibration --ckpt model.pt --event palisades
"""
from __future__ import annotations

import argparse
import hashlib
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ignis_ml.scripts.validate_checkpoint import EVENTS, normalize_dynamic


#: Cells that are already alight carry no information about where a fire will
#: spread - the model predicts new growth, so scoring them would let a large
#: burn scar dominate the fit with cells whose outcome was never in question.
EXCLUDE_PRIOR_FIRE = True

#: Below this many labelled cells the fit is reported but marked provisional.
MIN_SAMPLES_FOR_TRUST = 200_000


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic regression, weighted by bin count.

    Written out rather than pulled from sklearn so the produced curve has no
    dependency the serving image would need in order to be re-derived.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order].astype(np.float64), y[order].astype(np.float64)

    values = list(ys)
    weights = [1.0] * len(ys)
    positions = list(range(len(ys)))

    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        merged = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total
        values[i:i + 2] = [merged]
        weights[i:i + 2] = [total]
        positions[i:i + 2] = [positions[i]]
        if i > 0:
            i -= 1
    fitted = np.repeat(values, [int(w) for w in weights])
    return xs, fitted


def reliability_bins(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        n = int(m.sum())
        if not n:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "predicted": None, "observed": None})
            continue
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "predicted": float(prob[m].mean()),
            "observed": float(label[m].mean()),
        })
    return out


def expected_calibration_error(bins: Sequence[Dict[str, Any]]) -> float:
    total = sum(b["n"] for b in bins)
    if not total:
        return float("nan")
    return float(sum(
        b["n"] / total * abs(b["observed"] - b["predicted"])
        for b in bins if b["n"]
    ))


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((prob - label) ** 2))


def brier_skill_score(prob: np.ndarray, label: np.ndarray) -> float:
    """Against always predicting the base rate. Above 0 beats that, below loses."""
    base = float(label.mean())
    reference = float(np.mean((base - label) ** 2))
    if reference <= 0:
        return float("nan")
    return 1.0 - brier(prob, label) / reference


def curve_from_pairs(prob: np.ndarray, label: np.ndarray, knots: int = 32) -> List[List[float]]:
    """Thin the isotonic fit to a small monotone lookup the service interpolates."""
    xs, fitted = isotonic_fit(prob, label)
    # Quantile knots alone put every knot where the mass is, and in a fire
    # raster 97% of cells sit near zero - the curve came out with points at
    # 0.05 and then nothing until 0.99, straight through the range the model
    # actually makes decisions in. Uniform knots cover that range; quantile
    # knots keep resolution in the crowded low end. Use both.
    qs = np.linspace(0.0, 1.0, knots)
    raw = np.unique(np.concatenate([np.quantile(xs, qs), np.linspace(0.0, 1.0, knots)]))
    mapped = np.interp(raw, xs, fitted)
    # Force the ends and strip duplicate x, which np.interp needs to be strictly
    # increasing to behave.
    raw = np.concatenate(([0.0], raw, [1.0]))
    mapped = np.concatenate(([0.0], mapped, [max(1.0, float(mapped.max()))]))
    mapped = np.maximum.accumulate(np.clip(mapped, 0.0, 1.0))
    keep = np.concatenate(([True], np.diff(raw) > 1e-6))
    return [[float(a), float(b)] for a, b in zip(raw[keep], mapped[keep])]


def _event_env(event: str) -> None:
    """Point the builders at this event's cached FIRMS and HRRR, as the validator does."""
    snap = _REPO / ".cache" / "runtime_cache" / event / "firms_snapshots"
    if snap.is_dir():
        os.environ.setdefault("FIRMS_SNAPSHOT_DIR", str(snap))
        os.environ.setdefault("FIRMS_SNAPSHOT_REQUIRED", "1")
    noaa = _REPO / ".cache" / "runtime_cache" / event / "noaa_grid_cache"
    if noaa.is_dir():
        os.environ.setdefault("NOAA_GRID_CACHE_DIR", str(noaa))
        os.environ.setdefault("NOAA_GRIB_ENABLED", "1")


def collect_pairs(ckpt_path: Path, event: str, *, seq_len: Optional[int],
                  days: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Predicted probability against what actually burned, one pair per cell.

    The label for a forecast made at day D is the observed fire mask at D+1,
    read from the same FIRMS snapshots that feed the model, so prediction and
    outcome come from one source and one grid.
    """
    import torch

    from ignis_ml.src.data.features import append_derived_features
    from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet
    from services.tilesvc.dynamic_builder import build_dynamic_for_tile
    from services.tilesvc.grid import lonlat_to_tile
    from services.tilesvc.static_catalog import load_static_tensor_for_model

    _event_env(event)
    ev = EVENTS[event]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    Cd, Cs = int(ck["cd"]), int(ck["cs"])
    hidden = int(ck.get("hidden", 64))
    T = seq_len or int(ck.get("seq_len", 6))
    dyn_order = list(ck.get("dyn_order") or
                     ["fire_t", "u", "v", "gust", "tempC", "q", "precip"])
    stat_order = list(ck.get("stat_order") or [])

    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=hidden, drop=0.0, drop_decoder=0.0)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    tile = lonlat_to_tile(ev["lon"], ev["lat"])
    stat, _ = load_static_tensor_for_model(tile, stat_order)
    stat = np.asarray(stat, dtype=np.float32)

    base = dt.datetime.fromisoformat(ev["ref_time"].replace("Z", "+00:00"))
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    used: List[str] = []

    # Walk backward from the event reference time. The label for a forecast at
    # day D is the observed mask at D+1, so the last scorable day is the one
    # before the newest snapshot - stepping forward from the reference just
    # asks for days the cache does not have.
    for offset in range(1, days + 1):
        at = base - dt.timedelta(days=offset)
        nxt = at + dt.timedelta(days=1)
        try:
            x = np.asarray(build_dynamic_for_tile(
                ev["lat"], ev["lon"], T_seq=T, hours_step=24,
                ignition=ev["ignition"], ref_time=at, channel_order=dyn_order[:7],
            ), dtype=np.float32)
            after = np.asarray(build_dynamic_for_tile(
                ev["lat"], ev["lon"], T_seq=1, hours_step=24,
                ignition=False, ref_time=nxt, channel_order=dyn_order[:7],
            ), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - a missing day is not fatal
            print(f"  {at:%Y-%m-%d}: skipped ({type(exc).__name__})")
            continue

        prior = x[-1, 0] >= 0.5
        burned = after[-1, 0] >= 0.5
        if not burned.any():
            print(f"  {at:%Y-%m-%d}: skipped (no observed fire the next day)")
            continue

        xn = normalize_dynamic(x, dyn_order[:7])
        if xn.shape[1] < Cd:
            xn, _ = append_derived_features(xn, dyn_order=dyn_order[:7],
                                            include=None, days_since_fire_cap=None)
        with torch.no_grad():
            logits = model(torch.from_numpy(xn[None]).float(),
                           torch.from_numpy(stat[None]).float())
            prob = torch.sigmoid(logits)[0, 0].numpy()

        keep = ~prior if EXCLUDE_PRIOR_FIRE else np.ones_like(prior)
        probs.append(prob[keep].ravel())
        labels.append(burned[keep].ravel().astype(np.float32))
        used.append(f"{at:%Y-%m-%d}")
        print(f"  {at:%Y-%m-%d}: {int(keep.sum())} cells, "
              f"{int(burned[keep].sum())} burned next day")

    if not probs:
        raise SystemExit("no usable day produced a labelled pair")

    meta = {
        "event": event,
        "days": used,
        "seq_len": T,
        "excluded_prior_fire": EXCLUDE_PRIOR_FIRE,
        "model_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
    }
    return np.concatenate(probs), np.concatenate(labels), meta


def report(prob: np.ndarray, label: np.ndarray, mapped: np.ndarray) -> Dict[str, Any]:
    raw_bins = reliability_bins(prob, label)
    cal_bins = reliability_bins(mapped, label)
    return {
        "n": int(prob.size),
        "base_rate": float(label.mean()),
        "raw": {
            "brier": brier(prob, label),
            "brier_skill": brier_skill_score(prob, label),
            "ece": expected_calibration_error(raw_bins),
            "bins": raw_bins,
        },
        "calibrated": {
            "brier": brier(mapped, label),
            "brier_skill": brier_skill_score(mapped, label),
            "ece": expected_calibration_error(cal_bins),
            "bins": cal_bins,
        },
    }


def print_reliability(title: str, bins: Sequence[Dict[str, Any]]) -> None:
    print(f"\n  {title}")
    print(f"  {'bin':>12} {'n':>9} {'predicted':>10} {'observed':>9}  {'':<22}")
    for b in bins:
        if not b["n"]:
            continue
        # A bar of the gap: right of centre means the model over-predicts.
        gap = b["predicted"] - b["observed"]
        width = int(min(abs(gap), 0.5) * 40)
        bar = ("over  " + "#" * width) if gap > 0 else ("under " + "#" * width)
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}   {b['n']:>9,} {b['predicted']:>10.4f} "
              f"{b['observed']:>9.4f}  {bar}")


def collect_pairs_from_tiles(ckpt_path: Path, tiles: Path, *, seq_len: Optional[int],
                             limit: int, holdout: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Predicted probability against the tile's own next-day mask.

    This is the source calibration should normally use. Each ETL tile already
    carries y, the observed next-day fire, on the same grid as its inputs, so
    there is no alignment step to get wrong and no dependence on a live FIRMS
    window that only reaches a few days back. Fitting on a held-out slice keeps
    the curve honest: a calibration fitted on the training split measures how
    well the model memorised, not how well it forecasts.
    """
    import torch
    from torch.utils.data import DataLoader

    from ignis_ml.src.data.dataset import NpzTileDataset
    from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    Cd, Cs = int(ck["cd"]), int(ck["cs"])
    hidden = int(ck.get("hidden", 64))
    T = seq_len or int(ck.get("seq_len", 6))
    dyn_order = list(ck.get("dyn_order") or
                     ["fire_t", "u", "v", "gust", "tempC", "q", "precip"])
    stat_order = list(ck.get("stat_order") or [])

    # Match the checkpoint's own training setup, or the fit describes a
    # different model: derived channels make up the difference between the 7
    # stored dynamics and the Cd the network expects, and a delta checkpoint
    # predicts NEW growth, so its label is y_delta rather than the full mask.
    target_mode = str(ck.get("target_mode") or "mask")
    dataset = NpzTileDataset(
        tiles, augment=False, expected_dyn_order=dyn_order[:7],
        expected_stat_order=stat_order or None, seq_len=T,
        derived_features_enable=True, target_mode=target_mode,
    )
    total = len(dataset)
    if not total:
        raise SystemExit(f"no tiles found under {tiles}")

    # Deterministic tail slice: the same tiles every run, and disjoint from a
    # head-slice training split.
    start = int(total * (1.0 - holdout))
    indices = list(range(start, total))[:limit]
    subset = torch.utils.data.Subset(dataset, indices)

    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=hidden, drop=0.0, drop_decoder=0.0)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=8):
            x_dyn, x_stat, y = batch[0], batch[1], batch[2]
            prob = torch.sigmoid(model(x_dyn.float(), x_stat.float()))[:, 0].numpy()
            label = y[:, 0].numpy()
            if EXCLUDE_PRIOR_FIRE:
                prior = x_dyn[:, -1, 0].numpy() >= 0.5
                for p_i, l_i, pr_i in zip(prob, label, prior):
                    probs.append(p_i[~pr_i].ravel())
                    labels.append(l_i[~pr_i].ravel())
            else:
                probs.append(prob.ravel())
                labels.append(label.ravel())

    meta = {
        "source": "ndws_tiles",
        "tiles_dir": str(tiles),
        "tiles_scored": len(indices),
        "tiles_total": total,
        "holdout_fraction": holdout,
        "seq_len": T,
        "target_mode": target_mode,
        "excluded_prior_fire": EXCLUDE_PRIOR_FIRE,
        # Hash the file, do not read a key from it. Checkpoints written by
        # training do not carry their own sha, so this was writing null - and
        # a calibration with a null sha passes the serving guard for ANY
        # checkpoint, which is the exact mismatch the guard exists to catch.
        "model_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
    }
    print(f"  scored {len(indices)} held-out tiles of {total}")
    return np.concatenate(probs), np.concatenate(labels), meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ignis_ml.scripts.fit_calibration")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--event", default="palisades", choices=sorted(EVENTS))
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--days", type=int, default=6,
                    help="Event mode: how many consecutive forecast days to score")
    ap.add_argument("--tiles", type=Path, default=None,
                    help="ETL tile directory. Preferred: each tile carries its own "
                         "next-day mask, so the fit is not limited to the few days "
                         "a live FIRMS window still covers.")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="Tail fraction of tiles to fit on, disjoint from training")
    ap.add_argument("--limit", type=int, default=2000,
                    help="Cap on held-out tiles scored")
    ap.add_argument("--out", type=Path, default=Path("config/calibration.json"))
    args = ap.parse_args(argv)

    if args.tiles:
        print(f"collecting labelled pairs from held-out tiles in {args.tiles}")
        prob, label, meta = collect_pairs_from_tiles(
            args.ckpt, args.tiles, seq_len=args.seq_len,
            limit=args.limit, holdout=args.holdout)
    else:
        print(f"collecting labelled pairs for {args.event} from {args.ckpt.name}")
        prob, label, meta = collect_pairs(args.ckpt, args.event,
                                          seq_len=args.seq_len, days=args.days)

    points = curve_from_pairs(prob, label)
    raw = np.array([p[0] for p in points])
    mapped_curve = np.array([p[1] for p in points])
    mapped = np.interp(prob, raw, mapped_curve)

    metrics = report(prob, label, mapped)
    scored_units = meta.get("tiles_scored", len(meta.get("days", [])))
    trustworthy = metrics["n"] >= MIN_SAMPLES_FOR_TRUST and scored_units >= 3

    print(f"\n  labelled cells : {metrics['n']:,}")
    print(f"  base rate      : {metrics['base_rate']:.4f}  "
          f"({int(metrics['base_rate'] * metrics['n']):,} burned)")
    print(f"\n  {'':<14}{'Brier':>10}{'skill':>10}{'ECE':>10}")
    for name in ("raw", "calibrated"):
        m = metrics[name]
        print(f"  {name:<14}{m['brier']:>10.5f}{m['brier_skill']:>10.3f}{m['ece']:>10.4f}")

    print_reliability("reliability, raw model output", metrics["raw"]["bins"])
    print_reliability("reliability, after calibration", metrics["calibrated"]["bins"])

    payload = {
        "method": "isotonic",
        "points": points,
        "risk_breaks": [["low", 0.05], ["medium", 0.20], ["high", 0.50], ["extreme", 1.01]],
        "fitted_on": meta,
        "metrics": {k: v for k, v in metrics.items() if k != "raw" or True},
        "provisional": not trustworthy,
    }
    if meta.get("model_sha256"):
        payload["model_sha256"] = meta["model_sha256"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out}")

    if not trustworthy:
        print(f"\n  MARKED PROVISIONAL: fitted on {scored_units} unit(s), "
              f"{metrics['n']:,} cells.")
        print(f"  A calibration this thin is not evidence about a fire it has not seen.")
        print(f"  Needs >= {MIN_SAMPLES_FOR_TRUST:,} cells across several events before")
        print(f"  any number derived from it should be quoted to a third party.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
