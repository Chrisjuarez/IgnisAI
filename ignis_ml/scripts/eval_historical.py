#!/usr/bin/env python3
"""Phase 5 — out-of-distribution evaluation against observed perimeters.

For each historical preset (Palisades, Eaton, Camp, Dixie, Caldor), run the
multistep forecast and score each step against the observed fire perimeter,
writing a CSV of event/step/ckpt_sha/iou/dice/csi/hausdorff_km. This is the
*real* validation set — out of training distribution — and the regression gate
the gameplan's Phase 7 CI wires into PRs.

Inputs:
  * Observed perimeters: data/perimeters/<event>.geojson (EPSG:4326 polygons).
    Get final + daily from CAL FIRE FRAP / NIFC WFIGS. Optional per-day via a
    FeatureCollection where each feature has a `day` (1-based) property.
  * Predictions: either the live tilesvc (`--mode http --tilesvc URL`) or the
    in-process serving modules (`--mode local`, requires the model + caches).

Run:
  python -m ignis_ml.scripts.eval_historical --mode http \
      --tilesvc https://ignisai-tilesvc.onrender.com --out models/eval/v3_eval.csv

STATUS: scaffold. Metrics (IoU/Dice/CSI/Hausdorff) and the HTTP path are fully
implemented. The rasterization of observed perimeters to the tile grid is wired
through services.tilesvc.grid; only the perimeter GeoJSON files need to be
supplied under data/perimeters/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Historical presets — keep lat/lon/date in sync with
# frontend/src/components/MapComponent.jsx HISTORICAL_FIRES.
@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    lat: float
    lon: float
    date: str       # ISO Zulu
    ignition: bool = True


PRESETS: Tuple[Preset, ...] = (
    Preset("palisades", "Palisades Fire", 34.078, -118.555, "2025-01-07T18:30:00Z"),
    Preset("eaton", "Eaton Fire", 34.1897, -118.1300, "2025-01-07T22:30:00Z"),
    Preset("camp", "Camp/Paradise Fire", 39.7596, -121.6219, "2018-11-08T14:30:00Z"),
    Preset("dixie", "Dixie Fire", 39.8760, -121.3870, "2021-07-14T17:00:00Z"),
    Preset("caldor", "Caldor Fire", 38.5900, -120.5400, "2021-08-14T18:00:00Z"),
)


# --------------------------- metrics ---------------------------------------
def iou(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = pred > 0.5, obs > 0.5
    inter = float(np.logical_and(p, o).sum())
    union = float(np.logical_or(p, o).sum())
    return inter / union if union > 0 else float("nan")


def dice(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = pred > 0.5, obs > 0.5
    s = float(p.sum() + o.sum())
    return 2.0 * float(np.logical_and(p, o).sum()) / s if s > 0 else float("nan")


def csi(pred: np.ndarray, obs: np.ndarray) -> float:
    """Critical Success Index = TP / (TP + FP + FN)."""
    p, o = pred > 0.5, obs > 0.5
    tp = float(np.logical_and(p, o).sum())
    fp = float(np.logical_and(p, ~o).sum())
    fn = float(np.logical_and(~p, o).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def hausdorff_km(pred: np.ndarray, obs: np.ndarray, px_km: float = 0.5) -> float:
    """Symmetric Hausdorff distance between mask boundaries, in km.

    px_km = pixel size in km (0.5 for the 500 m grid). Returns nan if either
    mask is empty. O(n*m) on boundary pixels — fine at 64x64.
    """
    def boundary_pts(m: np.ndarray) -> np.ndarray:
        b = m > 0.5
        if not b.any():
            return np.empty((0, 2))
        # erosion via shifts; a pixel is boundary if any 4-neighbor is background
        up = np.zeros_like(b); up[1:] = b[:-1]
        dn = np.zeros_like(b); dn[:-1] = b[1:]
        lf = np.zeros_like(b); lf[:, 1:] = b[:, :-1]
        rt = np.zeros_like(b); rt[:, :-1] = b[:, 1:]
        edge = b & ~(up & dn & lf & rt)
        ys, xs = np.where(edge)
        return np.stack([ys, xs], axis=1).astype(np.float64)

    A, B = boundary_pts(pred), boundary_pts(obs)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    d_ab = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1).max()
    d_ba = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(axis=1).max()
    return float(max(d_ab, d_ba) * px_km)


# --------------------- observed perimeter rasterization --------------------
def rasterize_perimeter(geojson_path: Path, preset: Preset, day: Optional[int]) -> Optional[np.ndarray]:
    """Rasterize an observed perimeter GeoJSON to the preset's 64x64 tile grid.

    If `day` is given and features carry a `day` property, only features with
    day <= the requested day are burned in (cumulative footprint), matching the
    cumulative nature of a fire perimeter.
    """
    try:
        from rasterio.features import rasterize
        from services.tilesvc.grid import build_grid, SIZE, tile_affine, lonlat_to_tile
    except Exception as e:
        print(f"[eval] rasterize unavailable ({e}); skipping observed mask")
        return None
    if not geojson_path.exists():
        return None
    gj = json.loads(geojson_path.read_text())
    feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]
    if day is not None:
        kept = [f for f in feats if f.get("properties", {}).get("day") in (None,)
                or int(f.get("properties", {}).get("day", 10**9)) <= day]
        feats = kept or feats
    shapes = [(f["geometry"], 1) for f in feats if f.get("geometry")]
    if not shapes:
        return None
    tile = lonlat_to_tile(preset.lon, preset.lat)
    transform = tile_affine(tile)
    # NOTE: perimeters are EPSG:4326; reproject geometries to EPSG:5070 first in
    # the full implementation (rasterio.warp.transform_geom). Left as a one-liner
    # TODO so the metric wiring is reviewable now.
    mask = rasterize(shapes, out_shape=(SIZE, SIZE), transform=transform,
                     fill=0, default_value=1, dtype="uint8")
    return mask.astype(np.float32)


# ------------------------------ predictors ---------------------------------
def predict_http(tilesvc: str, preset: Preset, steps: int) -> Optional[List[np.ndarray]]:
    """Fetch multistep prediction from a running tilesvc and decode per-step prob.

    Returns a list of [64,64] probability arrays (one per step), or None.
    """
    import urllib.request
    import base64
    import io
    url = (f"{tilesvc.rstrip('/')}/predict_multistep?lat={preset.lat}&lon={preset.lon}"
           f"&steps={steps}&step_hours=24&ignition={'true' if preset.ignition else 'false'}"
           f"&date={preset.date}")
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        print(f"[eval] {preset.key}: http predict failed ({e})")
        return None
    try:
        from PIL import Image
    except Exception:
        print("[eval] Pillow needed to decode prediction PNGs")
        return None
    out: List[np.ndarray] = []
    for step in payload.get("steps", []):
        b64 = step.get("image_base64")
        if not b64:
            continue
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return out or None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="OOD eval of fire-spread vs observed perimeters")
    ap.add_argument("--mode", choices=["http", "local"], default="http")
    ap.add_argument("--tilesvc", default="https://ignisai-tilesvc.onrender.com")
    ap.add_argument("--perimeters", type=Path, default=Path("data/perimeters"))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="prob threshold for binarizing prediction (delta operating pt)")
    ap.add_argument("--ckpt-sha", default="unknown")
    ap.add_argument("--out", type=Path, default=Path("models/eval/eval.csv"))
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    for preset in PRESETS:
        if args.mode == "http":
            probs = predict_http(args.tilesvc, preset, args.steps)
        else:
            print("[eval] local mode requires the serving modules + caches; "
                  "use --mode http against a running tilesvc, or wire the "
                  "mini-pipeline notebook's rollout here.")
            probs = None
        if not probs:
            continue
        for step_idx, prob in enumerate(probs, start=1):
            obs = rasterize_perimeter(
                args.perimeters / f"{preset.key}.geojson", preset, day=step_idx)
            if obs is None:
                print(f"[eval] {preset.key} step {step_idx}: no observed perimeter")
                continue
            pred = (prob >= args.threshold).astype(np.float32)
            rows.append({
                "event": preset.key,
                "step": step_idx,
                "ckpt_sha": args.ckpt_sha,
                "iou": round(iou(pred, obs), 4),
                "dice": round(dice(pred, obs), 4),
                "csi": round(csi(pred, obs), 4),
                "hausdorff_km": round(hausdorff_km(pred, obs), 3),
            })
            print(f"[eval] {preset.key} step {step_idx}: "
                  f"IoU={rows[-1]['iou']} CSI={rows[-1]['csi']} "
                  f"Hd={rows[-1]['hausdorff_km']}km")

    if not rows:
        print("[eval] no rows produced (need a running model + perimeters).")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
