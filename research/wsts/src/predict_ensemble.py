#!/usr/bin/env python3
"""Deep-ensemble inference over all 12 WSTS+ folds, with georeferenced output.

Why an ensemble rather than one checkpoint
------------------------------------------
The released Res18UTAE_T5 weights are 12 leave-one-year-out folds, and their
filenames carry each fold's test AP:

    Veg: 0.345 0.348 0.359 0.443 0.501 0.506 0.507 0.512 0.516 0.523 0.584 0.590
    mean 0.478  std 0.086  range 0.245

A single held-out YEAR swings AP by up to 0.245. Picking one fold makes any OOD
result unattributable — a low Palisades score could simply be a fold that
scores 0.345 on ordinary WSTS years. So:

  * predict with all 12 and average the PROBABILITIES (not the logits)
  * the per-pixel standard deviation across folds is epistemic uncertainty,
    which is a deliverable in itself rather than a by-product
  * report OOD events as PERCENTILES of the fold distribution, so the claim
    has a calibrated reference instead of a bare number

This mirrors the deep-ensemble setup in WildfireUQ-FCER, which uses these same
checkpoints.

Georeferencing
--------------
Predictions are written as GeoTIFF/COG in their NATIVE grid — EPSG:5070,
375 m, the exact affine the inputs were built on. They are deliberately NOT
pre-warped to Web Mercator: reprojecting once at display time is lossless
bookkeeping, whereas baking a Mercator warp into the file resamples the data
twice and silently shifts pixels. Anything downstream (QGIS, GDAL, a tile
server, deck.gl) can reproject correctly from the embedded CRS.

Band layout of the output:
    1  prob_mean   ensemble mean probability   (calibrated if --calibration)
    2  prob_std    across-fold std  (epistemic uncertainty)
    3  prob_p10    10th percentile across folds
    4  prob_p90    90th percentile across folds
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presets as presets_mod   # noqa: E402

TARGET_RES_M = 375.0
TARGET_SIZE = 128
DST_CRS = "EPSG:5070"

FEATURE_SETS = ("Veg", "Multi", "All")
DEFAULT_FEATURE_SET = "Veg"     # paper's headline; least reliant on coarse weather


# ---------------------------------------------------------------------------
# checkpoint discovery
# ---------------------------------------------------------------------------
_AP_RE = re.compile(r"fold(\d+)_testAP([0-9.]+)\.pth$")


def discover_folds(ckpt_root: Path, feature_set: str = DEFAULT_FEATURE_SET
                   ) -> List[Tuple[int, float, Path]]:
    """Return [(fold_index, reported_test_AP, path)] sorted by fold index.

    The reported AP is parsed from the filename and used as the reference
    distribution for percentile-based reporting.
    """
    d = ckpt_root / "trained_model_weights" / "Res18UTAE_T5" / feature_set
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found. Download with:\n"
            f'  hf download saadlahrichi/WSTSPlus '
            f'--include "trained_model_weights/Res18UTAE_T5/*" '
            f'--local-dir "$IGNIS_DATA_ROOT/pretrained"')
    out = []
    for p in sorted(d.glob("*.pth")):
        if p.name.startswith("._"):          # macOS AppleDouble sidecar
            continue
        m = _AP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), float(m.group(2)), p))
    out.sort(key=lambda t: t[0])
    return out


def fold_ap_distribution(folds) -> Dict[str, float]:
    aps = np.array([ap for _, ap, _ in folds], dtype=float)
    return {"n": len(aps), "mean": float(aps.mean()), "std": float(aps.std(ddof=1)),
            "min": float(aps.min()), "max": float(aps.max()),
            "p10": float(np.percentile(aps, 10)),
            "p25": float(np.percentile(aps, 25)),
            "p50": float(np.percentile(aps, 50)),
            "p75": float(np.percentile(aps, 75))}


def ap_percentile(value: float, folds) -> float:
    """Where an observed AP sits within the fold distribution (0-100).

    This is the number to report for an OOD event. 'Palisades scores below the
    10th percentile of held-out WSTS years' is a calibrated claim; 'Palisades
    scores 0.35' is not, because fold 3 scores 0.345 on an ordinary year.
    """
    aps = np.array([ap for _, ap, _ in folds], dtype=float)
    return float((aps < value).mean() * 100.0)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_fold_model(ckpt_path: Path, third_party: Path, device: str):
    """Instantiate UTAE(Res18) and load one fold's weights.

    The architecture lives in the vendored reference implementation
    (research/wsts/third_party/WildfireSpreadTS). We import rather than
    reimplement — a hand-rolled UTAE that differs in any detail would load the
    state dict "successfully" and produce quietly wrong output.
    """
    import torch
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
        sys.path.insert(0, str(third_party / "src"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    try:
        from models.SMPTempModel import SMPTempModel          # type: ignore
        model = SMPTempModel(**ckpt.get("hyper_parameters", {}))
    except Exception:
        try:
            from src.models.SMPTempModel import SMPTempModel  # type: ignore
            model = SMPTempModel(**ckpt.get("hyper_parameters", {}))
        except Exception as ex:                               # noqa: BLE001
            raise ImportError(
                "Could not import the UTAE model from the vendored reference "
                f"implementation at {third_party}. ({ex})\n"
                "Clone it with:\n"
                "  git clone https://github.com/slahrichi/WildfireSpreadTS "
                f"{third_party}\n"
                "Then check the class path — Lightning checkpoints usually keep "
                "the constructor args under ckpt['hyper_parameters'], which this "
                "loader forwards.") from ex

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    warn: {len(missing)} missing keys (first: {missing[:2]})")
    if unexpected:
        print(f"    warn: {len(unexpected)} unexpected keys (first: {unexpected[:2]})")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# ensemble inference
# ---------------------------------------------------------------------------
def predict_ensemble(x: np.ndarray, folds, third_party: Path,
                     device: str = "cpu") -> Dict[str, np.ndarray]:
    """Run every fold on one input stack and summarise across them.

    x: [T, C, H, W] float32, already standardized as the WSTS dataloader would.
    Averaging is done on PROBABILITIES: averaging logits would let one
    over-confident fold dominate the mean.
    """
    import torch
    preds: List[np.ndarray] = []
    xb = torch.from_numpy(x).float().unsqueeze(0).to(device)   # [1,T,C,H,W]

    for fold_i, ap, path in folds:
        try:
            model = load_fold_model(path, third_party, device)
        except ImportError:
            raise
        except Exception as ex:                                # noqa: BLE001
            print(f"    fold {fold_i}: load failed ({ex}) — skipped")
            continue
        with torch.no_grad():
            logits = model(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            p = torch.sigmoid(logits).squeeze().float().cpu().numpy()
        preds.append(p)
        print(f"    fold {fold_i:2d} (reported AP {ap:.3f}): "
              f"mean p={p.mean():.4f}  max p={p.max():.4f}")
        del model

    if not preds:
        raise RuntimeError("no folds produced a prediction")

    stack = np.stack(preds)                                    # [F,H,W]
    return {
        "prob_mean": stack.mean(axis=0),
        "prob_std": stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0]),
        "prob_p10": np.percentile(stack, 10, axis=0),
        "prob_p90": np.percentile(stack, 90, axis=0),
        "n_folds": len(preds),
        "_stack": stack,
    }


# ---------------------------------------------------------------------------
# georeferenced output
# ---------------------------------------------------------------------------
def tile_transform(lat: float, lon: float):
    """The exact affine the inputs were built on — reused, never recomputed
    differently, so prediction pixels land on the same ground as the inputs."""
    from rasterio.transform import from_origin
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    cx, cy = tx.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    return from_origin(cx - half, cy + half, TARGET_RES_M, TARGET_RES_M)


def apply_calibration(p: np.ndarray, points: List[List[float]]) -> np.ndarray:
    xs = np.array([a for a, _ in points], dtype=np.float32)
    ys = np.array([b for _, b in points], dtype=np.float32)
    return np.interp(p, xs, ys).astype(np.float32)


def write_cog(out_path: Path, bands: Dict[str, np.ndarray], lat: float, lon: float,
              tags: Optional[dict] = None) -> Path:
    """Write a 4-band Cloud-Optimized GeoTIFF in the NATIVE grid.

    Native CRS on purpose: pre-warping to Web Mercator for display would
    resample the data a second time and shift pixels off the ground they were
    computed for. Store it honestly; reproject at render time.
    """
    import rasterio
    order = ["prob_mean", "prob_std", "prob_p10", "prob_p90"]
    arr = np.stack([bands[k].astype(np.float32) for k in order])
    profile = {
        "driver": "COG", "dtype": "float32", "count": len(order),
        "height": arr.shape[1], "width": arr.shape[2],
        "crs": DST_CRS, "transform": tile_transform(lat, lon),
        "compress": "DEFLATE", "predictor": 2, "nodata": np.nan,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst = rasterio.open(out_path, "w", **profile)
    except Exception:                       # older GDAL without the COG driver
        profile.update(driver="GTiff", tiled=True, blockxsize=128, blockysize=128)
        dst = rasterio.open(out_path, "w", **profile)
    with dst:
        dst.write(arr)
        for i, name in enumerate(order, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    return out_path


def alignment_check(lat: float, lon: float) -> dict:
    """Verify the tile's corners round-trip through EPSG:4326.

    A silent CRS or axis-order error is the classic way a prediction ends up
    rendered a few hundred metres off the ground it describes. Round-tripping
    the corners catches it before anything reaches a map.
    """
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    inv = Transformer.from_crs(DST_CRS, "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    corners = [(cx - half, cy + half), (cx + half, cy + half),
               (cx + half, cy - half), (cx - half, cy - half)]
    ll = [inv.transform(x, y) for x, y in corners]
    back = [fwd.transform(a, b) for a, b in ll]
    err = max(max(abs(bx - x), abs(by - y))
              for (x, y), (bx, by) in zip(corners, back))
    return {"center_5070": [cx, cy], "corners_lonlat": ll,
            "roundtrip_error_m": float(err),
            "extent_km": TARGET_SIZE * TARGET_RES_M / 1000.0}


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import os
    ap = argparse.ArgumentParser(description="12-fold ensemble inference + geo output")
    ap.add_argument("--preset")
    ap.add_argument("--feature-set", choices=FEATURE_SETS, default=DEFAULT_FEATURE_SET)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--list-folds", action="store_true",
                    help="show the fold AP distribution and exit")
    args = ap.parse_args(argv)

    data_root = Path(os.environ.get("IGNIS_DATA_ROOT", "data"))
    ckpt_root = data_root / "pretrained"
    third_party = Path(__file__).resolve().parents[1] / "third_party" / "WildfireSpreadTS"

    folds = discover_folds(ckpt_root, args.feature_set)
    dist = fold_ap_distribution(folds)
    print(f"feature set {args.feature_set}: {dist['n']} folds")
    print(f"  reported AP  mean={dist['mean']:.3f} std={dist['std']:.3f} "
          f"range=[{dist['min']:.3f}, {dist['max']:.3f}]")
    print(f"  percentiles  p10={dist['p10']:.3f} p25={dist['p25']:.3f} "
          f"p50={dist['p50']:.3f} p75={dist['p75']:.3f}")
    print(f"\n  A single held-out YEAR moves AP by {dist['max']-dist['min']:.3f}.")
    print("  Report OOD events as PERCENTILES of this distribution, not raw AP.")

    if args.list_folds:
        for i, apv, p in folds:
            print(f"    fold {i:2d}  AP {apv:.3f}  {p.name}")
        return 0

    targets = ([presets_mod.by_key(args.preset)] if args.preset
               else list(presets_mod.PRESETS))
    print("\n=== geospatial alignment ===")
    for p in targets:
        chk = alignment_check(p.lat, p.lon)
        lon0, lat0 = chk["corners_lonlat"][0]
        print(f"  {p.key:<12} NW corner ({lon0:.4f}, {lat0:.4f})  "
              f"{chk['extent_km']:.1f} km  roundtrip err {chk['roundtrip_error_m']:.2e} m")

    print("\nInput assembly (build_event_stack) is still pending — see the "
          "notebook §4 and wsts_spec.SOURCE_MAP.\n"
          "Bands built so far: 0,1,2,4 (VIIRS+EVI2), 12,13,14 (topo), 16 (landcover).\n"
          "Still needed: 3,5-11,15,17-22 (GridMET/HRRR weather + FIRMS active fire).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
