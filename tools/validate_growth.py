"""Score a 3-day forecast against what actually burned in those 3 days.

validate_perimeters compares predicted growth against each fire's FINAL
perimeter, and that is unfair in a way that grows as a fire winds down. Caldor
had 292 of its eventual 348 km2 already alight at the reference time, so only
56 km2 remained to burn EVER; any engine forecasting three vigorous days must
overshoot. Measured that way the over-prediction ratio tracked remaining fuel
almost exactly - 7.5x on eaton with 26 km2 left, 1.5x on camp with 391 km2 -
which says the metric was measuring how far through its life each fire was.

This verifies the actual claim instead: forecast from D-3, compare against the
observed burn at D. Both come from the same cached FIRMS window, so no extra
data is needed and prediction and truth share one source and one grid.

Reported as precision, recall and IoU over NEW growth. Cells already alight at
the reference time are excluded from all three - they are correct by
construction and would flatter every engine equally.

    python tools/validate_growth.py [--checkpoint model.pt]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("NOAA_GRIB_ENABLED", "1")

#: Profile -> (lat, lon, end of the cached window). The forecast starts
#: HORIZON_DAYS before that end and is verified against it.
EVENTS = {
    "palisades_full": (34.0780, -118.5550, "2025-01-10T18:30:00Z"),
    "eaton_mid":      (34.1897, -118.1300, "2025-01-10T22:30:00Z"),
    "camp_mid":       (39.7596, -121.6219, "2018-11-10T14:30:00Z"),
    "dixie_mid":      (39.8760, -121.3870, "2021-07-20T17:00:00Z"),
    "caldor_mid":     (38.5900, -120.5400, "2021-08-18T18:00:00Z"),
}

HORIZON_DAYS = 3

#: Three input frames, matching the deployed control60 checkpoint. Six would
#: need nine days of cache behind a forecast that already starts three days
#: inside a seven-day window, which only the palisades_full profile has.
T_SEQ = 3
ORDER = ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]


def use_profile(profile: str) -> None:
    base = _REPO / ".cache" / "runtime_cache" / profile
    os.environ["FIRMS_SNAPSHOT_DIR"] = str(base / "firms_snapshots")
    os.environ["FIRMS_SNAPSHOT_REQUIRED"] = "1"
    os.environ["NOAA_GRID_CACHE_DIR"] = str(base / "noaa_grid_cache")


def growth_scores(predicted: np.ndarray, truth: np.ndarray, prior: np.ndarray) -> Dict[str, Any]:
    """Precision, recall and IoU over cells that were not already alight."""
    p = predicted & ~prior
    t = truth & ~prior
    hit = int((p & t).sum())
    union = int((p | t).sum())
    return {
        "predicted": int(p.sum()),
        "actual": int(t.sum()),
        "precision": (hit / int(p.sum())) if p.any() else None,
        "recall": (hit / int(t.sum())) if t.any() else None,
        "iou": (hit / union) if union else None,
    }


def evaluate(profile: str, checkpoint: Optional[Path]) -> Optional[Dict[str, Any]]:
    from services.tilesvc.baseline_spread import baseline_rollout
    from services.tilesvc.dynamic_builder import build_dynamic_for_tile
    from services.tilesvc.fuel_raster import fuel_codes_for_tile
    from services.tilesvc.grid import PIX, SIZE, lonlat_to_tile
    from services.tilesvc.physics_spread import physics_rollout
    from services.tilesvc.pyretechnics_spread import pyretechnics_rollout
    from tools.validate_perimeters import learned_rollout

    lat, lon, iso = EVENTS[profile]
    use_profile(profile)
    verify_at = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    forecast_at = verify_at - dt.timedelta(days=HORIZON_DAYS)
    tile = lonlat_to_tile(lon, lat)

    def frames(ref):
        return np.asarray(build_dynamic_for_tile(lat, lon, T_seq=T_SEQ, hours_step=24,
                                                 ignition=True, ref_time=ref,
                                                 channel_order=ORDER), dtype=np.float32)
    try:
        x = frames(forecast_at)
        truth = frames(verify_at)[-1, 0] > 0.5
    except Exception as exc:  # noqa: BLE001 - a missing day is a result
        return {"profile": profile, "status": f"inputs unavailable ({type(exc).__name__})"}

    prior = x[-1, 0] > 0.5
    if not prior.any():
        return {"profile": profile, "status": "nothing burning at the forecast time"}
    if not (truth & ~prior).any():
        return {"profile": profile, "status": "no new growth observed in the window"}

    series = [(float(x[i, 1].mean()), float(x[i, 2].mean())) for i in (-3, -2, -1)]
    u, v = series[-1]
    codes = fuel_codes_for_tile(tile)
    cell = (PIX / 1000.0) ** 2

    engines = {
        "downwind": (baseline_rollout(prior.astype(np.float32), u_ms=u, v_ms=v,
                                      steps=HORIZON_DAYS, step_hours=24,
                                      ignition_rc=(SIZE // 2, SIZE // 2)), 0.1),
        "rothermel": (physics_rollout(prior.astype(np.float32), fuel_codes=codes, u_ms=u, v_ms=v,
                                      steps=HORIZON_DAYS, step_hours=24,
                                      ignition_rc=(SIZE // 2, SIZE // 2)), 0.1),
        "pyretechnics": (pyretechnics_rollout(prior.astype(np.float32), fuel_codes=codes,
                                              wind_series=series, steps=HORIZON_DAYS,
                                              step_hours=24), 0.5),
    }
    if checkpoint is not None:
        learned = learned_rollout(checkpoint, x, tile, HORIZON_DAYS)
        if learned is not None:
            engines["ignis (learned)"] = (learned, 0.5)

    scores = {}
    for name, (rollout, threshold) in engines.items():
        s = growth_scores(rollout[-1]["prob"] >= threshold, truth, prior)
        s["predicted_km2"] = s["predicted"] * cell
        s["actual_km2"] = s["actual"] * cell
        scores[name] = s

    return {
        "profile": profile,
        "status": "ok",
        "forecast_at": forecast_at.strftime("%Y-%m-%d"),
        "verify_at": verify_at.strftime("%Y-%m-%d"),
        "already_alight_km2": float(prior.sum()) * cell,
        "engines": scores,
    }


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/validate_growth.py")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args(argv)

    print(f"{HORIZON_DAYS}-day forecast against the burn observed {HORIZON_DAYS} days later")
    print("Prediction and truth come from the same cached FIRMS window.\n")

    totals: Dict[str, List[Dict[str, Any]]] = {}
    for profile in EVENTS:
        result = evaluate(profile, args.checkpoint)
        if not result or result["status"] != "ok":
            print("  %-15s %s" % (profile, (result or {}).get("status", "failed")))
            continue
        head = "%s -> %s" % (result["forecast_at"], result["verify_at"])
        print("  %-15s %s   already alight %6.1f km2" % (profile, head, result["already_alight_km2"]))
        for name, s in result["engines"].items():
            totals.setdefault(name, []).append(s)
            if s["precision"] is None or s["recall"] is None:
                print("      %-15s no growth predicted" % name)
                continue
            print("      %-15s predicts %6.1f km2 vs %6.1f actual   "
                  "precision %5.1f%%  recall %5.1f%%  IoU %5.1f%%"
                  % (name, s["predicted_km2"], s["actual_km2"],
                     100 * s["precision"], 100 * s["recall"], 100 * s["iou"]))
        print()

    if totals:
        print("  %-15s %10s %9s %8s %12s" % ("engine", "precision", "recall", "IoU", "over-predict"))
        for name, rows in totals.items():
            ok = [r for r in rows if r["iou"] is not None]
            if not ok:
                continue
            ratio = sum(r["predicted_km2"] for r in ok) / max(sum(r["actual_km2"] for r in ok), 1e-9)
            print("      %-15s %9.3f %9.3f %8.3f %11.1fx" % (
                name,
                sum(r["precision"] for r in ok) / len(ok),
                sum(r["recall"] for r in ok) / len(ok),
                sum(r["iou"] for r in ok) / len(ok),
                ratio))

        # The aggregate ratio hides the finding that matters. Per fire the
        # ratios run 0.3x to 21x and predicted growth is NEGATIVELY correlated
        # with actual growth - -0.88 for downwind, -0.81 for rothermel. The
        # engines predict least where the fire ran hardest.
        print()
        print("  %-15s %-34s %8s" % ("engine", "predicted/actual per fire", "corr"))
        for name, rows in totals.items():
            ok = [r for r in rows if r["iou"] is not None and r["actual_km2"] > 0]
            if len(ok) < 3:
                continue
            actual = [r["actual_km2"] for r in ok]
            pred = [r["predicted_km2"] for r in ok]
            print("      %-15s %-34s %8.2f" % (
                name, " ".join("%5.1fx" % (p / a) for p, a in zip(pred, actual)),
                _pearson(actual, pred)))
        print()
        print("  A negative correlation is the result to act on. It means no scalar")
        print("  spread-rate calibration can help: the engines are not uniformly")
        print("  fast or slow, they are uncorrelated with how these fires actually")
        print("  ran. Growth over three days is dominated by suppression, weather")
        print("  change and fuel exhaustion - none of which any engine here models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
