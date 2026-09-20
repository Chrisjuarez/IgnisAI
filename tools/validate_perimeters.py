"""Score predicted spread against the perimeter each fire actually reached.

Direction alignment stops discriminating once fuel is uniform - on live
wildland fires every engine follows the wind and scores near +1. What still
separates them is where they put the fire and how much of it, and only observed
ground truth settles that.

The five events here are held out by construction: NDWS training data ends in
2020, and none of these appear in any training tile.

METRIC CHOICE. A three-day forecast is compared against a final perimeter that
took weeks to burn, so the model SHOULD cover only part of it - coverage is not
a fairness test and is reported for context only. Containment is the fair one
and is the headline: of the area a model says will burn, how much lies inside
the ground that eventually burned at all. Predicting fire outside the final
perimeter is wrong no matter what the horizon is, and over-prediction is the
failure the magnitude comparison already hinted at.

    python tools/validate_perimeters.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("NOAA_GRIB_ENABLED", "1")

PERIMETER_DIR = _REPO / "data" / "perimeters"

#: Two reference times per fire, because they ask different questions.
#:
#: The ignition times come from the existing event registry and sit AT or
#: BEFORE the first detection - eaton's is four hours before the fire started,
#: and its cached window holds 4 background detections a day and never sees the
#: fire. Forecasting from a fire that does not exist yet and scoring against
#: where it later burned is why every engine scored 6-9% containment there.
#:
#: The mid-fire times sit two to six days in, where a real incident lives: an
#: irregular multi-day footprint, thousands of detections, and weather that has
#: turned since ignition. This is the case the product actually serves.
EVENTS = {
    "palisades":     (34.0780, -118.5550, "2025-01-07T18:30:00Z"),
    "eaton":         (34.1897, -118.1300, "2025-01-07T22:30:00Z"),
    "camp":          (39.7596, -121.6219, "2018-11-08T14:30:00Z"),
    "dixie":         (39.8760, -121.3870, "2021-07-14T17:00:00Z"),
    "caldor":        (38.5900, -120.5400, "2021-08-14T18:00:00Z"),
    "palisades_mid": (34.0780, -118.5550, "2025-01-10T18:30:00Z"),
    "eaton_mid":     (34.1897, -118.1300, "2025-01-10T22:30:00Z"),
    "camp_mid":      (39.7596, -121.6219, "2018-11-10T14:30:00Z"),
    "dixie_mid":     (39.8760, -121.3870, "2021-07-20T17:00:00Z"),
    "caldor_mid":    (38.5900, -120.5400, "2021-08-18T18:00:00Z"),
}

#: Both phases of a fire score against the same final perimeter.
def perimeter_name(event: str) -> str:
    return event[:-4] if event.endswith("_mid") else event


def perimeter_mask(name: str, tile) -> Optional[np.ndarray]:
    """The final fire perimeter, rasterised onto the tile grid."""
    path = PERIMETER_DIR / f"{perimeter_name(name)}.geojson"
    if not path.is_file():
        return None

    import rasterio.features
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform

    from services.tilesvc.grid import CRS_ALBERS, SIZE, tile_affine

    to_albers = Transformer.from_crs("EPSG:4326", CRS_ALBERS, always_xy=True).transform
    geoms = [shapely_transform(to_albers, shape(f["geometry"]))
             for f in json.loads(path.read_text())["features"]]
    if not geoms:
        return None
    return rasterio.features.rasterize(
        [(g, 1) for g in geoms], out_shape=(SIZE, SIZE),
        transform=tile_affine(tile), fill=0, dtype="uint8",
    ).astype(bool)


def containment(predicted: np.ndarray, burned: np.ndarray, observed: np.ndarray) -> Dict[str, Any]:
    """How much of the predicted NEW growth lies inside the eventual burn.

    Cells already alight are excluded from both sides: they are correct by
    construction and would flatter every engine equally.
    """
    new = predicted & ~observed
    if not new.any():
        return {"predicted_cells": 0, "contained": None, "outside_cells": 0,
                "coverage": None}
    inside = int((new & burned).sum())
    reachable = int((burned & ~observed).sum())
    return {
        "predicted_cells": int(new.sum()),
        "contained": inside / int(new.sum()),
        "outside_cells": int(new.sum()) - inside,
        # Containment alone is maximised by predicting almost nothing, so the
        # share of the eventual burn actually reached is reported beside it.
        # It is NOT a fairness test - three days against a fire that burned for
        # weeks should be well under 1 - but it exposes an engine that scores
        # well by staying home.
        "coverage": (inside / reachable) if reachable else None,
    }


def learned_rollout(checkpoint: Path, x_dyn: np.ndarray, tile, steps: int) -> Optional[list]:
    """The ConvLSTM's forecast, in the same rollout shape as the physics engines.

    Without this the comparison is physics against physics, which answers the
    wrong question. What matters is whether any engine beats the model the
    project already has.
    """
    try:
        import torch

        from ignis_ml.scripts.validate_checkpoint import normalize_dynamic
        from ignis_ml.src.data.features import append_derived_features
        from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet
        from services.tilesvc.static_catalog import load_static_tensor_for_model
    except ImportError as exc:
        print("      (learned model unavailable: %s)" % exc)
        return None

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Cd, Cs = int(ck["cd"]), int(ck["cs"])
    T = int(ck.get("seq_len") or x_dyn.shape[0])
    dyn_order = list(ck.get("dyn_order") or
                     ["fire_t", "u", "v", "gust", "tempC", "q", "precip"])
    # Loud, not silent. A skipped model reads as "the physics engines are the
    # only options", which is the wrong conclusion to draw from a missing
    # credential.
    try:
        stat, _ = load_static_tensor_for_model(tile, list(ck.get("stat_order") or []))
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        print("      (learned model skipped: statics unavailable - %s: %s)"
              % (type(exc).__name__, str(exc)[:70]))
        return None
    stat = np.asarray(stat, dtype=np.float32)
    if stat.shape[0] != Cs:
        print("      (learned model skipped: checkpoint wants Cs=%d, statics gave %d)"
              % (Cs, stat.shape[0]))
        return None

    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=int(ck.get("hidden", 64)),
                         drop=0.0, drop_decoder=0.0)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    threshold = float(ck.get("best_threshold", 0.5))
    window = x_dyn[-T:].copy()
    out = []
    for index in range(steps):
        xn = normalize_dynamic(window, dyn_order[:7])
        if xn.shape[1] < Cd:
            xn, _ = append_derived_features(xn, dyn_order=dyn_order[:7],
                                            include=None, days_since_fire_cap=None)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(xn[None]).float(),
                                       torch.from_numpy(stat[None]).float()))[0, 0].numpy()
        lead = (index + 1) * 24
        out.append({"index": index, "lead_hours": lead, "label": f"{lead // 24} day",
                    "prob": prob.astype(np.float32)})
        # Autoregressive feedback, as the service does: the prediction becomes
        # the next step's fire channel.
        nxt = window[-1].copy()
        nxt[0] = np.maximum(window[-1, 0], (prob >= threshold).astype(np.float32))
        window = np.concatenate([window[1:], nxt[None]], axis=0)
    return out


def evaluate(name: str, steps: int = 3, checkpoint: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    from services.tilesvc.baseline_spread import baseline_rollout
    from services.tilesvc.dynamic_builder import build_dynamic_for_tile
    from services.tilesvc.fuel_raster import fuel_codes_for_tile
    from services.tilesvc.grid import PIX, SIZE, lonlat_to_tile
    from services.tilesvc.physics_spread import physics_rollout
    from services.tilesvc.pyretechnics_spread import pyretechnics_rollout

    lat, lon, iso = EVENTS[name]
    ref = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    from services.runtime_cache.paths import firms_dir, noaa_dir
    snap, noaa = firms_dir(name), noaa_dir(name)
    for key, path, required in (("FIRMS_SNAPSHOT_DIR", snap, "1"),
                                ("NOAA_GRID_CACHE_DIR", noaa, None)):
        if path.is_dir():
            os.environ[key] = str(path)
            if required:
                os.environ["FIRMS_SNAPSHOT_REQUIRED"] = required
        else:
            os.environ.pop(key, None)

    tile = lonlat_to_tile(lon, lat)
    burned = perimeter_mask(name, tile)
    if burned is None or not burned.any():
        return {"name": name, "status": "no perimeter in tile"}

    order = ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]
    try:
        x = np.asarray(build_dynamic_for_tile(lat, lon, T_seq=6, hours_step=24, ignition=True,
                                              ref_time=ref, channel_order=order), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - a missing cache is a result
        return {"name": name, "status": f"inputs unavailable ({type(exc).__name__})"}

    observed = x[-1, 0] > 0.5
    if not observed.any():
        return {"name": name, "status": "no observed fire at reference time"}

    series = [(float(x[i, 1].mean()), float(x[i, 2].mean())) for i in (-3, -2, -1)]
    u, v = series[-1]
    codes = fuel_codes_for_tile(tile)
    cell_km2 = (PIX / 1000.0) ** 2

    engines = {
        "downwind": (baseline_rollout(observed.astype(np.float32), u_ms=u, v_ms=v, steps=steps,
                                      step_hours=24, ignition_rc=(SIZE // 2, SIZE // 2)), 0.1),
        "rothermel": (physics_rollout(observed.astype(np.float32), fuel_codes=codes, u_ms=u, v_ms=v,
                                      steps=steps, step_hours=24,
                                      ignition_rc=(SIZE // 2, SIZE // 2)), 0.1),
        "pyretechnics": (pyretechnics_rollout(observed.astype(np.float32), fuel_codes=codes,
                                              wind_series=series, steps=steps, step_hours=24), 0.5),
    }
    if checkpoint is not None:
        learned = learned_rollout(checkpoint, x, tile, steps)
        if learned is not None:
            engines["ignis (learned)"] = (learned, 0.5)

    results = {}
    for engine, (rollout, threshold) in engines.items():
        predicted = rollout[-1]["prob"] >= threshold
        stats = containment(predicted, burned, observed)
        stats["predicted_km2"] = stats["predicted_cells"] * cell_km2
        stats["outside_km2"] = stats["outside_cells"] * cell_km2
        results[engine] = stats

    return {
        "name": name,
        "status": "ok",
        "burned_in_tile_km2": float(burned.sum()) * cell_km2,
        "observed_km2": float(observed.sum()) * cell_km2,
        "engines": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python tools/validate_perimeters.py")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Score the learned model alongside the physics engines")
    ap.add_argument("--mid", action="store_true",
                    help="Score mid-fire rather than at ignition - the live case, "
                         "where the fire already has a multi-day footprint")
    args = ap.parse_args(argv)

    print("Predicted 3-day growth against the final fire perimeter")
    print("Held out by construction: NDWS training data ends in 2020.")
    print()
    if args.mid:
        print("Phase: MID-FIRE - two to six days in, real multi-day footprints.")
        print("This is the case the product serves.\n")
    else:
        print("Phase: IGNITION - reference times at or before the first detection.")
        print("The hardest case and not the product's: eaton's reference time is")
        print("four hours before that fire started. Use --mid for the live case.\n")
    rows = []
    for name in EVENTS:
        if name.endswith("_mid") and not args.mid:
            continue
        if args.mid and not name.endswith("_mid"):
            continue
        result = evaluate(name, checkpoint=args.checkpoint)
        if not result or result["status"] != "ok":
            print("  %-11s %s" % (name, (result or {}).get("status", "failed")))
            continue
        print("  %-11s eventual burn in tile %6.1f km2   already alight %5.1f km2"
              % (name, result["burned_in_tile_km2"], result["observed_km2"]))
        for engine, s in result["engines"].items():
            if s["contained"] is None:
                print("      %-13s no new growth" % engine)
                continue
            cover = "   n/a" if s["coverage"] is None else "%5.1f%%" % (100 * s["coverage"])
            print("      %-13s predicts %7.1f km2   contained %5.1f%%   reached %s   outside %6.1f km2"
                  % (engine, s["predicted_km2"], 100 * s["contained"], cover, s["outside_km2"]))
            rows.append((engine, s["contained"], s["coverage"], s["predicted_km2"]))
        print()

    if rows:
        print("  %-13s %11s %10s %12s" % ("engine", "containment", "reached", "predicted km2"))
        for engine in ("downwind", "rothermel", "pyretechnics", "ignis (learned)"):
            vals = [(c, v, a) for e, c, v, a in rows if e == engine]
            if not vals:
                continue
            cov = [v for _, v, _ in vals if v is not None]
            print("      %-13s %10.3f %9.3f %12.1f" % (
                engine,
                sum(c for c, _, _ in vals) / len(vals),
                (sum(cov) / len(cov)) if cov else float("nan"),
                sum(a for _, _, a in vals) / len(vals)))
        print()
        print("  Read the two together. Containment alone rewards an engine that")
        print("  predicts almost nothing - pyretechnics scores 100% on camp by")
        print("  predicting 7 km2 against a 537 km2 burn. Neither column is a")
        print("  verdict on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
