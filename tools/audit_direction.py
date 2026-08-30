#!/usr/bin/env python3
"""Does the forecast go the way the wind blows?

Spread is wind-dominated, so a forecast whose mass runs against the wind is
wrong regardless of how good its AP or CSI look - those metrics are
direction-blind, which is how the Palisades error survived review. This asks
the one question they cannot: for each fire, where did the predicted growth go
relative to where the wind was blowing?

It runs against the deployed API rather than a checkpoint on disk, so it audits
the whole live path - weather source, static inputs, model, calibration - and
not just the weights. A model that is fine in isolation and wrong in production
fails here, which is the point.

It also scores a trivial baseline: "the fire spreads straight downwind". If the
model cannot beat that, the model is not yet earning its complexity.

Usage:
    python -m tools.audit_direction --limit 10
    python -m tools.audit_direction --preset palisades
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BACKEND = "https://ignisai-backend.onrender.com"
TILESVC = "https://ignisai-tilesvc.onrender.com"

#: Known events with an established wind regime, for a check with a right answer.
PRESETS = {
    "palisades": {"lat": 34.078, "lon": -118.555, "date": "2025-01-07T18:30:00Z",
                  "expect": "SW", "note": "Santa Ana, offshore toward the ocean"},
    "eaton": {"lat": 34.19, "lon": -118.10, "date": "2025-01-07T18:30:00Z",
              "expect": "SW", "note": "Same Santa Ana event"},
}

#: Below this the wind gives no steer, so alignment is meaningless.
CALM_MS = 0.5


def get_json(url: str, params: Dict[str, Any], timeout: int = 280) -> Dict[str, Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
        return json.loads(response.read().decode())


def polygon_centroid(coords: List[List[float]]) -> Optional[Tuple[float, float]]:
    """Area-weighted centroid of one ring, in lon/lat."""
    if len(coords) < 3:
        return None
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-12:
        return None
    return cx / (3.0 * area), cy / (3.0 * area)


def feature_centroids(features: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    out = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        rings = []
        if geometry.get("type") == "Polygon":
            rings = geometry["coordinates"][:1]
        elif geometry.get("type") == "MultiPolygon":
            rings = [poly[0] for poly in geometry["coordinates"]]
        for ring in rings:
            centroid = polygon_centroid([(p[0], p[1]) for p in ring])
            if centroid:
                out.append(centroid)
    return out


def mean_centroid(features: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    centroids = feature_centroids(features)
    if not centroids:
        return None
    return (sum(c[0] for c in centroids) / len(centroids),
            sum(c[1] for c in centroids) / len(centroids))


def growth_origin(scene: Dict[str, Any], ignition: Tuple[float, float]) -> Tuple[float, float]:
    """Where the forecast growth starts from.

    The ignition point is the wrong origin once a fire has been burning: by
    forecast time the front is on the perimeter, kilometres from where it
    started, and growth is measured from there. Using the ignition instead
    mixes in all the spread that already happened, which shortens the vector
    and makes the bearing noise-dominated - most visibly for a forecast that
    grows outward on every side, whose bands stay centred on the origin no
    matter which way the front is actually running.
    """
    return mean_centroid((scene.get("observed") or {}).get("features") or []) or ignition


def spread_bearing(scene: Dict[str, Any], ignition: Tuple[float, float]) -> Optional[float]:
    """Direction the forecast pushes the fire, measured from the current front.

    Uses the latest day only: earlier bands sit on top of the footprint by
    construction, so including them pulls the vector toward zero and would
    flatter a model that has not moved the fire anywhere.
    """
    features = (scene.get("forecast") or {}).get("features") or []
    if not features:
        return None
    latest = max(f["properties"]["day"] for f in features)
    target = mean_centroid([f for f in features if f["properties"]["day"] == latest])
    if target is None:
        return None

    lon0, lat0 = growth_origin(scene, ignition)
    # Local flat-earth is fine over a 32 km tile; longitude shrinks with latitude.
    scale = math.cos(math.radians(lat0)) or 1.0
    dx = (target[0] - lon0) * scale
    dy = target[1] - lat0
    if math.hypot(dx, dy) < 1e-9:
        return None
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def angular_difference(a: float, b: float) -> float:
    """Smallest angle between two bearings, 0-180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def audit_one(name: str, lat: float, lon: float, date: Optional[str], steps: int,
              model: str = "ignis") -> Dict[str, Any]:
    try:
        payload = get_json(f"{TILESVC}/predict_multistep", {
            "lat": lat, "lon": lon, "steps": steps, "ignition": "true", "date": date,
            "model": model,
        })
    except Exception as exc:  # noqa: BLE001 - the failure itself is a result
        return {"name": name, "status": "error", "detail": str(exc)[:90]}

    # A build that predates ?model= drops it silently and answers with the
    # learned model whatever was asked for, which once produced a baseline
    # comparison where both arms were the same forecaster.
    served = payload.get("model")
    if served != model:
        return {"name": name, "status": f"served {served or 'unknown'}, asked {model}"}

    wind = (payload.get("input_summary") or {}).get("wind") or {}
    if not wind.get("available"):
        return {"name": name, "status": "no_wind"}
    if wind.get("calm"):
        return {"name": name, "status": "calm", "speed_ms": wind.get("speed_ms")}

    scene = payload.get("scene") or {}
    bearing = spread_bearing(scene, (lon, lat))
    if bearing is None:
        return {"name": name, "status": "no_spread", "wind_toward": wind.get("toward")}

    wind_toward = float(wind["toward_deg"])
    delta = angular_difference(bearing, wind_toward)
    return {
        "name": name,
        "status": "ok",
        "model": served,
        "wind_toward_deg": round(wind_toward, 1),
        "wind_toward": wind.get("toward"),
        "wind_ms": wind.get("speed_ms"),
        "spread_bearing_deg": round(bearing, 1),
        "off_by_deg": round(delta, 1),
        # Cosine of the angle: +1 straight downwind, -1 straight upwind.
        "alignment": round(math.cos(math.radians(delta)), 3),
        "verdict": ("downwind" if delta <= 60 else
                    "crosswind" if delta <= 120 else "UPWIND"),
    }


def live_incidents(limit: int) -> List[Dict[str, Any]]:
    payload = get_json(f"{BACKEND}/api/map/bootstrap", {})
    incidents = [i for i in (payload.get("incidents") or []) if i.get("lat") and i.get("lon")]
    western = [i for i in incidents
               if -125.1 <= i["lon"] <= -101.8 and 31 <= i["lat"] <= 49.5]
    western.sort(key=lambda i: -(i.get("acres") or 0))
    return western[:limit]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.audit_direction")
    ap.add_argument("--limit", type=int, default=8, help="How many live incidents to audit")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Audit one known event instead")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--model", default="ignis", choices=("ignis", "downwind"),
                    help="Which forecaster to score. 'downwind' is the bar to clear.")
    args = ap.parse_args(argv)

    if args.preset:
        preset = PRESETS[args.preset]
        targets = [(args.preset, preset["lat"], preset["lon"], preset["date"])]
        print(f"{args.preset}: expected spread {preset['expect']} ({preset['note']})\n")
    else:
        targets = [(i.get("name", "?")[:22], i["lat"], i["lon"], None)
                   for i in live_incidents(args.limit)]
        print(f"auditing {len(targets)} live incidents\n")

    header = f"{'fire':<24}{'wind':>7}{'spread':>9}{'off by':>9}{'align':>8}  verdict"
    print(header)
    print("-" * len(header))

    results = []
    for name, lat, lon, date in targets:
        r = audit_one(name, lat, lon, date, args.steps, args.model)
        results.append(r)
        if r["status"] != "ok":
            print(f"{name:<24}{'—':>7}{'—':>9}{'—':>9}{'—':>8}  {r['status']}")
            continue
        print(f"{name:<24}{r['wind_toward']:>7}{r['spread_bearing_deg']:>9.0f}"
              f"{r['off_by_deg']:>9.0f}{r['alignment']:>8.2f}  {r['verdict']}")

    scored = [r for r in results if r["status"] == "ok"]
    if not scored:
        print("\nNo fire produced a scorable forecast.")
        return 1

    mean_alignment = sum(r["alignment"] for r in scored) / len(scored)
    upwind = sum(1 for r in scored if r["verdict"] == "UPWIND")
    print(f"\n  scored           : {len(scored)} of {len(results)}")
    print(f"  mean alignment   : {mean_alignment:+.3f}   (+1 downwind, -1 upwind)")
    print(f"  running upwind   : {upwind} of {len(scored)}")
    print(f"  forecaster       : {args.model}")
    if args.model == "ignis":
        print("\n  Run again with --model downwind for the bar this has to clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
