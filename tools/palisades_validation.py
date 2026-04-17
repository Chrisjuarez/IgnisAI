#!/usr/bin/env python3
"""Validate Palisades forecast contours against time-matched perimeter data.

This script expects the local/backend API to be running. It writes GeoJSON
artifacts plus metrics JSON so overlay screenshots can be generated from the
same files later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_DATE = "2025-01-07T18:30:00Z"


def fetch_json(api_base: str, path: str, params: dict) -> dict:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}?{urlencode(params)}"
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def union_features(geojson: dict):
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:
        raise SystemExit("Install shapely to compute Palisades validation metrics.") from exc

    geoms = [
        shape(feature["geometry"])
        for feature in geojson.get("features", [])
        if feature.get("geometry")
    ]
    if not geoms:
        return None
    return unary_union(geoms)


def metric_row(step: dict, observed_union) -> dict:
    predicted = union_features(step.get("contour") or {"type": "FeatureCollection", "features": []})
    row = {
        "index": step.get("index"),
        "lead_hours": step.get("lead_hours"),
        "label": step.get("label"),
        "prob_min": step.get("prob_min"),
        "prob_mean": step.get("prob_mean"),
        "prob_max": step.get("prob_max"),
        "area_fraction": step.get("area_fraction"),
        "iou": None,
        "area_ratio": None,
        "centroid_distance_km": None,
    }
    if predicted is None or observed_union is None or predicted.is_empty or observed_union.is_empty:
        return row

    union = predicted.union(observed_union)
    intersection = predicted.intersection(observed_union)
    if union.area > 0:
        row["iou"] = float(intersection.area / union.area)
    if observed_union.area > 0:
        row["area_ratio"] = float(predicted.area / observed_union.area)
    pc = predicted.centroid
    oc = observed_union.centroid
    row["centroid_distance_km"] = float(haversine_km(pc.x, pc.y, oc.x, oc.y))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:5000/api")
    parser.add_argument("--lat", type=float, default=34.05)
    parser.add_argument("--lon", type=float, default=-118.55)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out", default="artifacts/palisades-validation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast = fetch_json(args.api_base, "/predict-fire-spread/multistep", {
        "lat": args.lat,
        "lon": args.lon,
        "date": args.date,
        "steps": args.steps,
    })
    perimeters = fetch_json(args.api_base, "/fire-perimeters", {
        "bbox": "-119.15,33.75,-118.0,34.45",
        "incidentName": "Palisades",
        "from": args.date,
    })

    observed = perimeters.get("geojson") or {"type": "FeatureCollection", "features": []}
    (out_dir / "observed_perimeters.geojson").write_text(json.dumps(observed, indent=2))

    observed_union = union_features(observed)
    rows = []
    for step in forecast.get("steps", []):
        contour = step.get("contour") or {"type": "FeatureCollection", "features": []}
        lead = step.get("lead_hours", step.get("index", 0))
        (out_dir / f"forecast_lead_{lead}.geojson").write_text(json.dumps(contour, indent=2))
        rows.append(metric_row(step, observed_union))

    summary = {
        "incident": "Palisades",
        "ignition_time_utc": args.date,
        "model_meta": forecast.get("model_meta"),
        "input_summary": forecast.get("input_summary"),
        "perimeter_count": len(observed.get("features", [])),
        "metrics": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
