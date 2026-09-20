"""Fire exposure at a fixed asset, sampled from a spread forecast.

The map answers "where will this fire go". An asset owner asks the narrower
question "does it reach *my* site, and when" — which is the same forecast read
at one coordinate instead of rendered as an image.

A rollout is centred on an ignition and spans one 32 km tile, so a site more
than half a tile away is outside the model's field of view. That is reported as
uncovered rather than as zero probability: absence of a forecast is not absence
of risk, and a solar asset owner acting on a fabricated zero is the one outcome
worth engineering against.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import calibrate_probability
from .grid import PIX, SIZE, TileID, lonlat_to_xy_m, tile_affine


#: Calibrated probability at which a site counts as reached. Deliberately low:
#: the operational question is when to act, not when burning is certain.
DEFAULT_ARRIVAL_THRESHOLD = 0.10


def site_pixel(tile: TileID, lon: float, lat: float) -> Optional[Tuple[int, int]]:
    """Row/col of a coordinate inside a tile-aligned raster, or None if outside."""
    x, y = lonlat_to_xy_m(lon, lat)
    col, row = ~tile_affine(tile) * (x, y)
    row, col = math.floor(row), math.floor(col)
    if 0 <= row < SIZE and 0 <= col < SIZE:
        return int(row), int(col)
    return None


def separation_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    """Ground distance in the equal-area projection the tiles are built on."""
    ax, ay = lonlat_to_xy_m(lon_a, lat_a)
    bx, by = lonlat_to_xy_m(lon_b, lat_b)
    return math.hypot(ax - bx, ay - by) / 1000.0


def _uncovered(site_lon: float, site_lat: float, ignition_lon: float, ignition_lat: float) -> Dict[str, Any]:
    return {
        "covered": False,
        "reason": "site_outside_forecast_tile",
        "detail": (
            f"The forecast covers a {SIZE * PIX // 1000} km tile centred on the ignition; "
            "this site falls outside it. Re-run with an ignition nearer the site."
        ),
        "separation_km": round(separation_km(site_lon, site_lat, ignition_lon, ignition_lat), 2),
        "series": [],
        "probability_within_horizon": None,
        "risk_within_horizon": None,
        "peak_day": None,
        "arrival": None,
    }


def sample_exposure(
    rollout: Sequence[Dict[str, Any]],
    tile: TileID,
    *,
    site_lon: float,
    site_lat: float,
    ignition_lon: float,
    ignition_lat: float,
    arrival_threshold: float = DEFAULT_ARRIVAL_THRESHOLD,
    model_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a rollout at one coordinate and describe the exposure over time.

    Probabilities go through the same calibration the rendered map uses, so a
    number quoted for a site matches the colour a viewer sees at that spot.
    """
    pixel = site_pixel(tile, site_lon, site_lat)
    if pixel is None:
        return _uncovered(site_lon, site_lat, ignition_lon, ignition_lat)

    row, col = pixel
    raw = np.array([float(step["prob"][row, col]) for step in rollout], dtype=np.float32)

    # The model predicts a delta - cells that newly burn on each step - so a
    # site that burns on day 2 shows a LOW daily value on day 3 simply because
    # it has already burned. Reporting that raw series to an asset owner would
    # say "day 3: low risk" about a site the forecast already burned down.
    # Carrying the running maximum forward mirrors the AR feedback the rollout
    # itself uses (next_fire = max(prev_fire, prob)), so cumulative exposure
    # stays consistent with what the model carries between steps.
    cumulative_raw = np.maximum.accumulate(raw)

    daily_score, _daily_risk, calibration = calibrate_probability(raw, model_sha256=model_sha256)
    cumulative_score, cumulative_risk, _ = calibrate_probability(cumulative_raw, model_sha256=model_sha256)

    series: List[Dict[str, Any]] = []
    for index, step in enumerate(rollout):
        series.append({
            "day": index + 1,
            "lead_hours": int(step["lead_hours"]),
            "label": step.get("label"),
            "daily_probability": round(float(daily_score[index]), 4),
            "cumulative_probability": round(float(cumulative_score[index]), 4),
            "risk": str(cumulative_risk[index]),
        })

    peak_day = max(series, key=lambda entry: entry["daily_probability"]) if series else None
    reached = next((e for e in series if e["cumulative_probability"] >= arrival_threshold), None)

    return {
        "covered": True,
        "separation_km": round(separation_km(site_lon, site_lat, ignition_lon, ignition_lat), 2),
        "series": series,
        "probability_within_horizon": series[-1]["cumulative_probability"] if series else None,
        "risk_within_horizon": series[-1]["risk"] if series else None,
        "peak_day": peak_day,
        "arrival": {
            "threshold": arrival_threshold,
            "reached": reached is not None,
            "day": reached["day"] if reached else None,
            "lead_hours": reached["lead_hours"] if reached else None,
        },
        "calibration": calibration,
    }
