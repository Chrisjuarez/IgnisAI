#!/usr/bin/env python3
"""Inspect cached NOAA wind vectors for a given event window.

Usage:
    python -m tools.dump_cached_wind --profile palisades

For each NOAA cache npz under `.cache/runtime_cache/{profile}/noaa_grid_cache/`
this prints the mean u (eastward m/s), mean v (northward m/s), mean wind
speed, and the bearing the wind is BLOWING TOWARD (compass degrees, 0 = N,
90 = E). That last number is the directional sanity check:

* Santa Ana (offshore from the LA basin) → wind blowing toward 180-270°
  (south through west). For Palisades Jan 7-8 we expect bearing-toward
  in the 200-250° range.
* Onshore flow (post-Santa Ana, Jan 9+) → wind blowing toward 30-90°
  (north through east, inland).

If the bearing-toward number for Jan 7-8 is in the 0-90° range (NE),
the cache is genuinely telling the model the wind is blowing inland —
which would explain the prediction bias east. If it's 200-250° (SW),
the cache is correct and the model itself is biased.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def bearing_toward(u: float, v: float) -> float:
    """
    Convert (u eastward m/s, v northward m/s) into a compass bearing
    indicating the direction the wind is BLOWING TOWARD (degrees,
    0 = north, increasing clockwise). Returns NaN if speed ~ 0.
    """
    speed = math.sqrt(u * u + v * v)
    if speed < 0.1:
        return float("nan")
    # atan2(u, v) gives compass bearing (0=N, 90=E) for (eastward, northward).
    deg = math.degrees(math.atan2(u, v))
    return (deg + 360.0) % 360.0


def compass_label(bearing: float) -> str:
    if math.isnan(bearing):
        return "calm"
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) // 45) % 8
    return sectors[idx]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.dump_cached_wind")
    parser.add_argument("--profile", default="palisades")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/runtime_cache"),
        help="Root that holds {profile}/noaa_grid_cache/*.npz",
    )
    args = parser.parse_args(argv)
    cache_dir = args.cache_root / args.profile / "noaa_grid_cache"
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        print(f"no npz files under {cache_dir}", file=sys.stderr)
        return 1
    print(f"{'file':>40s}  {'u m/s':>8s}  {'v m/s':>8s}  {'spd':>6s}  {'toward':>7s}  dir  source")
    for p in files:
        with np.load(p, allow_pickle=False) as d:
            u = float(np.mean(d["u"]))
            v = float(np.mean(d["v"]))
            source = (
                str(np.asarray(d["source"]).item()) if "source" in d.files else "unknown"
            )
        speed = math.sqrt(u * u + v * v)
        bearing = bearing_toward(u, v)
        b_str = f"{bearing:6.1f}°" if not math.isnan(bearing) else "  -   "
        print(
            f"{p.name:>40s}  {u:8.2f}  {v:8.2f}  {speed:6.2f}  {b_str}  {compass_label(bearing):>3s}  {source}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
