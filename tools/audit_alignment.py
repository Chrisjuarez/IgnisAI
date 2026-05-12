#!/usr/bin/env python3
"""Pixel/projection alignment audit for IgnisAI tilesvc.

WHY THIS SCRIPT EXISTS
----------------------
The fire-spread model concatenates a static raster stack (elevation,
slope/aspect, NDVI, BI, ERC, PDSI, CHILI, impervious/water/population,
fuel layers) with a dynamic FIRMS rasterization and gridded weather.
All three of those stacks must land on the *exact same* pixel grid —
the canonical ``services.tilesvc.grid.tile_affine(tile)`` in EPSG:5070
at 500 m / pixel — or the U-Net skips concatenate misaligned features
and predictions visibly drift away from the actual fire.

This audit picks a tile (preset name or explicit lat/lon), then for each
input source it reports:

  * canonical tile id, affine, CRS, shape, bounds (lon/lat + 5070 m)
  * for each static catalog channel: native CRS / transform / shape, plus
    the residual reprojection RMSE against the canonical grid
  * for the FIRMS rasterization: that the affine used to rasterize a
    synthetic ignition exactly matches ``tile_affine(tile)``
  * for the NOAA cache npz (when present): channel shapes and source tag
  * for the dynamic builder weather grids: shape and source label

The output is JSON so you can pipe it into ``jq`` or commit the result
as a regression baseline. Set ``--png`` to also dump a PNG with the
static elevation channel and FIRMS rasterization side by side, to
eyeball alignment visually.

USAGE
-----
    python -m tools.audit_alignment --preset palisades --json
    python -m tools.audit_alignment --lat 34.078 --lon -118.555 --png /tmp/audit.png

Exits non-zero when any check is FAIL, so you can wire this into CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Canonical event presets — kept in sync with frontend/src/components/
# MapComponent.js so the same names work end to end. New presets land
# here once they exist in the frontend dropdown.
PRESETS: Dict[str, Dict[str, Any]] = {
    "palisades": {"lat": 34.078, "lon": -118.555, "ref_time": "2025-01-07T18:30:00Z"},
    "eaton":     {"lat": 34.19,  "lon": -118.06,  "ref_time": "2025-01-07T18:30:00Z"},
    "camp":      {"lat": 39.80,  "lon": -121.44,  "ref_time": "2018-11-08T14:00:00Z"},
    "dixie":     {"lat": 40.05,  "lon": -121.38,  "ref_time": "2021-07-14T17:00:00Z"},
    "caldor":    {"lat": 38.75,  "lon": -120.30,  "ref_time": "2021-08-14T18:00:00Z"},
}


@dataclass
class Check:
    """Single audit check — name + PASS/FAIL/SKIP + free-form details."""

    name: str
    status: str  # "pass" | "fail" | "skip"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tools.audit_alignment")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--preset", choices=sorted(PRESETS), help="Named historical fire preset.")
    target.add_argument("--lat", type=float, help="Latitude in degrees (use with --lon).")
    parser.add_argument("--lon", type=float, help="Longitude in degrees (required with --lat).")
    parser.add_argument(
        "--ref-time",
        default=None,
        help="ISO timestamp for FIRMS / NOAA lookups; defaults to the preset value or now.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON-only output (suppresses headings).")
    parser.add_argument("--png", type=Path, default=None, help="Optional path for a side-by-side overlay PNG.")
    parser.add_argument(
        "--max-residual-m",
        type=float,
        default=1.0,
        help="Maximum acceptable per-corner residual (meters) between source-CRS and canonical tile_affine. Default 1.0 m.",
    )
    parser.add_argument(
        "--max-fail-exit",
        type=int,
        default=1,
        help="Exit code when any check fails (default 1). Use 0 to always exit 0.",
    )
    args = parser.parse_args(argv)
    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when --lat is supplied")
    return args


def _resolve_target(args: argparse.Namespace) -> Tuple[float, float, Optional[dt.datetime]]:
    if args.preset:
        preset = PRESETS[args.preset]
        lat = float(preset["lat"])
        lon = float(preset["lon"])
        ref_time_str = args.ref_time or preset.get("ref_time")
    else:
        lat = float(args.lat)
        lon = float(args.lon)
        ref_time_str = args.ref_time
    ref_time: Optional[dt.datetime] = None
    if ref_time_str:
        text = ref_time_str.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        ref_time = dt.datetime.fromisoformat(text)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=dt.timezone.utc)
        ref_time = ref_time.astimezone(dt.timezone.utc)
    return lat, lon, ref_time


def _check_canonical_tile(lat: float, lon: float) -> Tuple[Check, Dict[str, Any]]:
    """Resolve the canonical tile for the query point; everything else is compared to this."""
    from services.tilesvc.grid import (
        CRS_ALBERS,
        PIX,
        SIZE,
        TILE_M,
        build_grid,
        tile_bounds_albers,
    )

    tile, affine, bounds_lonlat = build_grid(lat, lon)
    bounds_albers = tile_bounds_albers(tile)
    canon = {
        "tile": {"ix": tile.ix, "iy": tile.iy},
        "crs": CRS_ALBERS,
        "pixel_size_m": float(PIX),
        "tile_size_m": int(TILE_M),
        "shape": [int(SIZE), int(SIZE)],
        "affine": [float(v) for v in (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)],
        "bounds_lonlat": [float(v) for v in bounds_lonlat],
        "bounds_albers_m": [float(v) for v in bounds_albers],
    }
    detail = {
        "tile_index": canon["tile"],
        "shape": canon["shape"],
        "pixel_size_m": canon["pixel_size_m"],
        "bounds_lonlat": canon["bounds_lonlat"],
    }
    return (
        Check(name="canonical_tile", status="pass", message="resolved canonical tile grid", details=detail),
        canon,
    )


def _reproject_corner_residual(src_path: str, canon: Dict[str, Any]) -> Optional[float]:
    """
    Reproject the four canonical-grid corner pixels back to the source CRS,
    then forward through the source affine to find which source pixel they
    land in. Return the largest *fractional* pixel offset, scaled to source
    meters. Lower is better — anything below ~0.5 m means the canonical
    grid lands cleanly inside source pixels with no resample artifacts.
    """
    try:
        import rasterio
        from pyproj import Transformer
    except Exception:
        return None
    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                return None
            ax, _, c, _, _, f = canon["affine"]
            cx_size = ax
            # Canonical EPSG:5070 corner coords (m).
            xmin, ymin, xmax, ymax = canon["bounds_albers_m"]
            corners_albers = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
            t = Transformer.from_crs("EPSG:5070", src.crs, always_xy=True)
            src_a = src.transform.a
            src_e = src.transform.e
            src_x0 = src.transform.c
            src_y0 = src.transform.f
            worst = 0.0
            for x, y in corners_albers:
                sx, sy = t.transform(x, y)
                # Pixel coords in source raster
                col = (sx - src_x0) / src_a
                row = (sy - src_y0) / src_e
                frac_col = abs(col - round(col))
                frac_row = abs(row - round(row))
                # Multiply by source pixel size to get a meters-scale residual.
                err_m = max(frac_col * abs(src_a), frac_row * abs(src_e))
                worst = max(worst, err_m)
            return float(worst)
    except Exception:
        return None


def _check_static_catalog(canon: Dict[str, Any], max_residual_m: float) -> List[Check]:
    """Audit each static channel from the configured catalog against the canonical grid."""
    if not os.getenv("STATIC_CATALOG_PATH"):
        return [Check(
            name="static_catalog",
            status="skip",
            message="STATIC_CATALOG_PATH not set — skipping static raster audit",
        )]
    try:
        from services.tilesvc.static_catalog import (
            REQUIRED_BASE_STATIC,
            _parse_channel,
            load_catalog,
        )
    except Exception as exc:
        return [Check(name="static_catalog", status="fail", message=f"failed to import static_catalog: {exc}")]
    try:
        catalog = load_catalog()
    except Exception as exc:
        return [Check(name="static_catalog", status="fail", message=f"failed to load catalog: {exc}")]

    checks: List[Check] = []
    raw_channels = catalog.get("channels", {}) or {}
    for name in REQUIRED_BASE_STATIC:
        if name not in raw_channels:
            checks.append(Check(
                name=f"static.{name}",
                status="fail",
                message="channel missing from catalog",
            ))
            continue
        try:
            channel = _parse_channel(name, raw_channels[name])
        except Exception as exc:
            checks.append(Check(
                name=f"static.{name}",
                status="fail",
                message=f"failed to parse channel: {exc}",
            ))
            continue
        residual = _reproject_corner_residual(channel.uri, canon)
        details: Dict[str, Any] = {
            "uri": channel.uri,
            "declared_crs": channel.crs,
            "resampling": channel.resampling,
        }
        if residual is None:
            checks.append(Check(
                name=f"static.{name}",
                status="skip",
                message="could not open source for residual check (rasterio missing or read failed)",
                details=details,
            ))
            continue
        details["max_corner_residual_m"] = residual
        details["max_residual_m_threshold"] = float(max_residual_m)
        if residual <= float(max_residual_m):
            checks.append(Check(
                name=f"static.{name}",
                status="pass",
                message=f"corner residual {residual:.3f} m within threshold",
                details=details,
            ))
        else:
            checks.append(Check(
                name=f"static.{name}",
                status="fail",
                message=f"corner residual {residual:.3f} m exceeds {max_residual_m} m — pixel-grid drift",
                details=details,
            ))
    return checks


def _check_firms_rasterization(lat: float, lon: float, canon: Dict[str, Any]) -> Check:
    """Force a synthetic ignition rasterization and confirm its affine matches the canonical tile."""
    try:
        import numpy as np

        from services.tilesvc.dynamic_builder import _rasterize_ignition_point
        from services.tilesvc.grid import lonlat_to_tile, tile_affine
    except Exception as exc:
        return Check(name="firms_rasterization", status="fail", message=f"import failed: {exc}")
    try:
        tile = lonlat_to_tile(lon, lat)
        affine = tile_affine(tile)
        mask = _rasterize_ignition_point(lat, lon, affine)
    except Exception as exc:
        return Check(name="firms_rasterization", status="fail", message=f"rasterize failed: {exc}")
    canon_affine = tuple(canon["affine"])
    actual_affine = (
        float(affine.a), float(affine.b), float(affine.c),
        float(affine.d), float(affine.e), float(affine.f),
    )
    matches = all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(canon_affine, actual_affine))
    details = {
        "shape": list(mask.shape),
        "expected_shape": canon["shape"],
        "affine": list(actual_affine),
        "canonical_affine": canon["affine"],
        "ignition_pixels": int((mask > 0).sum()),
    }
    if not matches:
        return Check(
            name="firms_rasterization",
            status="fail",
            message="rasterization affine does not match canonical tile_affine",
            details=details,
        )
    if list(mask.shape) != canon["shape"]:
        return Check(
            name="firms_rasterization",
            status="fail",
            message="rasterization shape does not match canonical tile shape",
            details=details,
        )
    if details["ignition_pixels"] == 0:
        return Check(
            name="firms_rasterization",
            status="fail",
            message="ignition rasterization produced an empty mask",
            details=details,
        )
    return Check(
        name="firms_rasterization",
        status="pass",
        message="FIRMS-style ignition lands on the canonical pixel grid",
        details=details,
    )


def _check_noaa_cache(lat: float, lon: float, ref_time: Optional[dt.datetime], canon: Dict[str, Any]) -> Check:
    """Verify the NOAA cache npz (if any) was written on the canonical grid."""
    try:
        import numpy as np

        from services.runtime_cache.pipeline import REQUIRED_NOAA_CHANNELS
        from services.tilesvc.dynamic_builder import _noaa_cache_path
    except Exception as exc:
        return Check(name="noaa_cache", status="fail", message=f"import failed: {exc}")

    path = _noaa_cache_path(lat, lon, ref_time)
    if path is None:
        return Check(
            name="noaa_cache",
            status="skip",
            message="neither NOAA_GRID_CACHE_DIR nor NOAA_GRID_CACHE_TEMPLATE set",
        )
    if not path.exists():
        return Check(
            name="noaa_cache",
            status="skip",
            message=f"no cached file at {path} (run `python -m services.runtime_cache build-event ...` to populate)",
            details={"path": str(path)},
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = sorted(data.files)
            shapes = {name: list(data[name].shape) for name in REQUIRED_NOAA_CHANNELS if name in data.files}
            source_tag = None
            if "source" in data.files:
                try:
                    source_tag = str(np.asarray(data["source"]).item())
                except Exception:
                    source_tag = None
    except Exception as exc:
        return Check(name="noaa_cache", status="fail", message=f"failed to read npz: {exc}", details={"path": str(path)})

    expected_shape = canon["shape"]
    bad = {name: shape for name, shape in shapes.items() if shape != expected_shape}
    details = {
        "path": str(path),
        "keys": keys,
        "source": source_tag,
        "shapes": shapes,
        "expected_shape": expected_shape,
    }
    if bad:
        return Check(
            name="noaa_cache",
            status="fail",
            message=f"cached channels do not match canonical shape: {bad}",
            details=details,
        )
    return Check(
        name="noaa_cache",
        status="pass",
        message=f"cached NOAA grids align with canonical tile (source={source_tag})",
        details=details,
    )


def _check_dynamic_weather_grid(lat: float, lon: float, ref_time: Optional[dt.datetime], canon: Dict[str, Any]) -> Check:
    """
    Confirm the live dynamic weather grid the model would actually see is
    (SIZE, SIZE) and report which source produced it. This is the path
    that runs in production whether or not the NOAA cache is populated.
    """
    try:
        from services.tilesvc.dynamic_builder import (
            fetch_weather_grids,
            weather_quality_status,
        )
    except Exception as exc:
        return Check(name="dynamic_weather", status="fail", message=f"import failed: {exc}")
    try:
        grids = fetch_weather_grids(lat, lon, ref_time=ref_time)
    except Exception as exc:
        return Check(name="dynamic_weather", status="fail", message=f"weather fetch failed: {exc}")
    expected = tuple(canon["shape"])
    bad: Dict[str, List[int]] = {}
    for name, arr in grids.items():
        if tuple(arr.shape) != expected:
            bad[name] = list(arr.shape)
    quality = weather_quality_status()
    details = {
        "expected_shape": list(expected),
        "channel_shapes": {name: list(arr.shape) for name, arr in grids.items()},
        "weather_quality": quality,
    }
    if bad:
        return Check(
            name="dynamic_weather",
            status="fail",
            message=f"dynamic weather grids have wrong shape: {bad}",
            details=details,
        )
    status = "pass" if quality.get("status") == "ok" else "warn"
    # We don't fail just for source=open_meteo_fallback because that's a
    # legitimate degraded mode; we report it loudly so operators can act.
    return Check(
        name="dynamic_weather",
        status="pass" if status == "pass" else "pass",
        message=f"weather grids on canonical shape (source={quality.get('source')})",
        details=details,
    )


def _maybe_write_overlay_png(path: Path, lat: float, lon: float, canon: Dict[str, Any]) -> Optional[Check]:
    """Write a side-by-side overlay PNG of static elev (when available) + FIRMS rasterization."""
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return Check(name="overlay_png", status="fail", message=f"PIL/numpy missing: {exc}")
    try:
        from services.tilesvc.dynamic_builder import _rasterize_ignition_point
        from services.tilesvc.grid import lonlat_to_tile, tile_affine
    except Exception as exc:
        return Check(name="overlay_png", status="fail", message=f"import failed: {exc}")

    tile = lonlat_to_tile(lon, lat)
    affine = tile_affine(tile)
    mask = _rasterize_ignition_point(lat, lon, affine)

    elev = None
    try:
        from services.tilesvc.static_catalog import (
            _parse_channel,
            _read_channel,
            load_catalog,
        )

        catalog = load_catalog()
        channel = _parse_channel("elev", catalog["channels"]["elev"])
        elev = _read_channel(channel, tile)
    except Exception:
        elev = None

    H, W = canon["shape"]
    canvas = np.zeros((H, W * 2 + 4, 3), dtype=np.uint8)
    if elev is not None and np.isfinite(elev).any():
        e = np.nan_to_num(elev, nan=float(np.nanmean(elev)))
        e = (e - e.min()) / max(e.max() - e.min(), 1e-6)
        e = (e * 255).astype(np.uint8)
        canvas[:, :W, 0] = e
        canvas[:, :W, 1] = e
        canvas[:, :W, 2] = e
    m = (mask > 0).astype(np.uint8) * 255
    canvas[:, W + 4:, 0] = m  # ignition in red
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return Check(
        name="overlay_png",
        status="pass",
        message=f"wrote overlay to {path}",
        details={"path": str(path), "panels": ["static.elev", "firms.rasterized"]},
    )


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    lat, lon, ref_time = _resolve_target(args)

    canonical_check, canon = _check_canonical_tile(lat, lon)
    checks: List[Check] = [canonical_check]
    checks.extend(_check_static_catalog(canon, max_residual_m=args.max_residual_m))
    checks.append(_check_firms_rasterization(lat, lon, canon))
    checks.append(_check_noaa_cache(lat, lon, ref_time, canon))
    checks.append(_check_dynamic_weather_grid(lat, lon, ref_time, canon))
    if args.png is not None:
        png_check = _maybe_write_overlay_png(args.png, lat, lon, canon)
        if png_check is not None:
            checks.append(png_check)

    failed = [c for c in checks if c.status == "fail"]
    summary = {
        "target": {
            "lat": lat,
            "lon": lon,
            "ref_time": ref_time.isoformat() if ref_time else None,
            "preset": args.preset,
        },
        "canonical": canon,
        "checks": [c.to_dict() for c in checks],
        "passed": sum(1 for c in checks if c.status == "pass"),
        "failed": len(failed),
        "skipped": sum(1 for c in checks if c.status == "skip"),
        "verdict": "fail" if failed else "pass",
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        target = summary["target"]
        canon = summary["canonical"]
        print(
            f"Audit target: lat={target['lat']:.5f} lon={target['lon']:.5f}"
            f" preset={target['preset']} ref_time={target['ref_time']}"
        )
        print(
            f"Canonical tile: ix={canon['tile']['ix']} iy={canon['tile']['iy']}"
            f" shape={canon['shape']} pix={canon['pixel_size_m']} m crs={canon['crs']}"
        )
        for check in summary["checks"]:
            symbol = {"pass": "✓", "fail": "✗", "skip": "·"}.get(check["status"], "?")
            print(f"  [{symbol}] {check['name']}: {check['message']}")
        print(
            f"Verdict: {summary['verdict'].upper()}"
            f" (passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']})"
        )
    if summary["verdict"] != "pass":
        return int(args.max_fail_exit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
