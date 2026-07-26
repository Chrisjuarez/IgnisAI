#!/usr/bin/env python3
"""Georeferenced export + registration validation for prediction rasters.

Purpose
-------
Getting a fire prediction onto a map *accurately* is a separate problem from
producing a good prediction, and it fails silently. A raster shifted by two
pixels still looks plausible — it just describes the wrong 750 m of ground,
which for an evacuation decision is the whole game.

This module does two things:

  1. **Export** any band on the preset grid as a Cloud-Optimized GeoTIFF in its
     NATIVE CRS (EPSG:5070, 375 m). Never pre-warped to Web Mercator: baking a
     display projection into stored data resamples it twice and moves pixels off
     the ground they were computed for. Store honestly, reproject at render.

  2. **Validate registration WITHOUT external reference data**, by cross-checking
     two independently-sourced rasters that were pushed through the same
     reprojection code:

         water from NLCD landcover (IGBP class 17)   [local NLCD GeoTIFF]
         water from NASADEM elevation (near 0 m)      [Earthdata NASADEM]

     Different agencies, different sensors, different download paths, same
     ground truth. If the reprojection is correct they agree spatially; if the
     transform, CRS, or axis order is wrong they diverge. Palisades is the
     decisive case — 12% of that tile is the Pacific, so a misregistration
     shows up immediately as a displaced coastline.

Why this matters for display
----------------------------
* **Do not smooth.** Bilinear upsampling of a 128x128 probability field renders
  as though it were high-resolution data. It is 375 m. Nearest-neighbour at
  native resolution is the honest choice; `--smooth` exists only to demonstrate
  the difference.
* **Display CALIBRATED probability only.** On the Phase A checkpoint a raw 0.50
  corresponds to an observed burn rate of 0.137 — a raw heatmap overstates risk
  by ~3.6x through the mid-range. `risk_breaks` are meaningful only post-calibration.
* **Show uncertainty.** The 12-fold ensemble yields per-pixel spread; collapsing
  it to a single mean discards the most decision-relevant signal.

Usage
-----
    python render_geo.py --validate              # registration cross-check
    python render_geo.py --export-inputs         # COGs of the built bands
    python render_geo.py --preset palisades --figure out.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presets as presets_mod          # noqa: E402

TARGET_RES_M = 375.0
TARGET_SIZE = 128
DST_CRS = "EPSG:5070"

IGBP_WATER = 17
#: NASADEM "this is essentially the ocean" threshold.
#: 5 m was too permissive: in coastal LA it also captures Santa Monica, Venice
#: and the beaches, so DEM-low covered 27.1% of the Palisades tile against
#: NLCD's 11.6% water. That is a semantics gap, not a registration error.
SEA_LEVEL_M = 2.0

#: Minimum NLCD water fraction for the check to carry any signal. The landcover
#: layer is the REFERENCE here, so its coverage is what determines power —
#: Camp had 0.8% NLCD water but 0% DEM-low and was reported "ALIGNED", which
#: was meaningless.
MIN_WATER_FRAC = 0.02


# ---------------------------------------------------------------------------
def tile_transform(lat: float, lon: float):
    """The one affine everything on this grid must share.

    Inputs, predictions and overlays all derive their geometry here. If this
    is computed differently in two places, layers drift apart and nothing
    downstream can detect it.
    """
    from rasterio.transform import from_origin
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    cx, cy = tx.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    return from_origin(cx - half, cy + half, TARGET_RES_M, TARGET_RES_M)


def bounds_lonlat(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """(west, south, east, north) — what a map library needs for placement."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    inv = Transformer.from_crs(DST_CRS, "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    w, s = inv.transform(cx - half, cy - half)
    e, n = inv.transform(cx + half, cy + half)
    return (w, s, e, n)


def export_cog(arr: np.ndarray, lat: float, lon: float, out_path: Path,
               band_names: Optional[List[str]] = None,
               tags: Optional[dict] = None) -> Path:
    """Write [H,W] or [B,H,W] as a COG in the native grid."""
    import rasterio
    a = arr[None] if arr.ndim == 2 else arr
    profile = {
        "driver": "COG", "dtype": "float32", "count": a.shape[0],
        "height": a.shape[1], "width": a.shape[2],
        "crs": DST_CRS, "transform": tile_transform(lat, lon),
        "compress": "DEFLATE", "predictor": 2, "nodata": np.nan,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst = rasterio.open(out_path, "w", **profile)
    except Exception:
        profile.update(driver="GTiff", tiled=True, blockxsize=128, blockysize=128)
        dst = rasterio.open(out_path, "w", **profile)
    with dst:
        dst.write(a.astype(np.float32))
        for i, nm in enumerate(band_names or [], start=1):
            if i <= a.shape[0]:
                dst.set_band_description(i, nm)
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    return out_path


# ---------------------------------------------------------------------------
# registration cross-check
# ---------------------------------------------------------------------------
def load_band(root: Path, preset_key: str, name: str) -> Optional[np.ndarray]:
    for sub in ("static", "viirs"):
        p = root / sub / preset_key / f"{preset_key}_{name}_375m.npy"
        if p.is_file():
            return np.load(p)
    return None


def registration_check(root: Path, preset) -> dict:
    """Agreement between NLCD-water and NASADEM-water on the same grid.

    Two independent sources through the same reprojection path. High agreement
    means the transform is right; low agreement means one layer is displaced,
    and no amount of downstream styling will fix that.
    """
    lc = load_band(root, preset.key, "landcover_igbp")
    elev = load_band(root, preset.key, "elev")
    if lc is None or elev is None:
        return {"preset": preset.key, "status": "missing_bands",
                "have_landcover": lc is not None, "have_elev": elev is not None}

    water_lc = (lc == IGBP_WATER)
    water_el = (elev < SEA_LEVEL_M)
    frac_lc, frac_el = float(water_lc.mean()), float(water_el.mean())

    # The landcover layer is the reference; if it has almost no water the test
    # cannot say anything, regardless of what the DEM shows.
    if frac_lc < MIN_WATER_FRAC:
        return {"preset": preset.key, "status": "insufficient_water",
                "note": (f"NLCD water {frac_lc:.1%} < {MIN_WATER_FRAC:.0%} — "
                         "inland tile, this check has no power"),
                "water_frac_landcover": frac_lc, "water_frac_elev": frac_el}

    inter = float((water_lc & water_el).sum())
    union = float((water_lc | water_el).sum())
    iou = inter / union if union else 0.0

    # CONTAINMENT is the metric that actually tests registration: every NLCD
    # water pixel should sit on DEM-low ground. IoU additionally penalises
    # DEM-low pixels that are legitimately dry land near sea level (beaches,
    # coastal flats), which is a difference in what the two layers MEAN, not a
    # difference in where they are.
    containment = inter / float(water_lc.sum()) if water_lc.sum() else 0.0
    # How far would we have to shift to agree better? Search +/-3 px.
    #
    # Ties MUST resolve toward the smallest shift. A coastline that is uniform
    # along one axis is invariant to shifts along that axis, so many offsets
    # score identically; keeping the first maximum found reports a large
    # spurious displacement. Require a real improvement to move away from zero,
    # and among equals prefer the smallest magnitude.
    TOL = 1e-9
    best = (0, 0, iou)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = np.roll(np.roll(water_el, dy, axis=0), dx, axis=1)
            u = float((water_lc | shifted).sum())
            if not u:
                continue
            v = float((water_lc & shifted).sum()) / u
            better = v > best[2] + TOL
            tied_but_closer = (abs(v - best[2]) <= TOL
                               and np.hypot(dy, dx) < np.hypot(best[0], best[1]))
            if better or tied_but_closer:
                best = (dy, dx, v)
    # A genuine offset shows up as a LARGE IoU gain from shifting. A tiny gain
    # means the layers already coincide and the residual is threshold semantics.
    gain = best[2] - iou
    return {"preset": preset.key, "status": "ok",
            "water_frac_landcover": frac_lc, "water_frac_elev": frac_el,
            "containment": containment,
            "iou_at_zero_shift": iou,
            "best_shift_px": [best[0], best[1]], "iou_at_best_shift": best[2],
            "iou_gain_from_shift": gain,
            "offset_m": float(np.hypot(best[0], best[1]) * TARGET_RES_M),
            "registered": bool(containment >= 0.90 and gain < 0.05)}


# ---------------------------------------------------------------------------
def make_figure(root: Path, preset, out_png: Path, smooth: bool = False) -> Path:
    """Input-band panel with correct geographic extent and honest resampling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, s, e, n = bounds_lonlat(preset.lat, preset.lon)
    extent = (w, e, s, n)
    interp = "bilinear" if smooth else "nearest"

    panels = [("elev", "Elevation (m)", "terrain"),
              ("slope", "Slope (deg)", "magma"),
              ("landcover_igbp", "Landcover (IGBP)", "tab20"),
              ("EVI2", "EVI2", "YlGn")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (name, title, cmap) in zip(np.atleast_1d(axes), panels):
        a = load_band(root, preset.key, name)
        if a is None:
            ax.text(0.5, 0.5, f"{name}\nnot built", ha="center", va="center")
            ax.set_axis_off(); continue
        if a.ndim == 3:
            a = a[-1]                       # last timestep for temporal cubes
        im = ax.imshow(a, extent=extent, origin="upper",
                       interpolation=interp, cmap=cmap)
        ax.plot(preset.lon, preset.lat, "r*", ms=13, mec="k", mew=0.6)
        ax.set_title(f"{title}\n{preset.name}", fontsize=9)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{preset.key} — {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_RES_M:.0f} m, "
                 f"{DST_CRS}  (resampling: {interp})", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Geo export + registration validation")
    ap.add_argument("--preset")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--export-inputs", action="store_true")
    ap.add_argument("--figure", type=Path)
    ap.add_argument("--smooth", action="store_true",
                    help="bilinear instead of nearest — for demonstrating why not to")
    args = ap.parse_args(argv)

    data_root = Path(os.environ.get("IGNIS_DATA_ROOT", "data"))
    root = data_root / "wsts_inputs"
    targets = ([presets_mod.by_key(args.preset)] if args.preset
               else list(presets_mod.PRESETS))

    print(f"grid: {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_RES_M:.0f} m  {DST_CRS}")
    for p in targets:
        w, s, e, n = bounds_lonlat(p.lat, p.lon)
        print(f"  {p.key:<12} bounds ({w:.4f},{s:.4f}) .. ({e:.4f},{n:.4f})")

    if args.validate:
        print("\n=== registration cross-check (NLCD water vs NASADEM sea level) ===")
        results = []
        for p in targets:
            r = registration_check(root, p)
            results.append(r)
            if r["status"] == "missing_bands":
                print(f"  {p.key:<12} bands missing "
                      f"(landcover={r['have_landcover']} elev={r['have_elev']})")
            elif r["status"] == "insufficient_water":
                print(f"  {p.key:<12} no power — NLCD water "
                      f"{r['water_frac_landcover']:.1%}")
            else:
                verdict = "REGISTERED" if r["registered"] else "SUSPECT"
                print(f"  {p.key:<12} {verdict}  containment="
                      f"{r['containment']:.3f}  IoU={r['iou_at_zero_shift']:.3f}")
                print(f"               water: NLCD {r['water_frac_landcover']:.1%} / "
                      f"DEM<{SEA_LEVEL_M:.0f}m {r['water_frac_elev']:.1%}")
                print(f"               best shift {r['best_shift_px']} px, "
                      f"IoU gain {r['iou_gain_from_shift']:+.3f} "
                      f"({'negligible — no offset' if r['iou_gain_from_shift'] < 0.05 else 'MATERIAL — real offset'})")
        out = root / "_registration_check.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  -> {out}")
        print("  containment = fraction of NLCD water sitting on DEM-low ground;")
        print("  it tests WHERE the layers are. IoU also penalises dry land near")
        print("  sea level, which is a difference in meaning, not position.")
        print("  A real misregistration shows a LARGE IoU gain from shifting.")

    if args.export_inputs:
        print("\n=== COG export (native CRS, no Mercator pre-warp) ===")
        for p in targets:
            for name in ("elev", "slope", "aspect", "landcover_igbp"):
                a = load_band(root, p.key, name)
                if a is None:
                    continue
                dst = root / "cog" / p.key / f"{p.key}_{name}.tif"
                export_cog(a, p.lat, p.lon, dst, [name],
                           {"source": "research/wsts", "res_m": TARGET_RES_M})
                print(f"  {dst.relative_to(root)}")

    if args.figure:
        for p in targets:
            out = (args.figure if len(targets) == 1
                   else args.figure.with_name(f"{args.figure.stem}_{p.key}.png"))
            print(f"\nfigure -> {make_figure(root, p, out, args.smooth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
