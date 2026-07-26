#!/usr/bin/env python3
"""Build the last three WSTS bands from data IgnisAI already owns.

After `fetch_viirs.py`, 20 of 23 bands are sourceable. The remaining three are
local computation — no downloads beyond pulling the DEM if it is not cached:

    band 12 : Slope             <- SRTM DEM
    band 13 : Aspect (DEGREES)  <- SRTM DEM
    band 16 : Landcover class   <- NLCD, crosswalked to MODIS IGBP 1..17

Output matches fetch_viirs.py exactly: 128x128 @ 375 m, EPSG:5070, centered on
each preset, saved as .npy beside the VIIRS cubes.

----------------------------------------------------------------------------
TWO THINGS THAT ARE EASY TO GET WRONG
----------------------------------------------------------------------------
1. **Aspect must be DEGREES, not cos/sin.** WSTS applies `sin(deg2rad(x))` to
   bands 7, 13, 19 inside its dataloader. IgnisAI stores aspect as a
   (aspect_cos, aspect_sin) pair, so feeding those directly would double-apply
   the transform and silently corrupt the channel.

2. **The NLCD -> IGBP crosswalk is lossy and judgement-laden.** NLCD has 20
   CONUS classes; IGBP has 17 defined on different axes. Several mappings are
   genuinely ambiguous, and one of them lands squarely on the fuel type that
   matters most for this study (see CROSSWALK_NOTES). Every ambiguous choice is
   recorded so it can be reported rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presets as presets_mod   # noqa: E402
import wsts_spec                # noqa: E402

try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:
    pass

TARGET_RES_M = 375.0
TARGET_SIZE = 128
DST_CRS = "EPSG:5070"

S3_BUCKET = "ignisai-static-chrisjuarez-2026"
S3_DEM_KEY = "source-data/topography/srtm_dem_western_conus_500m.tif"
S3_NLCD_KEY = "source-data/nlcd/nlcd_lndcov_western_conus_2024_500m.tif"

#: Repo-relative fallbacks (data/source-rasters/... is gitignored but may be present).
LOCAL_DEM = Path("data/source-rasters/topography/srtm_dem_western_conus_500m.tif")
LOCAL_NLCD = Path("data/source-rasters/nlcd/nlcd_lndcov_western_conus_2024_500m.tif")


# ---------------------------------------------------------------------------
# NLCD -> MODIS IGBP crosswalk
# ---------------------------------------------------------------------------
#: NLCD 2019/2021 Legend -> IGBP class index (1..17), matching
#: wsts_spec.LANDCOVER_CLASSES order (which is 1-indexed on disk; the WSTS
#: dataloader subtracts 1 before one-hot encoding).
NLCD_TO_IGBP: Dict[int, int] = {
    11: 17,   # Open Water                     -> Water Bodies
    12: 15,   # Perennial Ice/Snow             -> Permanent Snow and Ice
    21: 13,   # Developed, Open Space          -> Urban and Built-up
    22: 13,   # Developed, Low Intensity       -> Urban and Built-up
    23: 13,   # Developed, Medium Intensity    -> Urban and Built-up
    24: 13,   # Developed, High Intensity      -> Urban and Built-up
    31: 16,   # Barren Land                    -> Barren
    41: 4,    # Deciduous Forest               -> Deciduous Broadleaf Forests
    42: 1,    # Evergreen Forest               -> Evergreen Needleleaf Forests  [AMBIGUOUS]
    43: 5,    # Mixed Forest                   -> Mixed Forests
    51: 7,    # Dwarf Scrub (AK)               -> Open Shrublands
    52: 7,    # Shrub/Scrub                    -> Open Shrublands              [AMBIGUOUS]
    71: 10,   # Grassland/Herbaceous           -> Grasslands
    72: 10,   # Sedge/Herbaceous (AK)          -> Grasslands
    73: 10,   # Lichens (AK)                   -> Grasslands
    74: 10,   # Moss (AK)                      -> Grasslands
    81: 14,   # Pasture/Hay                    -> Cropland/Natural Veg Mosaics
    82: 12,   # Cultivated Crops               -> Croplands
    90: 11,   # Woody Wetlands                 -> Permanent Wetlands
    95: 11,   # Emergent Herbaceous Wetlands   -> Permanent Wetlands
}

CROSSWALK_NOTES: Dict[str, str] = {
    "NLCD 42 -> IGBP 1": (
        "NLCD 'Evergreen Forest' does not distinguish needleleaf from broadleaf. "
        "Mapped to Evergreen NEEDLELEAF (1) because western-CONUS evergreen forest "
        "is overwhelmingly conifer. Coastal California evergreen broadleaf (live "
        "oak) is therefore misassigned. Affects Palisades/Eaton more than the "
        "Sierra events."),
    "NLCD 52 -> IGBP 7": (
        "THE MOST CONSEQUENTIAL AMBIGUITY FOR THIS STUDY. IGBP splits shrubland "
        "by canopy cover: Closed (6) >60%, Open (7) <60%. NLCD 'Shrub/Scrub' "
        "carries no density information, so the split is unrecoverable. Mapped to "
        "Open (7) as the areal majority across the western US — but Santa Monica "
        "Mountains chaparral, the dominant fuel at Palisades, is dense and would "
        "properly be Closed (6). This single choice may materially affect the "
        "OOD result and MUST be reported. Sensitivity-test it by re-running with "
        "SHRUB_AS_CLOSED=True."),
    "IGBP 8, 9 unreachable": (
        "Woody Savannas (8) and Savannas (9) have no NLCD equivalent, so those "
        "one-hot channels are always zero in our reconstruction. Native WSTS "
        "data does populate them, so the model sees an input distribution we "
        "cannot reproduce."),
    "IGBP 3 unreachable": (
        "Deciduous Needleleaf Forests (3) is effectively absent from CONUS "
        "(larch); expected-zero, low impact."),
    "resolution": (
        "NLCD source is 500 m here and resampled to 375 m with NEAREST "
        "neighbour (categorical data must never be interpolated)."),
}

#: Flip for the sensitivity test called out above.
SHRUB_AS_CLOSED = False


def crosswalk_nlcd(nlcd: np.ndarray, shrub_as_closed: bool = SHRUB_AS_CLOSED) -> np.ndarray:
    """Map NLCD codes -> IGBP 1..17. Unknown codes -> 16 (Barren) as a neutral default."""
    table = dict(NLCD_TO_IGBP)
    if shrub_as_closed:
        table[52] = 6          # Closed Shrublands
        table[51] = 6
    out = np.full(nlcd.shape, 16, dtype=np.uint8)
    for src, dst in table.items():
        out[nlcd == src] = dst
    return out


# ---------------------------------------------------------------------------
# slope / aspect
# ---------------------------------------------------------------------------
def slope_aspect_degrees(elev: np.ndarray, res_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Slope (deg) and aspect (deg, 0=N clockwise) via central differences.

    Mirrors ignis_ml/src/data/dataset.py::_compute_slope_aspect, but returns
    aspect in DEGREES rather than the (cos, sin) pair — WSTS applies its own
    sin() to this band, so handing it cos/sin would double-transform it.
    """
    dy, dx = np.gradient(elev.astype(np.float64), res_m, res_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    # Downslope direction; y grows south in raster space.
    aspect = np.degrees(np.arctan2(-dx, -dy)) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


# ---------------------------------------------------------------------------
# raster access
# ---------------------------------------------------------------------------
def resolve_source(local: Path, s3_key: str, cache_dir: Path, repo: Path) -> Optional[Path]:
    """Local file if present, else pull from S3 into cache_dir."""
    for cand in (repo / local, local):
        if cand.is_file():
            return cand
    cached = cache_dir / Path(s3_key).name
    if cached.is_file():
        return cached
    try:
        import boto3
    except ImportError:
        print(f"  boto3 unavailable and no local copy of {local.name}")
        print(f"    aws s3 cp s3://{S3_BUCKET}/{s3_key} {cached}")
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  downloading s3://{S3_BUCKET}/{s3_key}")
        boto3.client("s3").download_file(S3_BUCKET, s3_key, str(cached))
        return cached
    except Exception as ex:                                    # noqa: BLE001
        print(f"  S3 fetch failed: {ex}")
        print(f"    aws s3 cp s3://{S3_BUCKET}/{s3_key} {cached}")
        return None


def window_to_tile(src_path: Path, lat: float, lon: float,
                   categorical: bool = False) -> np.ndarray:
    """Warp a CONUS raster onto the preset's 128x128 @375 m EPSG:5070 tile."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    from pyproj import Transformer

    tx = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    cx, cy = tx.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    dst_transform = from_origin(cx - half, cy + half, TARGET_RES_M, TARGET_RES_M)

    with rasterio.open(src_path) as src:
        dtype = np.float32 if not categorical else np.int16
        dst = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=dtype)
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            dst_transform=dst_transform, dst_crs=DST_CRS,
            # Categorical data must NEVER be interpolated.
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
    return dst


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_for_preset(preset, dem: Optional[Path], nlcd: Optional[Path],
                     out_root: Path) -> dict:
    print(f"\n=== {preset.key} ({preset.name}) ===")
    out_dir = out_root / preset.key
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"preset": preset.key}

    if dem:
        elev = window_to_tile(dem, preset.lat, preset.lon)
        slope, aspect = slope_aspect_degrees(elev, TARGET_RES_M)
        for name, arr, band in (("elev", elev, 14), ("slope", slope, 12),
                                ("aspect", aspect, 13)):
            np.save(out_dir / f"{preset.key}_{name}_375m.npy", arr)
            res[name] = {"band": band, "min": float(np.nanmin(arr)),
                         "max": float(np.nanmax(arr))}
            print(f"  band {band:>2} {name:<7} {arr.shape} "
                  f"[{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]")
    else:
        print("  DEM unavailable — bands 12/13/14 skipped")

    if nlcd:
        raw = window_to_tile(nlcd, preset.lat, preset.lon, categorical=True)
        igbp = crosswalk_nlcd(raw)
        np.save(out_dir / f"{preset.key}_landcover_igbp_375m.npy", igbp)
        vals, counts = np.unique(igbp, return_counts=True)
        frac = {int(v): round(float(c) / igbp.size, 4) for v, c in zip(vals, counts)}
        res["landcover"] = {"band": 16, "class_fractions": frac}
        print(f"  band 16 landcover {igbp.shape}  classes present: {sorted(frac)}")
        for v in sorted(frac, key=lambda k: -frac[k])[:4]:
            print(f"          {frac[v]:6.1%}  {v:2d} {wsts_spec.LANDCOVER_CLASSES[v-1]}")
    else:
        print("  NLCD unavailable — band 16 skipped")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build WSTS bands 12/13/16 (+14)")
    ap.add_argument("--preset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shrub-as-closed", action="store_true",
                    help="sensitivity test: map NLCD 52 -> IGBP 6 (Closed Shrublands)")
    args = ap.parse_args(argv)

    global SHRUB_AS_CLOSED
    SHRUB_AS_CLOSED = args.shrub_as_closed

    repo = Path(__file__).resolve().parents[3]
    data_root = Path(os.environ.get("IGNIS_DATA_ROOT", repo / "data"))
    out_root = args.out or data_root / "wsts_inputs" / "static"
    cache = data_root / "wsts_inputs" / "_source_cache"

    print(f"target grid: {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_RES_M:.0f} m, {DST_CRS}")
    print(f"out: {out_root}")
    if SHRUB_AS_CLOSED:
        print("SENSITIVITY MODE: NLCD Shrub/Scrub -> IGBP 6 (Closed Shrublands)")

    print("\nresolving sources...")
    dem = resolve_source(LOCAL_DEM, S3_DEM_KEY, cache, repo)
    nlcd = resolve_source(LOCAL_NLCD, S3_NLCD_KEY, cache, repo)
    print(f"  DEM : {dem or 'MISSING'}")
    print(f"  NLCD: {nlcd or 'MISSING'}")
    if not dem and not nlcd:
        print("\nNeither source available — nothing to do.")
        return 1

    targets = ([presets_mod.by_key(args.preset)] if args.preset
               else list(presets_mod.PRESETS))
    results = [build_for_preset(p, dem, nlcd, out_root) for p in targets]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_static_manifest.json").write_text(json.dumps({
        "target_res_m": TARGET_RES_M, "target_size": TARGET_SIZE, "crs": DST_CRS,
        "shrub_as_closed": SHRUB_AS_CLOSED,
        "nlcd_to_igbp": NLCD_TO_IGBP,
        "crosswalk_notes": CROSSWALK_NOTES,
        "aspect_units": "degrees (WSTS applies sin(deg2rad(x)) downstream)",
        "results": results,
    }, indent=2))

    print("\n=== crosswalk caveats (report these) ===")
    for k, v in CROSSWALK_NOTES.items():
        print(f"  {k}:\n    {v}")
    print(f"\nmanifest -> {out_root / '_static_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
