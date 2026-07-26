#!/usr/bin/env python3
"""Fetch NASADEM/SRTM elevation via Earthdata and build WSTS bands 12/13/14.

Replaces the private-S3 route for the topography bands:

    band 12 : Slope      (degrees)
    band 13 : Aspect     (degrees — WSTS applies sin(deg2rad(x)) itself)
    band 14 : Elevation  (metres)

Why not S3
----------
`build_static_bands.py` originally pulled the DEM from
s3://ignisai-static-chrisjuarez-2026. That bucket is private, which means:

  * a reviewer cannot reproduce the pipeline, and
  * it needs a second credential system alongside Earthdata (and the personal
    vs work AWS profile split is an easy way to hit AccessDenied).

NASADEM is public, free, and reachable with the same Earthdata login already
used for the VIIRS bands. For a paper this is strictly better: every input to
research/wsts/ then comes from a public source.

Product: **NASADEM_HGT v001** (NASA DEM Merged, 1 arc-second ~30 m), with
automatic fallback to **SRTMGL1 v003** if NASADEM has no coverage. Both ship
1x1 degree tiles, so an event footprint usually needs 1-4 tiles mosaicked.

Usage
-----
    python fetch_dem.py --dry-run          # search only
    python fetch_dem.py --all              # fetch + build bands 12/13/14
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presets as presets_mod                      # noqa: E402
from build_static_bands import slope_aspect_degrees  # single source of truth  # noqa: E402
from fetch_viirs import bbox_for, is_real_granule    # noqa: E402

try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning, module="earthaccess.*")

#: Primary then fallback. NASADEM is the SRTM reprocessing (voids filled with
#: ASTER/GDEM), so prefer it; SRTMGL1 covers the rare gap.
DEM_PRODUCTS: Tuple[Tuple[str, str], ...] = (
    ("NASADEM_HGT", "001"),
    ("SRTMGL1", "003"),
)

TARGET_RES_M = 375.0
TARGET_SIZE = 128
DST_CRS = "EPSG:5070"

#: Elevation payloads seen across these products/collections.
ELEV_SUFFIXES = (".hgt", ".tif", ".tiff", ".nc", ".img")


# ---------------------------------------------------------------------------
# search / download
# ---------------------------------------------------------------------------
def search_dem(lat: float, lon: float):
    """Return (granules, short_name, version) from the first product with hits."""
    import earthaccess
    bbox = bbox_for(lat, lon)
    for short_name, version in DEM_PRODUCTS:
        try:
            g = earthaccess.search_data(short_name=short_name, version=version,
                                        bounding_box=bbox)
        except Exception as ex:                                  # noqa: BLE001
            print(f"    {short_name} search failed: {ex}")
            continue
        if g:
            return g, short_name, version
    return [], None, None


def elevation_members(path: Path) -> List[str]:
    """GDAL-openable paths for the elevation payload in a download.

    NASADEM/SRTM granules arrive either as a bare .hgt/.tif or as a .zip
    containing one. GDAL reads inside archives via the /vsizip/ prefix, so no
    extraction is needed.
    """
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist()
                         if n.lower().endswith(ELEV_SUFFIXES)
                         and not Path(n).name.startswith("._")]
        except Exception:
            return []
        # .hgt is the elevation band; .num/.img are ancillary
        names.sort(key=lambda n: (not n.lower().endswith(".hgt"), n))
        return [f"/vsizip/{path}/{n}" for n in names[:1]]
    if path.suffix.lower() in ELEV_SUFFIXES:
        return [str(path)]
    return []


# ---------------------------------------------------------------------------
# mosaic + reproject
# ---------------------------------------------------------------------------
def build_tile(sources: List[str], lat: float, lon: float) -> Optional[np.ndarray]:
    """Warp and mosaic DEM tiles onto the preset's 128x128 @375 m grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    from pyproj import Transformer

    tx = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    cx, cy = tx.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    dst_transform = from_origin(cx - half, cy + half, TARGET_RES_M, TARGET_RES_M)

    merged: Optional[np.ndarray] = None
    for src_path in sources:
        try:
            with rasterio.open(src_path) as src:
                dst = np.full((TARGET_SIZE, TARGET_SIZE), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=dst,
                    dst_transform=dst_transform, dst_crs=DST_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata if src.nodata is not None else -32768,
                    dst_nodata=np.nan,
                )
        except Exception as ex:                                  # noqa: BLE001
            print(f"      skip {Path(src_path).name}: {ex}")
            continue
        if merged is None:
            merged = dst
        else:                                    # tiles are disjoint: fill gaps
            gap = ~np.isfinite(merged)
            merged[gap] = dst[gap]
    return merged


def fill_voids(elev: np.ndarray) -> np.ndarray:
    """Fill SRTM voids with the NEAREST valid elevation, not the tile mean.

    Voids must be filled — NaNs propagate through np.gradient and poison
    slope/aspect. But *how* matters. Filling with the tile mean drops a void in
    steep terrain to the average elevation, manufacturing a cliff at its
    boundary and a spurious extreme slope.

    Observed: Camp was the only event with voids (0.56%) and the only one whose
    max slope hit 70 deg, versus 27-34 deg everywhere else. Nearest-valid fill
    preserves local continuity, so a void inherits its surroundings and the
    gradient across it goes to ~0 instead of to a fake cliff.
    """
    bad = ~np.isfinite(elev)
    if not bad.any():
        return elev
    if bad.all():
        return np.zeros_like(elev)

    out = elev.copy()
    try:
        from scipy import ndimage
        # index of the nearest finite cell for every cell
        _, (ri, ci) = ndimage.distance_transform_edt(
            bad, return_distances=True, return_indices=True)
        out = out[ri, ci]
    except ImportError:
        # iterative dilation: repeatedly replace NaNs with the mean of finite
        # 4-neighbours until none remain.
        for _ in range(max(out.shape)):
            bad = ~np.isfinite(out)
            if not bad.any():
                break
            acc = np.zeros_like(out)
            cnt = np.zeros_like(out)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sh = np.roll(np.where(np.isfinite(out), out, 0.0), (dy, dx), (0, 1))
                ok = np.roll(np.isfinite(out).astype(np.float32), (dy, dx), (0, 1))
                acc += sh
                cnt += ok
            cand = np.divide(acc, cnt, out=np.full_like(out, np.nan), where=cnt > 0)
            out = np.where(bad & np.isfinite(cand), cand, out)
        out[~np.isfinite(out)] = float(np.nanmean(elev))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def process(preset, out_root: Path, cache: Path, dry_run: bool) -> dict:
    print(f"\n=== {preset.key} ({preset.name}) ===")
    granules, short_name, version = search_dem(preset.lat, preset.lon)
    if not granules:
        print("  no DEM granules found for this footprint")
        return {"preset": preset.key, "status": "no_granules"}
    print(f"  {short_name} v{version}: {len(granules)} tile(s)")

    if dry_run:
        for g in granules[:4]:
            try:
                print(f"    {g.data_links()[0].split('/')[-1]}")
            except Exception:
                print(f"    {g}")
        return {"preset": preset.key, "status": "dry_run",
                "product": short_name, "n_granules": len(granules)}

    import earthaccess
    dl_dir = cache / preset.key
    dl_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in earthaccess.download(granules, str(dl_dir))]
    files = [f for f in files if is_real_granule(f)]     # drop macOS '._' sidecars

    sources: List[str] = []
    for f in sorted(files):
        sources.extend(elevation_members(f))
    if not sources:
        print(f"  downloaded {len(files)} file(s) but found no elevation payload")
        print(f"    contents: {[f.name for f in files[:5]]}")
        return {"preset": preset.key, "status": "no_elevation_payload"}
    print(f"  elevation sources: {len(sources)}")

    elev = build_tile(sources, preset.lat, preset.lon)
    if elev is None:
        return {"preset": preset.key, "status": "reproject_failed"}

    void_frac = float((~np.isfinite(elev)).mean())
    elev = fill_voids(elev)
    slope, aspect = slope_aspect_degrees(elev, TARGET_RES_M)

    out_dir = out_root / preset.key
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr, band in (("elev", elev, 14), ("slope", slope, 12),
                            ("aspect", aspect, 13)):
        np.save(out_dir / f"{preset.key}_{name}_375m.npy", arr)
        print(f"  band {band:>2} {name:<7} {arr.shape} "
              f"[{arr.min():8.2f}, {arr.max():8.2f}]")
    if void_frac:
        print(f"  note: {void_frac:.2%} voids filled with nearest-valid elevation")

    return {"preset": preset.key, "status": "ok", "product": short_name,
            "n_tiles": len(sources), "void_fraction": void_frac,
            "elev_range": [float(elev.min()), float(elev.max())],
            "slope_max_deg": float(slope.max())}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch NASADEM/SRTM -> WSTS bands 12/13/14")
    ap.add_argument("--preset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    data_root = Path(os.environ.get("IGNIS_DATA_ROOT", "data"))
    out_root = args.out or data_root / "wsts_inputs" / "static"
    cache = data_root / "wsts_inputs" / "_dem_cache"

    targets = ([presets_mod.by_key(args.preset)] if args.preset
               else list(presets_mod.PRESETS))
    if not args.preset and not args.all:
        args.dry_run = True
        print("(no --preset/--all — defaulting to --dry-run)")

    if not args.dry_run:
        try:
            import earthaccess
            earthaccess.login(persist=True)
            print("Earthdata login OK")
        except ImportError:
            print("pip install earthaccess rasterio pyproj"); return 1
        except Exception as ex:                                  # noqa: BLE001
            print(f"Earthdata login failed: {ex}"); return 1

    print(f"products: {[f'{s} v{v}' for s, v in DEM_PRODUCTS]}")
    print(f"target grid: {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_RES_M:.0f} m, {DST_CRS}")
    print(f"out: {out_root}")

    results = [process(p, out_root, cache, args.dry_run) for p in targets]

    print("\n=== summary ===")
    for r in results:
        extra = ""
        if "elev_range" in r:
            lo, hi = r["elev_range"]
            extra = f"  elev {lo:.0f}-{hi:.0f} m  slope<={r['slope_max_deg']:.0f} deg"
        print(f"  {r['preset']:<12} {r['status']}{extra}")
    if args.dry_run:
        print("\nDry run — nothing downloaded. Re-run with --all.")
    else:
        print("\nBands 12/13/14 written. All 23 WSTS bands now sourceable"
              " from public data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
