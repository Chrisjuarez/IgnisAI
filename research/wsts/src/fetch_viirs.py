#!/usr/bin/env python3
"""Fetch the three VIIRS surface-reflectance bands WSTS needs but IgnisAI lacks.

Per `wsts_spec.SOURCE_MAP`, 16 of the 23 WSTS bands already exist in IgnisAI's
S3 / runtime pipeline and 4 more are derivable. Only three are genuinely
missing, and one derived band is blocked on them:

    band  0 : VIIRS M11  (2.25 um SWIR)   <- VNP09GA 1km grid
    band  1 : VIIRS I2   (0.865 um NIR)   <- VNP09GA 500m grid
    band  2 : VIIRS I1   (0.640 um red)   <- VNP09GA 500m grid
    band  4 : EVI2                        <- derived here from I1, I2

Product: **VNP09GA** (VIIRS/NPP Surface Reflectance Daily L2G Global 1km and
500m SIN Grid), LP DAAC. I1-I3 are delivered at ~500 m and M-bands at ~1 km,
both already resampled by NASA from the native 375 m / 750 m Level-2 data.

----------------------------------------------------------------------------
FIDELITY NOTES — these belong in the paper's limitations, not in a footnote
----------------------------------------------------------------------------
1. **WSTS used Google Earth Engine, not Earthdata.** Their creation code
   (SebastianGer/WildfireSpreadTSCreateDataset) pulls via GEE. GEE serves the
   same VNP09GA product, so the source data are identical, but GEE's internal
   reprojection/compositing may differ in detail from ours. Our reconstruction
   is close, not byte-identical.

2. **Resolution.** VNP09GA ships I-bands at ~463 m and M-bands at ~926 m. WSTS
   works at 375 m. We upsample. No detail is created by doing so; M11 in
   particular is being upsampled ~2.5x.

3. **WIND DIRECTION BUG IN THE PUBLISHED DATASET.** The WSTS creation repo
   README (Feb 2026) records that wind direction was originally computed with
   `atan(v/u)` instead of `atan2(v,u)`, which collapses quadrant information —
   a 45 deg wind and a 225 deg wind became indistinguishable. This does not
   affect the VIIRS bands fetched here, but it bears directly on any
   evaluation of a *wind-driven* fire regime: if `Res18UTAE_T5` was trained on
   pre-fix data, its wind-direction channel carried corrupted input, and poor
   Santa Ana performance may reflect that rather than a genuine
   generalization limit. Determine which data vintage the checkpoint used
   before attributing any directional failure to the model.

Usage
-----
    # what exists, download nothing
    python -m fetch_viirs --dry-run

    # one event
    python -m fetch_viirs --preset palisades --days 5

    # everything
    python -m fetch_viirs --all --days 5

Auth: needs a NASA Earthdata Login. `earthaccess.login(persist=True)` writes
~/.netrc on first run.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Expected and harmless, but emitted once per subdataset open (~18x per event),
# which buries the actual results:
#   * NotGeoreferencedWarning — the HDF5 CONTAINER has no geotransform; the
#     subdatasets inside it do, and those are what we read.
#   * earthaccess FutureWarning — DataGranule.size() deprecation in 0.14.
warnings.filterwarnings("ignore", message=".*not recognized as being in a supported.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="earthaccess.*")
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presets as presets_mod  # noqa: E402

SHORT_NAME = "VNP09GA"
VERSION = "002"

#: v002 HDF-EOS5 subdataset paths, VERIFIED against a real granule
#: (VNP09GA.A2021221.h08v05.002.2022357170552.h5) via `--probe`.
#: Note the grid prefix is VIIRS_Grid_*, not VNP_Grid_* — the product short
#: name and the internal grid name differ. `_find_subdataset` still falls back
#: to leaf matching if a future collection renames the grids.
BAND_SUBDATASETS: Dict[str, str] = {
    "I1":  "VIIRS_Grid_500m_2D/Data_Fields/SurfReflect_I1_1",
    "I2":  "VIIRS_Grid_500m_2D/Data_Fields/SurfReflect_I2_1",
    "M11": "VIIRS_Grid_1km_2D/Data_Fields/SurfReflect_M11_1",
}

#: Quality-flag subdatasets, available but not yet used. VNP09GA ships
#: SurfReflect_QF1..QF7 plus obscov_*; QF2 carries cloud/shadow state. Masking
#: on these would raise the reliability of the reflectance bands, which matters
#: most for the January LA events where marine layer is likely. Not wired in
#: yet — record it as a known quality gap rather than pretending it is done.
QUALITY_SUBDATASETS: Dict[str, str] = {
    "QF1": "VIIRS_Grid_1km_2D/Data_Fields/SurfReflect_QF1_1",
    "QF2": "VIIRS_Grid_1km_2D/Data_Fields/SurfReflect_QF2_1",
    "obscov_500m": "VIIRS_Grid_500m_2D/Data_Fields/obscov_500m_1",
}
#: WSTS base-band index for each fetched band (see wsts_spec.BASE_FEATURES).
BAND_TO_WSTS_INDEX = {"M11": 0, "I2": 1, "I1": 2}

SCALE_FACTOR = 1e-4          # VNP09GA reflectance scaling
FILL_VALUE = -28672          # VNP09GA fill
VALID_RANGE = (-100, 16000)

TILE_KM = 32.0               # match IgnisAI preset footprint
TARGET_RES_M = 375.0         # WSTS native
TARGET_SIZE = 128            # WSTS crop side


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def bbox_for(lat: float, lon: float, tile_km: float = TILE_KM) -> Tuple[float, float, float, float]:
    """(west, south, east, north) around a point, with a margin for reprojection."""
    half = tile_km / 2.0 * 1.5          # 50% margin
    dlat = half / 111.0
    dlon = half / (111.0 * max(np.cos(np.radians(lat)), 0.1))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def date_window(ref_time: str, days: int) -> Tuple[str, str]:
    """`days` of history ending at ref_time (the model's observation window)."""
    ref = datetime.fromisoformat(ref_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    return ((ref - timedelta(days=days)).strftime("%Y-%m-%d"),
            ref.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------------------
# search / download
# ---------------------------------------------------------------------------
def search_granules(lat: float, lon: float, ref_time: str, days: int):
    import earthaccess
    t0, t1 = date_window(ref_time, days)
    return earthaccess.search_data(
        short_name=SHORT_NAME, version=VERSION,
        bounding_box=bbox_for(lat, lon), temporal=(t0, t1),
    )


def is_real_granule(p: Path) -> bool:
    """Reject macOS filesystem cruft that masquerades as a granule.

    Writing to a non-native volume (T9 is exFAT) makes macOS emit AppleDouble
    sidecars named `._<original>`, which carry resource-fork metadata and are
    NOT valid HDF5. They sort BEFORE the real files ('.' < 'V'), so a naive
    `sorted(glob(...))` hands one to rasterio first and every read fails. The
    same sidecars sit next to the model checkpoints on T9.
    """
    n = p.name
    return not (n.startswith("._") or n == ".DS_Store" or n.startswith("."))


def _norm(s: str) -> str:
    """Lowercase, strip separators — so 'Data Fields' == 'Data_Fields' etc.

    HDF-EOS5 grids are conventionally named with a SPACE ('Data Fields'), but
    GDAL builds differ and some releases use an underscore. Matching on a
    normalized string avoids depending on either.
    """
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_subdataset(path: Path, suffix: str) -> Optional[str]:
    """Locate a band's GDAL subdataset URI, tolerant of naming variation."""
    import rasterio
    want = _norm(suffix)
    with rasterio.open(path) as src:
        subs = list(src.subdatasets)
    for sd in subs:                       # exact-ish match first
        if _norm(sd).endswith(want):
            return sd
    for sd in subs:                       # then containment
        if want in _norm(sd):
            return sd
    # last resort: match on the variable name alone (tail of the path)
    leaf = _norm(suffix.split("/")[-1])
    for sd in subs:
        if _norm(sd).endswith(leaf):
            return sd
    return None


def probe_subdatasets(path: Path) -> List[str]:
    """List every subdataset in a granule. Use when band lookup fails."""
    import rasterio
    with rasterio.open(path) as src:
        return list(src.subdatasets)


def read_band(granule_file: Path, band: str) -> Optional[Tuple[np.ndarray, dict]]:
    """Read one band, apply scale + fill masking. Returns (array, profile)."""
    import rasterio
    sd = _find_subdataset(granule_file, BAND_SUBDATASETS[band])
    if sd is None:
        return None
    with rasterio.open(sd) as src:
        raw = src.read(1).astype(np.float32)
        prof = src.profile.copy()
        prof.update(crs=src.crs, transform=src.transform)
    bad = (raw == FILL_VALUE) | (raw < VALID_RANGE[0]) | (raw > VALID_RANGE[1])
    out = raw * SCALE_FACTOR
    out[bad] = np.nan
    return out, prof


def evi2(i1_red: np.ndarray, i2_nir: np.ndarray) -> np.ndarray:
    """Two-band EVI2 (Jiang et al. 2008) — WSTS base band 4.

    EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1)
    """
    denom = i2_nir + 2.4 * i1_red + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.5 * (i2_nir - i1_red) / denom
    return np.clip(np.where(np.isfinite(out), out, np.nan), -1.0, 1.0)


def granule_date(path: Path) -> str:
    """Acquisition date from a VNP09GA filename.

    VNP09GA.A2025002.h08v05.002.2025003140757.h5
            ^^^^^^^^ AYYYYDDD (year + day-of-year)

    Needed because a single event can span several MODIS sinusoidal tiles: the
    Dixie fire returns h08v04 AND h08v05, i.e. 12 granules for 6 days. Grouping
    by date is what keeps those two halves as one mosaicked timestep instead of
    two half-empty ones.
    """
    for part in path.name.split("."):
        if len(part) == 8 and part.startswith("A") and part[1:].isdigit():
            year, doy = int(part[1:5]), int(part[5:8])
            return (datetime(year, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
    return path.stem            # fall back to the filename; keeps sort stable


def mosaic_date(files_for_date: List[Path], band: str,
                lat: float, lon: float) -> Optional[np.ndarray]:
    """Reproject every sinusoidal tile for one date and merge onto the event grid.

    Tiles are spatially disjoint, so merging is 'take the first finite value'.
    Returns None when no tile yielded usable data.
    """
    merged: Optional[np.ndarray] = None
    for f in sorted(files_for_date):
        got = read_band(f, band)
        if got is None:
            continue
        arr, prof = got
        warped = reproject_to_tile(arr, prof, lat, lon)
        if merged is None:
            merged = warped
        else:
            gap = ~np.isfinite(merged)
            merged[gap] = warped[gap]
    return merged


def reproject_to_tile(arr: np.ndarray, src_profile: dict,
                      lat: float, lon: float) -> np.ndarray:
    """Warp a VNP09GA array onto the event's 375 m / 128 px tile."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    from pyproj import Transformer

    dst_crs = "EPSG:5070"
    tx = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    cx, cy = tx.transform(lon, lat)
    half = TARGET_SIZE * TARGET_RES_M / 2.0
    dst_transform = from_origin(cx - half, cy + half, TARGET_RES_M, TARGET_RES_M)

    dst = np.full((TARGET_SIZE, TARGET_SIZE), np.nan, dtype=np.float32)
    reproject(
        source=arr, destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,          # continuous reflectance
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def fetch_preset(preset, out_root: Path, days: int, dry_run: bool) -> dict:
    print(f"\n=== {preset.key} ({preset.name}) — {preset.regime} ===")
    t0, t1 = date_window(preset.ref_time, days)
    w, s, e, n = bbox_for(preset.lat, preset.lon)
    print(f"  window {t0} .. {t1}   bbox ({w:.3f},{s:.3f},{e:.3f},{n:.3f})")

    try:
        granules = search_granules(preset.lat, preset.lon, preset.ref_time, days)
    except ImportError:
        print("  earthaccess not installed:  pip install earthaccess rasterio pyproj")
        return {"preset": preset.key, "status": "no_earthaccess"}
    except Exception as ex:                                  # noqa: BLE001
        print(f"  search failed: {ex}")
        return {"preset": preset.key, "status": "search_failed", "error": str(ex)}

    print(f"  granules found: {len(granules)}")
    if not granules:
        # Suomi-NPP VIIRS starts 2012; Camp (2018) and 2021 events are fine.
        print("  none — check the date window and that VNP09GA v002 covers it")
        return {"preset": preset.key, "status": "no_granules"}

    if dry_run:
        for g in granules[:4]:
            try:
                print(f"    {g.data_links()[0].split('/')[-1]}")
            except Exception:
                print(f"    {g}")
        if len(granules) > 4:
            print(f"    ... +{len(granules)-4} more")
        return {"preset": preset.key, "status": "dry_run", "n_granules": len(granules)}

    import earthaccess
    out_dir = out_root / preset.key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading -> {out_dir}")
    files = [Path(p) for p in earthaccess.download(granules, str(out_dir))]

    # Group by acquisition date FIRST. One date may span several sinusoidal
    # tiles (Dixie: h08v04 + h08v05); those must mosaic into a single timestep.
    by_date: Dict[str, List[Path]] = {}
    n_sidecar = 0
    for f in sorted(files):
        if f.suffix.lower() not in (".h5", ".hdf", ".nc"):
            continue
        if not is_real_granule(f):          # macOS '._' AppleDouble sidecars
            n_sidecar += 1
            continue
        by_date.setdefault(granule_date(f), []).append(f)
    if n_sidecar:
        print(f"  ignored {n_sidecar} macOS '._' sidecar files")

    dates = sorted(by_date)
    multi = {d: len(v) for d, v in by_date.items() if len(v) > 1}
    print(f"  {len(files)} files -> {len(dates)} dates"
          + (f"  (mosaicking multi-tile dates: {multi})" if multi else ""))

    stacks: Dict[str, List[np.ndarray]] = {b: [] for b in BAND_SUBDATASETS}
    used_dates: List[str] = []
    for d in dates:
        layers = {b: mosaic_date(by_date[d], b, preset.lat, preset.lon)
                  for b in BAND_SUBDATASETS}
        if any(v is None for v in layers.values()):
            missing = [b for b, v in layers.items() if v is None]
            print(f"    {d}: skipped — no data for {missing}")
            continue
        for b, v in layers.items():
            stacks[b].append(v)
        used_dates.append(d)

    # "ok" must mean bands were actually extracted — not merely that files
    # downloaded. Reporting ok on an empty extraction is how a broken run gets
    # mistaken for a good one.
    extracted = sum(1 for v in stacks.values() if v)
    result = {"preset": preset.key,
              "status": "ok" if extracted == len(stacks) else
                        ("partial" if extracted else "no_bands_extracted"),
              "n_files": len(files), "dates": used_dates,
              "n_timesteps": len(used_dates), "bands_extracted": extracted}
    if not extracted:
        print("  ⚠ downloaded but extracted NOTHING — subdataset paths are wrong.")
        print("    Run:  python fetch_viirs.py --probe")
    for band, layers in stacks.items():
        if not layers:
            print(f"  {band}: no usable layers"); continue
        cube = np.stack(layers)                        # [T,128,128]
        npy = out_dir / f"{preset.key}_{band}_375m.npy"
        np.save(npy, cube)
        valid = float(np.isfinite(cube).mean())
        print(f"  {band:>4}: {cube.shape}  valid={valid:.1%}  -> {npy.name}")
        result[f"{band}_shape"] = list(cube.shape)
        result[f"{band}_valid_frac"] = valid

    if stacks["I1"] and stacks["I2"]:
        ev = evi2(np.stack(stacks["I1"]), np.stack(stacks["I2"]))
        np.save(out_dir / f"{preset.key}_EVI2_375m.npy", ev)
        print(f"  EVI2: {ev.shape}  valid={np.isfinite(ev).mean():.1%}  (derived)")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch VIIRS M11/I2/I1 for WSTS reconstruction")
    ap.add_argument("--preset", help="single preset key")
    ap.add_argument("--all", action="store_true", help="all presets")
    ap.add_argument("--days", type=int, default=5, help="observation window (WSTS T=5)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="search only, download nothing")
    ap.add_argument("--probe", action="store_true",
                    help="list subdatasets in an already-downloaded granule and exit "
                         "(use when bands report 'no usable layers')")
    args = ap.parse_args(argv)

    if args.probe:
        import os as _os
        root = args.out or Path(
            _os.environ.get("IGNIS_DATA_ROOT", "data")) / "wsts_inputs" / "viirs"
        all_h5 = sorted(root.rglob("*.h5"))
        found = [p for p in all_h5 if is_real_granule(p)]
        skipped = len(all_h5) - len(found)
        if skipped:
            print(f"(ignoring {skipped} macOS AppleDouble '._' sidecars)")
        if not found:
            print(f"No real .h5 granules under {root} — download first.")
            return 1
        g = found[0]
        print(f"probing {g.name}\n  ({g.parent})\n")
        subs = probe_subdatasets(g)
        if not subs:
            print("  NO SUBDATASETS — GDAL lacks HDF5/HDF-EOS5 support in this env.")
            print("  Check:  python -c \"import rasterio; "
                  "print([d for d in rasterio.drivers.raster_driver_extensions()])\"")
            return 1
        print(f"  {len(subs)} subdatasets:")
        for sd in subs:
            print(f"    {sd}")
        print("\n  current expectations:")
        for band, suffix in BAND_SUBDATASETS.items():
            hit = _find_subdataset(g, suffix)
            print(f"    {band:>4} want ...{suffix}")
            print(f"         got  {hit if hit else 'NO MATCH'}")
        return 0

    import os
    out_root = args.out or Path(
        os.environ.get("IGNIS_DATA_ROOT", "data")) / "wsts_inputs" / "viirs"

    if args.preset:
        targets = [presets_mod.by_key(args.preset)]
    elif args.all:
        targets = list(presets_mod.PRESETS)
    else:
        targets = list(presets_mod.PRESETS)
        args.dry_run = True
        print("(no --preset/--all given — defaulting to --dry-run over all presets)")

    if not args.dry_run:
        try:
            import earthaccess
            earthaccess.login(persist=True)
            print("Earthdata login OK")
        except ImportError:
            print("pip install earthaccess rasterio pyproj"); return 1
        except Exception as ex:                              # noqa: BLE001
            print(f"Earthdata login failed: {ex}"); return 1

    print(f"product {SHORT_NAME} v{VERSION}  ->  WSTS bands "
          f"{sorted(BAND_TO_WSTS_INDEX.values())} (+ EVI2 derived, band 4)")
    print(f"target grid: {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_RES_M:.0f} m, EPSG:5070")
    print(f"out: {out_root}")

    results = [fetch_preset(p, out_root, args.days, args.dry_run) for p in targets]

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['preset']:<12} {r['status']}"
              + (f"  granules={r['n_granules']}" if "n_granules" in r else ""))
    if args.dry_run:
        print("\nDry run — nothing downloaded. Re-run with --all to fetch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
