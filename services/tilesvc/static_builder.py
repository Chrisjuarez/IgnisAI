# services/tilesvc/static_builder.py
# Per-tile elevation via OpenTopography → reproject to EPSG:5070 → slope/aspect → cache to S3

import os
import time
import tempfile
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling
import requests

from .grid import SIZE, CRS_ALBERS, tile_affine, tile_bounds_lonlat
from .cache import save_npz_s3, s3_key_static, BUCKET_STATIC


# ---------------- Env / Tunables ----------------
OT_DEM        = os.getenv("OT_DEM", "NASADEM")
OT_KEY        = os.getenv("OPENTOPO_API_KEY", "")
OT_TIMEOUT    = float(os.getenv("OT_TIMEOUT", "180"))
OT_BUFFER_DEG = float(os.getenv("OT_BUFFER_DEG", "0.20"))
ELEVATION_SOURCE = os.getenv("ELEVATION_COG_URL", "opentopo:")


# ---------------- Helpers ----------------
def _download_ot_dem_for_tile(ix: int, iy: int) -> str:
    """Fetch a DEM GeoTIFF for the tile bbox via OpenTopography Global DEM API."""
    tile = type("T", (), {"ix": ix, "iy": iy})()
    w, s, e, n = tile_bounds_lonlat(tile)

    # add small buffer for resampling support at edges
    w -= OT_BUFFER_DEG
    s -= OT_BUFFER_DEG
    e += OT_BUFFER_DEG
    n += OT_BUFFER_DEG

    params = {
        "demtype": OT_DEM,
        "south": s, "north": n, "west": w, "east": e,
        "outputFormat": "GTiff",
    }
    if OT_KEY:
        params["API_Key"] = OT_KEY

    url = "https://portal.opentopography.org/API/globaldem"

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=OT_TIMEOUT)
            if r.status_code == 200 and r.content:
                fd, tmp_path = tempfile.mkstemp(suffix=".tif", prefix=f"ot_dem_{ix}_{iy}_")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                return tmp_path
            last_err = RuntimeError(f"OpenTopography HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))

    raise last_err or RuntimeError("Failed to fetch DEM from OpenTopography")


def _reproject_to_tile(src_path: str, dst_affine) -> np.ndarray:
    """Reproject/resample src raster to the tile grid in EPSG:5070; returns [H,W] float32."""
    with rio.open(src_path) as src:
        dst = np.zeros((SIZE, SIZE), np.float32)
        reproject(
            source=rio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_affine,
            dst_crs=CRS_ALBERS,
            resampling=Resampling.bilinear,
            num_threads=2,
        )

    # Replace non-finite values with median to avoid NaNs in slope/aspect
    med = np.nanmedian(dst)
    if not np.isfinite(med):
        med = 0.0
    dst = np.where(np.isfinite(dst), dst, med).astype(np.float32)
    return dst


def _slope_aspect(elev_m: np.ndarray, pix_m: float):
    """
    Compute slope (deg) and aspect unit vectors from elevation (meters).
    Pixel size is passed in **meters** from the tile affine (robust if SIZE/PIX changes).
    """
    gy, gx = np.gradient(elev_m, pix_m, pix_m)
    slope = np.rad2deg(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    # Downslope azimuth: 0° = north, clockwise positive
    aspect_deg = (np.rad2deg(np.arctan2(-gx, gy)) + 360.0) % 360.0
    asp_cos = np.cos(np.deg2rad(aspect_deg)).astype(np.float32)
    asp_sin = np.sin(np.deg2rad(aspect_deg)).astype(np.float32)
    return slope, asp_cos, asp_sin


def _src_path_for_tile(ix: int, iy: int) -> str:
    """
    If ELEVATION_SOURCE is http(s) (single COG), use it; otherwise fetch per-tile from OT.
    """
    src = (ELEVATION_SOURCE or "").strip()
    if src.lower().startswith("http://") or src.lower().startswith("https://"):
        return src
    return _download_ot_dem_for_tile(ix, iy)


# ---------------- Public API ----------------
def build_static(ix: int, iy: int) -> np.ndarray:
    """
    Build static bands for this tile:
      [elevation, slope_deg, aspect_cos, aspect_sin] with shape [4, SIZE, SIZE]
    Uploads a compressed NPZ to S3 and returns the array.
    """
    A = tile_affine(type("T", (), {"ix": ix, "iy": iy})())
    pix_m = float(abs(A.a))  # pixel size in meters from affine (e.g., 500)

    src_path = _src_path_for_tile(ix, iy)
    tmp_path = None
    try:
        # Remember to unlink if we downloaded an OT temp file
        if not (src_path.lower().startswith("http://") or src_path.lower().startswith("https://")):
            tmp_path = src_path

        elev = _reproject_to_tile(src_path, A)            # [H,W] float32 in 5070
        slope, asp_cos, asp_sin = _slope_aspect(elev, pix_m)
        x_static = np.stack([elev, slope, asp_cos, asp_sin], axis=0)  # [4,H,W]

        # Best-effort S3 write
        try:
            save_npz_s3(BUCKET_STATIC, s3_key_static(ix, iy), x=x_static)
        except Exception as e:
            print(f"[static_builder] S3 save warning: {e}")

        return x_static

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass