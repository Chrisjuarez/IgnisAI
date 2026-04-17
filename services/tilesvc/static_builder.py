# services/tilesvc/static_builder.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import rasterio
import requests
from rasterio.warp import reproject, Resampling

from .grid import TileID, SIZE, tile_bounds_lonlat, tile_affine

# Where we store per-tile static npz files inside the container
DEFAULT_CACHE_DIR = os.environ.get("TILESVC_CACHE_DIR", "/app/services/tilesvc/cache")

# DEM source (OpenTopography Global DEM endpoint — free with API key)
OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY")
# safer default DEM; can override via OT_DEM
OPENTOPO_DEM = os.environ.get("OT_DEM", "SRTMGL1")
OPENTOPO_TIMEOUT = int(os.environ.get("OT_TIMEOUT", "120"))
OPENTOPO_BUFFER_DEG = float(os.environ.get("OT_BUFFER_DEG", "0.2"))

# Expected static channel order for the model (length must equal MODEL_CS)
# Match notebook training static_order:
# [elev, slope, aspect_cos, aspect_sin, ndvi, bi, erc, pdsi, chili, impervious, water, population, fuel1, fuel2, fuel3]
CHANNEL_ORDER = [
    "elev",
    "slope",
    "aspect_cos",
    "aspect_sin",
    "ndvi",
    "bi",
    "erc",
    "pdsi",
    "chili",
    "impervious",
    "water",
    "population",
    "fuel1",
    "fuel2",
    "fuel3",
]


def _coerce_tile(tile: Union[TileID, Tuple[int, int], Dict[str, Any]]) -> TileID:
    """
    Accept a TileID, (ix, iy) tuple, or dict-like {'ix':..,'iy':..}
    and return a TileID.
    """
    if isinstance(tile, TileID):
        return tile
    if isinstance(tile, tuple) and len(tile) == 2:
        return TileID(int(tile[0]), int(tile[1]))
    if isinstance(tile, dict) and "ix" in tile and "iy" in tile:
        return TileID(int(tile["ix"]), int(tile["iy"]))
    raise TypeError(f"Unsupported tile type: {type(tile)}")


def _static_npz_path(tile: TileID, cache_dir: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"static_ix{tile.ix}_iy{tile.iy}.npz"


def _fetch_dem_tile(tile: TileID) -> np.ndarray:
    """Download DEM for this tile via OpenTopography and resample to SIZE x SIZE in EPSG:5070."""
    w, s, e, n = tile_bounds_lonlat(tile)
    # Add small buffer to avoid edge gaps
    w -= OPENTOPO_BUFFER_DEG
    s -= OPENTOPO_BUFFER_DEG
    e += OPENTOPO_BUFFER_DEG
    n += OPENTOPO_BUFFER_DEG

    def _download(demtype: str) -> bytes:
        params = {
            "demtype": demtype,
            "south": s, "north": n, "west": w, "east": e,
            "outputFormat": "GTiff",
        }
        if OPENTOPO_API_KEY:
            params["API_Key"] = OPENTOPO_API_KEY
        url = "https://portal.opentopography.org/API/globaldem"
        r = requests.get(url, params=params, timeout=OPENTOPO_TIMEOUT)
        r.raise_for_status()
        return r.content

    content = None
    try:
        content = _download(OPENTOPO_DEM)
    except Exception as first_err:
        # Fallback to SRTMGL1 if custom DEM fails
        if OPENTOPO_DEM != "SRTMGL1":
            try:
                content = _download("SRTMGL1")
            except Exception:
                raise first_err
        else:
            raise first_err

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        with rasterio.open(tmp.name) as src:
            dst = np.zeros((SIZE, SIZE), dtype=np.float32)
            reproject(
                source=src.read(1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tile_affine(tile),
                dst_crs="EPSG:5070",
                resampling=Resampling.bilinear,
            )
    return dst


def _compute_slope_aspect(dem_m: np.ndarray, pix_size_m: float = 500.0):
    """Approximate slope (rad) and aspect (cos/sin) from DEM using central differences."""
    # Pad edges to avoid NaNs at border
    dem = np.pad(dem_m, 1, mode="edge")
    gy, gx = np.gradient(dem, pix_size_m, pix_size_m)
    gy = gy[1:-1, 1:-1]
    gx = gx[1:-1, 1:-1]

    slope = np.arctan(np.hypot(gx, gy))  # radians
    aspect = np.arctan2(-gy, gx)  # GIS-style aspect
    aspect_cos = np.cos(aspect)
    aspect_sin = np.sin(aspect)
    roughness = np.hypot(gx, gy).astype(np.float32)
    return slope.astype(np.float32), aspect_cos.astype(np.float32), aspect_sin.astype(np.float32), roughness


def _build_static_arrays(tile: TileID) -> Dict[str, np.ndarray]:
    """
    Build per-tile static rasters with real elevation/slope/aspect from OpenTopography.
    Remaining channels are filled with reasonable placeholders so we reach 15 channels.
    """
    h = int(SIZE)
    w = int(SIZE)

    try:
        elev = _fetch_dem_tile(tile)
    except Exception as e:
        print(f"⚠️  DEM fetch failed for tile {tile}: {e}. Falling back to zeros.")
        elev = np.zeros((h, w), dtype=np.float32)

    slope, aspect_cos, aspect_sin, _ = _compute_slope_aspect(elev)

    # Placeholders for missing static layers; keep 0..1
    zero = np.zeros((h, w), dtype=np.float32)

    return {
        "elev": elev.astype(np.float32),
        "slope": slope.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        # These are explicit placeholders until production raster sources are wired.
        # Keep both training-case names and legacy lowercase aliases available.
        "fuel1": zero,
        "fuel2": zero,
        "fuel3": zero,
        "NDVI": zero,
        "BI": zero,
        "ERC": zero,
        "PDSI": zero,
        "water": zero,
        "impervious": zero,
        "population": zero,
        "CHILI": zero,
        "ndvi": zero,
        "bi": zero,
        "bli": zero,
        "erc": zero,
        "pdsi": zero,
        "chili": zero,
    }


def build_static_for_tile(
    tile: Union[TileID, Tuple[int, int], Dict[str, Any]],
    cache_dir: str = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> str:
    """
    Build (or load) cached static rasters for a tile.

    Returns:
        str path to the cached .npz file (this is a common pattern in tilesvc apps)
    """
    t = _coerce_tile(tile)
    out = _static_npz_path(t, cache_dir)

    if out.exists() and not force:
        return str(out)

    arrays = _build_static_arrays(t)
    np.savez_compressed(out, **arrays)
    return str(out)


def load_static_for_tile(
    tile: Union[TileID, Tuple[int, int], Dict[str, Any]],
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, np.ndarray]:
    """
    Convenience loader: returns dict of arrays.
    """
    t = _coerce_tile(tile)
    p = Path(build_static_for_tile(t, cache_dir=cache_dir, force=False))
    with np.load(p) as z:
        return {k: z[k] for k in z.files}
