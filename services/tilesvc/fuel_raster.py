"""Raw FBFM40 fuel codes for a tile.

The model's fuel1/fuel2/fuel3 channels are a latent transform of this raster -
the catalog still marks them "candidate" with parity pending - and a latent
component cannot be used to look up a fuel model's load, depth or moisture of
extinction. Rothermel needs the FBFM40 class itself, so this reads it directly.

Nearest-neighbour only. A fuel model is a category; interpolating between GR2
and TL3 produces a code that names no fuel at all.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .grid import CRS_ALBERS, SIZE, tile_affine

#: Where the FBFM40 raster lives. Production reads the same object the
#: fuel1/2/3 channels are derived from; a local path is enough for development.
FUEL_RASTER_ENV = "FBFM40_PATH"
DEFAULT_LOCAL = "data/source-rasters/landfire/fbfm40_western_conus_2024_500m.tif"

#: FBFM40 reserves 99 for barren. An unreadable or out-of-coverage tile is
#: barren rather than flammable, so a missing raster cannot invent fire.
NO_DATA_CODE = 99


def fuel_raster_path() -> Optional[Path]:
    raw = os.getenv(FUEL_RASTER_ENV) or DEFAULT_LOCAL
    path = Path(raw)
    return path if path.is_file() else None


def fuel_codes_for_tile(tile) -> np.ndarray:
    """FBFM40 codes on the tile grid, as int16."""
    path = fuel_raster_path()
    out = np.full((SIZE, SIZE), NO_DATA_CODE, dtype=np.int16)
    if path is None:
        return out

    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tile_affine(tile),
            dst_crs=CRS_ALBERS,
            resampling=Resampling.nearest,
            dst_nodata=NO_DATA_CODE,
        )
    # LANDFIRE writes its own nodata as a negative sentinel.
    out[out <= 0] = NO_DATA_CODE
    return out
