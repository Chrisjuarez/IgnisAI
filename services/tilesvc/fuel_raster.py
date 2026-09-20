"""Raw FBFM40 fuel codes for a tile.

The model's fuel1/fuel2/fuel3 channels are a latent transform of this raster -
the catalog still marks them "candidate" with parity pending - and a latent
component cannot be used to look up a fuel model's load, depth or moisture of
extinction. Rothermel needs the FBFM40 class itself, so this reads it directly.

Nearest-neighbour only. A fuel model is a category; interpolating between GR2
and TL3 produces a code that names no fuel at all.

A missing raster raises. It used to return barren everywhere on the reasoning
that a missing raster must not invent fire - true, but barren is itself a
claim, and every engine that reads fuel then reported zero spread on every
fire. That is indistinguishable from a real forecast of no growth, and it was
read as one: two of four engines scored zero across five fires and the result
was taken for a property of the engines rather than an absent 8 MB file.
Refusing to answer cannot invent fire either, and it cannot be misread.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from services.runtime_cache.paths import cache_root

from .grid import CRS_ALBERS, SIZE, tile_affine
from .static_catalog import InputUnavailable

#: Where the FBFM40 raster lives. Production reads the same object the
#: fuel1/2/3 channels are derived from; a local path is enough for development.
FUEL_RASTER_ENV = "FBFM40_PATH"
FUEL_RASTER_NAME = "fbfm40_western_conus_2024_500m.tif"

#: FBFM40 reserves 91-99 for non-burnable. Cells outside the raster's coverage
#: are barren rather than flammable; that is a statement about those cells, not
#: about a raster we could not open at all.
NO_DATA_CODE = 99


def candidate_paths() -> List[Path]:
    """Checked in order. The cache root comes first so `fetch-fuel` lands
    somewhere that is found without further configuration."""
    explicit = (os.getenv(FUEL_RASTER_ENV) or "").strip()
    if explicit:
        return [Path(explicit)]
    return [
        cache_root() / "source-rasters" / FUEL_RASTER_NAME,
        Path("data/source-rasters/landfire") / FUEL_RASTER_NAME,
    ]


def fuel_raster_path() -> Optional[Path]:
    for path in candidate_paths():
        if path.is_file():
            return path
    return None


def fuel_codes_for_tile(tile) -> np.ndarray:
    """FBFM40 codes on the tile grid, as int16."""
    path = fuel_raster_path()
    if path is None:
        searched = ", ".join(str(p) for p in candidate_paths())
        raise InputUnavailable(
            "FBFM40 fuel raster not found. Fetch it with "
            "`python -m services.runtime_cache fetch-fuel`, or point "
            f"{FUEL_RASTER_ENV} at a local copy. Searched: {searched}",
            reason="fuel_raster_missing",
            details={"searched": [str(p) for p in candidate_paths()]},
        )

    out = np.full((SIZE, SIZE), NO_DATA_CODE, dtype=np.int16)
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
