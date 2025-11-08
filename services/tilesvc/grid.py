# services/tilesvc/grid.py
from dataclasses import dataclass
from affine import Affine
from pyproj import Transformer
import numpy as np

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"   # lon/lat
CRS_ALBERS = "EPSG:5070"  # US Albers Equal Area (meters)

# Tile/grid spec
PIX = 500          # meters per pixel
TILE_M = 64_000    # tile length in meters
SIZE = TILE_M // PIX  # pixels per side -> 128

# Transformers
_to_albers = Transformer.from_crs(CRS_WGS84, CRS_ALBERS, always_xy=True)
_to_wgs84  = Transformer.from_crs(CRS_ALBERS, CRS_WGS84, always_xy=True)


@dataclass(frozen=True)
class TileID:
    ix: int
    iy: int


def lonlat_to_xy_m(lon: float, lat: float):
    """WGS84 lon/lat → Albers meters (x,y)."""
    x, y = _to_albers.transform(lon, lat)
    return float(x), float(y)


def lonlat_to_tile(lon: float, lat: float) -> TileID:
    """Map a lon/lat point to the containing 64 km tile index in EPSG:5070 meters."""
    x, y = lonlat_to_xy_m(lon, lat)
    ix = int(np.floor(x / TILE_M))
    iy = int(np.floor(y / TILE_M))
    return TileID(ix, iy)


def tile_affine(tile: TileID) -> Affine:
    """Affine transform for pixel→map (EPSG:5070) for this tile (upper-left origin)."""
    x0 = tile.ix * TILE_M
    y0 = (tile.iy + 1) * TILE_M  # northing at upper-left
    return Affine(PIX, 0.0, x0, 0.0, -PIX, y0)


def tile_bounds_albers(tile: TileID):
    """Return (minx, miny, maxx, maxy) in EPSG:5070 meters for the tile."""
    x0 = tile.ix * TILE_M
    y1 = (tile.iy + 1) * TILE_M
    x1 = x0 + TILE_M
    y0 = y1 - TILE_M
    return x0, y0, x1, y1


def tile_bounds_lonlat(tile: TileID):
    """Return (W,S,E,N) in lon/lat for the tile."""
    minx, miny, maxx, maxy = tile_bounds_albers(tile)
    (w, s) = _to_wgs84.transform(minx, miny)
    (e, n) = _to_wgs84.transform(maxx, maxy)
    return float(w), float(s), float(e), float(n)