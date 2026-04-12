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
# Match training config: 32 km tiles at 500 m -> 64x64
TILE_M = 32_000    # tile length in meters
SIZE = TILE_M // PIX  # pixels per side -> 64

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


def tile_coordinates_lonlat(tile: TileID):
    """Return tile image coordinates as [[NW], [NE], [SE], [SW]] in lon/lat."""
    minx, miny, maxx, maxy = tile_bounds_albers(tile)
    corners = [
        _to_wgs84.transform(minx, maxy),
        _to_wgs84.transform(maxx, maxy),
        _to_wgs84.transform(maxx, miny),
        _to_wgs84.transform(minx, miny),
    ]
    return [[float(lon), float(lat)] for lon, lat in corners]


def tile_bounds_lonlat(tile: TileID):
    """Return (W,S,E,N) in lon/lat for the tile."""
    coords = tile_coordinates_lonlat(tile)
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    return min(lons), min(lats), max(lons), max(lats)


# -------------------------------------------------------------------
# Compatibility helpers expected by services/tilesvc/app.py
# (Your app imports: build_grid, raster_to_png_bytes, and sometimes tile_bounds)
# -------------------------------------------------------------------

def tile_bounds(lat: float, lon: float):
    """
    API helper: given lat/lon, return the containing tile bounds in lon/lat.
    """
    tile = lonlat_to_tile(lon, lat)
    return tile_bounds_lonlat(tile)


def build_grid(lat: float, lon: float):
    """
    API helper: build basic grid metadata for a query point.
    Returns: (tile_id, affine, bounds_lonlat)
    """
    tile = lonlat_to_tile(lon, lat)
    aff = tile_affine(tile)
    bounds = tile_bounds_lonlat(tile)
    return tile, aff, bounds


def raster_to_png_bytes(arr: np.ndarray):
    """
    Convert a 2D float/bool mask to PNG bytes (grayscale).
    - bool -> 0/255
    - float -> clipped [0,1] then scaled to [0,255]
    """
    from PIL import Image  # Pillow
    import io

    a = np.asarray(arr)

    if a.dtype == np.bool_:
        img = (a.astype(np.uint8) * 255)
    else:
        a = a.astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, 0.0, 1.0)
        img = (a * 255.0).astype(np.uint8)

    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
