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
    """A SIZE x SIZE window in EPSG:5070, named by its upper-left corner.

    The origin is metres, not a grid index, because serving windows are centred
    on the fire and so do not generally start on a TILE_M boundary. ix/iy are
    derived and only identify a window that happens to be grid-aligned; they
    exist for the pre-generated static tiles and the TS-SatFire ingest, which
    both address the fixed grid.
    """

    x0: float   # easting at the upper-left corner
    y0: float   # northing at the upper-left corner

    @classmethod
    def from_grid(cls, ix: int, iy: int) -> "TileID":
        return cls(float(ix) * TILE_M, float(iy + 1) * TILE_M)

    @property
    def ix(self) -> int:
        return int(np.floor(self.x0 / TILE_M))

    @property
    def iy(self) -> int:
        return int(np.floor(self.y0 / TILE_M)) - 1

    @property
    def key(self) -> str:
        """Stable identity for caching, in whole pixels from the CRS origin."""
        return f"x{int(round(self.x0 / PIX))}_y{int(round(self.y0 / PIX))}"


def lonlat_to_xy_m(lon: float, lat: float):
    """WGS84 lon/lat → Albers meters (x,y)."""
    x, y = _to_albers.transform(lon, lat)
    return float(x), float(y)


def lonlat_to_tile(lon: float, lat: float) -> TileID:
    """The prediction window CENTRED on a point, snapped to the pixel lattice.

    Centring is what makes serving match training. Training tiles come from
    NDWS samples that are built around the fire and then centre-cropped, so
    every one has the fire at the middle cell. Serving used to snap to a fixed
    TILE_M grid and take whichever cell the fire fell in, which put 9 of 10
    real fires within 6 km of an edge - Palisades landed at column 2 with one
    kilometre of room to the west while the Santa Ana blew west-southwest. The
    model was being asked to spread fire into cells that were not in its input.

    Snapping the origin to whole pixels keeps the raster aligned to a single
    500 m lattice, so statics resample consistently and nearby requests share
    cache entries. The fire lands within half a pixel of the centre.
    """
    x, y = lonlat_to_xy_m(lon, lat)
    x0 = round((x - TILE_M / 2.0) / PIX) * PIX
    y0 = round((y + TILE_M / 2.0) / PIX) * PIX
    return TileID(float(x0), float(y0))


def snapped_tile(lon: float, lat: float) -> TileID:
    """The grid-aligned tile containing a point.

    For the fixed-grid addressing that pre-generated static tiles and the
    TS-SatFire ingest use. Prediction should use lonlat_to_tile.
    """
    x, y = lonlat_to_xy_m(lon, lat)
    return TileID.from_grid(int(np.floor(x / TILE_M)), int(np.floor(y / TILE_M)))


def tile_affine(tile: TileID) -> Affine:
    """Affine transform for pixel→map (EPSG:5070) for this tile (upper-left origin)."""
    return Affine(PIX, 0.0, tile.x0, 0.0, -PIX, tile.y0)


def tile_bounds_albers(tile: TileID):
    """Return (minx, miny, maxx, maxy) in EPSG:5070 meters for the tile."""
    return tile.x0, tile.y0 - TILE_M, tile.x0 + TILE_M, tile.y0


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
