"""Serving windows must be centred on the fire.

Training tiles come from NDWS samples built around the fire and centre-cropped,
so the fire is always at the middle cell. Serving used to snap to a fixed 32 km
lattice and take whichever cell the fire fell in, which put 9 of 10 real fires
within 6 km of an edge and asked the model to spread fire into cells that were
not in its input. Palisades landed at column 2 with a west-southwest wind.
"""
import numpy as np
import pytest

from services.tilesvc.grid import (
    SIZE,
    TILE_M,
    PIX,
    TileID,
    lonlat_to_tile,
    lonlat_to_xy_m,
    snapped_tile,
    tile_affine,
    tile_bounds_albers,
)

# Two presets and a spread of live-fire locations that previously cornered.
FIRES = [
    ("palisades", -118.5550, 34.0780),
    ("eaton", -118.1000, 34.1900),
    ("rowe creek", -119.9000, 44.7000),
    ("coleman creek", -122.8000, 42.4000),
    ("sinlahekin", -119.6000, 48.7000),
    ("tartar", -110.9000, 38.4000),
]


def fire_cell(lon: float, lat: float):
    """Where the fire lands inside its own prediction window, in (col, row)."""
    x, y = lonlat_to_xy_m(lon, lat)
    col, row = ~tile_affine(lonlat_to_tile(lon, lat)) * (x, y)
    return col, row


@pytest.mark.parametrize("name,lon,lat", FIRES)
def test_fire_lands_at_the_centre_cell(name, lon, lat):
    col, row = fire_cell(lon, lat)

    # Within half a pixel of centre: the origin is snapped to the 500 m lattice
    # so the raster stays aligned, which costs at most half a cell of centring.
    assert col == pytest.approx(SIZE / 2, abs=0.5), f"{name} is off-centre in column"
    assert row == pytest.approx(SIZE / 2, abs=0.5), f"{name} is off-centre in row"


@pytest.mark.parametrize("name,lon,lat", FIRES)
def test_every_direction_has_room_to_spread(name, lon, lat):
    col, row = fire_cell(lon, lat)
    room = min(col, SIZE - col, row, SIZE - row)

    # The failure this fix exists for: Palisades had 2 cells of room westward
    # while the wind blew west-southwest.
    assert room > SIZE / 2 - 1, f"{name} has only {room:.1f} cells of room"


def test_window_is_aligned_to_the_pixel_lattice():
    # Statics reproject into this affine; an unaligned origin would resample
    # every request onto a slightly different grid.
    tile = lonlat_to_tile(-118.5550, 34.0780)

    assert tile.x0 % PIX == 0
    assert tile.y0 % PIX == 0


def test_window_covers_exactly_one_tile():
    tile = lonlat_to_tile(-118.5550, 34.0780)
    minx, miny, maxx, maxy = tile_bounds_albers(tile)

    assert maxx - minx == pytest.approx(TILE_M)
    assert maxy - miny == pytest.approx(TILE_M)
    assert tile_affine(tile) * (SIZE, SIZE) == pytest.approx((maxx, miny))


def test_snapped_tile_still_addresses_the_fixed_lattice():
    # Pre-generated statics and the TS-SatFire ingest address the grid, not a
    # fire, and neighbour walks are only meaningful there.
    tile = snapped_tile(-118.5550, 34.0780)

    assert tile.x0 % TILE_M == 0
    assert tile.y0 % TILE_M == 0
    assert TileID.from_grid(tile.ix, tile.iy) == tile


def test_distinct_windows_in_one_grid_cell_get_distinct_cache_keys():
    # Two fires 10 km apart share an ix/iy but need different statics; keying
    # the cache on ix/iy would serve one fire's terrain to the other.
    a = lonlat_to_tile(-118.5550, 34.0780)
    b = lonlat_to_tile(-118.4500, 34.0780)

    assert (a.ix, a.iy) == (b.ix, b.iy), "precondition: same grid cell"
    assert a.key != b.key
