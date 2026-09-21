"""
Bands are drawn as adjacent filled polygons, so the geometry has two jobs at
once: read as a fire perimeter rather than a pixel staircase, and tile without
gaps. Smoothing each band on its own satisfies the first and breaks the second.
"""

import math

import numpy as np
import pytest
from shapely.geometry import shape
from shapely.ops import unary_union

from services.tilesvc import spread_bands as sb
from services.tilesvc.grid import SIZE, lonlat_to_tile

IDENTITY = lambda x, y: (x, y)  # noqa: E731 - tile CRS in, tile CRS out
TILE = lonlat_to_tile(-118.555, 34.078)


def _front(step):
    """A lopsided spreading front - a circle hides staircase artefacts."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    distance = np.hypot(xx - 30 - step * 2.0, (yy - 32) * 1.5)
    return np.clip(1.4 - distance / (7 + step * 3.0), 0.0, 1.0).astype(np.float32)


def _rollout(days=5):
    return [{"prob": _front(i), "lead_hours": (i + 1) * 24, "label": f"{i + 1} day"}
            for i in range(days)]


def _by_day(collection):
    grouped = {}
    for feature in collection["features"]:
        grouped.setdefault(feature["properties"]["day"], []).append(shape(feature["geometry"]))
    return {day: unary_union(parts).buffer(0) for day, parts in sorted(grouped.items())}


def _corner_angles(geometry):
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    angles = []
    for polygon in polygons:
        ring = list(polygon.exterior.coords)
        for before, here, after in zip(ring, ring[1:], ring[2:]):
            a = np.subtract(before, here)
            b = np.subtract(after, here)
            if np.linalg.norm(a) and np.linalg.norm(b):
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                angles.append(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))))
    return angles


@pytest.fixture(scope="module")
def bands():
    return _by_day(sb.spread_bands(_rollout(), TILE, IDENTITY))


def test_the_pixel_staircase_is_gone(bands):
    """A rasterised outline is all 90-degree turns. simplify() removes
    vertices but leaves the corners square, which is why this needed Chaikin
    and not a bigger tolerance."""
    for day, geometry in bands.items():
        square = [a for a in _corner_angles(geometry) if abs(a - 90.0) < 6.0]
        assert len(square) <= 2, f"day {day} still has {len(square)} right-angle corners"


def test_adjacent_bands_share_their_edge_exactly(bands):
    """The reason smoothing happens on the cumulative outline. Smooth each
    band separately and the two sides of a shared edge drift apart, leaving
    hairline gaps that show as seams against the basemap."""
    days = sorted(bands)
    for earlier, later in zip(days, days[1:]):
        assert bands[earlier].distance(bands[later]) == pytest.approx(0.0, abs=1e-6)
        assert bands[earlier].intersection(bands[later]).area == pytest.approx(0.0, abs=1.0)


def test_bands_stay_disjoint_after_smoothing(bands):
    total = sum(geometry.area for geometry in bands.values())
    union = unary_union(list(bands.values())).area
    assert total == pytest.approx(union, rel=1e-6)


def test_bands_stay_nested_so_no_day_punches_through_an_earlier_one(bands):
    """Chaikin perturbs each outline independently, so without forcing the
    cumulative union a later day can fall inside an earlier one and difference
    into a ring with a hole in the wrong place."""
    days = sorted(bands)
    running = None
    for day in days:
        running = bands[day] if running is None else unary_union([running, bands[day]])
        assert running.is_valid, f"cumulative shape through day {day} is invalid"


def test_every_geometry_is_valid_and_closed(bands):
    for day, geometry in bands.items():
        assert geometry.is_valid, f"day {day} geometry is invalid"
        assert not geometry.is_empty


def test_smoothing_does_not_move_the_perimeter_far(bands):
    """Chaikin cuts corners inward. Over 500 m cells that must stay well under
    a cell, or the map is no longer showing where the model put the fire."""
    raw = sb.cumulative_masks(_rollout(), sb.DEFAULT_BAND_THRESHOLD)
    unsmoothed = sb._mask_to_polygon(raw[-1], TILE)
    smoothed = unary_union(list(bands.values()))
    drift = smoothed.symmetric_difference(unsmoothed).area / max(unsmoothed.area, 1.0)
    assert drift < 0.05, f"smoothing moved {drift:.1%} of the burned area"


def test_a_single_cell_is_dropped_rather_than_smoothed_into_a_blob():
    """One pixel is model noise. Rounding it produces a confident little
    circle, which is more authority than one cell has earned."""
    speck = np.zeros((SIZE, SIZE), np.float32)
    speck[10, 10] = 1.0
    collection = sb.spread_bands([{"prob": speck, "lead_hours": 24, "label": "1 day"}],
                                 TILE, IDENTITY)
    assert collection["features"] == []


def test_chaikin_keeps_a_ring_closed():
    square = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    rounded = sb._chaikin(square, 2)
    assert rounded[0] == rounded[-1]
    assert len(rounded) > len(square)


def test_chaikin_leaves_a_degenerate_ring_alone():
    assert sb._chaikin([(0, 0), (1, 1), (0, 0)], 2) == [(0, 0), (1, 1), (0, 0)]


def _pinched():
    """Two lobes joined by a one-cell isthmus.

    This is the shape that broke it in production and that the smooth blob
    above never produced: Chaikin cuts the corners on both sides of a
    single-cell neck until the ring crosses itself.
    """
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[20:28, 12:22] = 1.0
    mask[20:28, 34:44] = 1.0
    mask[23:25, 22:34] = 1.0          # the isthmus
    mask[26, 21] = mask[26, 34] = 1.0  # single-cell spurs off the lobes
    return mask


def test_a_pinched_front_does_not_emit_self_intersecting_rings():
    rollout = [
        {"prob": _pinched(), "lead_hours": 24, "label": "1 day"},
        {"prob": np.clip(_pinched() + _front(2), 0, 1), "lead_hours": 48, "label": "2 days"},
        {"prob": np.clip(_pinched() + _front(4), 0, 1), "lead_hours": 72, "label": "3 days"},
    ]
    collection = sb.spread_bands(rollout, TILE, IDENTITY)
    assert collection["features"], "a pinched front should still produce bands"
    for feature in collection["features"]:
        geometry = shape(feature["geometry"])
        assert geometry.is_valid, (
            f"day {feature['properties']['day']} is invalid: "
            f"{__import__('shapely.validation', fromlist=['x']).explain_validity(geometry)}"
        )
        assert geometry.geom_type == "Polygon"


def test_repair_rescues_a_self_intersecting_ring():
    from shapely.geometry import Polygon
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    assert not bowtie.is_valid
    repaired = sb._repair(bowtie)
    assert repaired is not None and repaired.is_valid


def test_repair_reports_failure_rather_than_returning_rubbish():
    from shapely.geometry import Polygon
    assert sb._repair(None) is None
    assert sb._repair(Polygon()) is None
