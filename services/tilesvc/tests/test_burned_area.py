"""The official burned area, as a forecast uses it.

A perimeter is fuel that has gone. These pin down the two ways the forecast
leans on that - no new fire inside it, bands only beyond it - and the cell rule
that keeps the front free to advance through cells the boundary only clips.
"""
import numpy as np
import pytest
from shapely.geometry import box

from services.tilesvc import spread_bands as sb
from services.tilesvc.burned_area import burned_area_geometry, burned_cells
from services.tilesvc.dynamic_builder import _detected_near
from services.tilesvc.grid import PIX, SIZE, lonlat_to_tile, lonlat_to_xy_m, tile_affine

LON, LAT = -118.40183, 34.561835
TILE = lonlat_to_tile(LON, LAT)
IDENTITY = lambda x, y: (x, y)  # noqa: E731 - keeps band geometry in projected metres


def _square(lon, lat, half_deg):
    return {
        "type": "Polygon",
        "coordinates": [[[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
                         [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
                         [lon - half_deg, lat - half_deg]]],
    }


def _cells_box(row0, row1, col0, col1):
    """The tile-CRS rectangle exactly covering rows row0..row1-1, cols col0..col1-1."""
    affine = tile_affine(TILE)
    x0, y0 = affine * (col0, row0)
    x1, y1 = affine * (col1, row1)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class TestBurnedAreaGeometry:
    def test_reads_a_polygon_feature_or_collection_alike(self):
        polygon = _square(LON, LAT, 0.01)
        as_polygon = burned_area_geometry(polygon)
        as_feature = burned_area_geometry({"type": "Feature", "properties": {}, "geometry": polygon})
        as_collection = burned_area_geometry({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": polygon}]})

        assert as_polygon.area == pytest.approx(as_feature.area) == pytest.approx(as_collection.area)
        # A 0.02-degree square at this latitude is about 2.2 by 1.8 km.
        assert 3.0e6 < as_polygon.area < 5.0e6

    def test_merges_successive_perimeters_instead_of_double_counting(self):
        earlier, later = _square(LON, LAT, 0.01), _square(LON + 0.005, LAT, 0.01)
        merged = burned_area_geometry({"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": earlier}, {"type": "Feature", "geometry": later}]})

        single = burned_area_geometry(earlier)
        assert single.area < merged.area < 2 * single.area

    @pytest.mark.parametrize("value", [
        None, {}, "not geojson",
        {"type": "FeatureCollection", "features": []},
        {"type": "Point", "coordinates": [LON, LAT]},
        {"type": "Feature", "geometry": None},
    ])
    def test_anything_without_a_polygon_is_no_burned_area(self, value):
        assert burned_area_geometry(value) is None


class TestBurnedCells:
    def test_marks_the_cells_the_area_covers(self):
        cells = burned_cells(_cells_box(10, 14, 20, 24), TILE)

        assert cells.shape == (SIZE, SIZE)
        assert cells[10:14, 20:24].all()
        assert cells.sum() == 16

    def test_a_cell_the_boundary_only_clips_stays_open_for_the_front(self):
        # The area reaches 30% of the way into column 24 - short of its centres.
        inside = _cells_box(10, 14, 20, 24)
        x_edge = inside.bounds[2]
        sliver = box(x_edge, inside.bounds[1], x_edge + PIX * 0.3, inside.bounds[3])
        clipped = inside.union(sliver)

        cells = burned_cells(clipped, TILE)

        assert not cells[10:14, 24].any()

    def test_no_area_marks_nothing(self):
        assert not burned_cells(None, TILE).any()


class TestBandsBeyondTheBurnedArea:
    @staticmethod
    def _rollout():
        filled = np.zeros((SIZE, SIZE), dtype=bool)
        filled[16:48, 16:48] = True
        return [{"prob": np.where(filled, 0.9, 0.0).astype(np.float32), "lead_hours": 24, "label": "1 day"}]

    def test_no_band_is_drawn_over_ground_that_has_burned(self):
        burned = _cells_box(16, 48, 16, 32)

        clipped = sb.spread_bands(self._rollout(), TILE, IDENTITY, burned_area=burned)

        from shapely.geometry import shape
        drawn = [shape(f["geometry"]) for f in clipped["features"]]
        assert drawn, "growth beyond the burned area must still be drawn"
        assert sum(g.intersection(burned).area for g in drawn) == pytest.approx(0, abs=1.0)

    def test_without_a_burned_area_the_bands_are_unchanged(self):
        assert sb.spread_bands(self._rollout(), TILE, IDENTITY) == \
            sb.spread_bands(self._rollout(), TILE, IDENTITY, burned_area=None)


class TestIgnitionSeed:
    def _fire_at(self, row_offset):
        x, y = lonlat_to_xy_m(LON, LAT)
        col, row = ~tile_affine(TILE) * (x, y)
        fire = np.zeros((SIZE, SIZE), dtype=np.float32)
        fire[int(row) + row_offset, int(col)] = 1.0
        return fire

    def test_a_detection_nearby_means_the_fire_is_already_in_the_channel(self):
        assert _detected_near(self._fire_at(3), LAT, LON, tile_affine(TILE), 2000.0)

    def test_a_detection_far_off_does_not(self):
        assert not _detected_near(self._fire_at(6), LAT, LON, tile_affine(TILE), 2000.0)

    def test_no_detections_at_all_is_a_new_fire(self):
        assert not _detected_near(np.zeros((SIZE, SIZE), dtype=np.float32), LAT, LON, tile_affine(TILE), 2000.0)
