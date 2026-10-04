"""The official burned area, on the model's grid and in its projection.

A perimeter is mapped burned area. The model's fire input is something else:
the cells satellites saw burning in the last day, which is what it was trained
on. Handing it a perimeter as fire would tell it that the whole burned interior
is alight, so the perimeter is used for what it is - fuel that has gone:

- no new fire is forecast in a cell whose centre it covers, and
- the arrival bands show growth beyond it, since everything inside has burned.

Cells the boundary only clips stay open, so the front can still advance
through them.
"""
from typing import Any, Dict, Optional

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

from .grid import SIZE, tile_affine

_WGS84_TO_ALBERS = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform


def _geometries(geojson: Dict[str, Any]):
    kind = geojson.get("type")
    if kind == "FeatureCollection":
        for feature in geojson.get("features") or []:
            yield from _geometries(feature or {})
    elif kind == "Feature":
        if geojson.get("geometry"):
            yield from _geometries(geojson["geometry"])
    elif kind in ("Polygon", "MultiPolygon"):
        yield shape(geojson)


def burned_area_geometry(geojson: Optional[Dict[str, Any]]) -> Optional[BaseGeometry]:
    """A GeoJSON perimeter, or several, as one area in EPSG:5070.

    Accepts a Polygon, MultiPolygon, Feature or FeatureCollection in WGS84.
    Successive perimeters of one fire overlap, so they are merged rather than
    trusted to be the newest alone. Anything that is not a polygon is ignored.
    """
    if not isinstance(geojson, dict):
        return None
    parts = [make_valid(geometry) for geometry in _geometries(geojson) if not geometry.is_empty]
    if not parts:
        return None
    area = unary_union([transform(_WGS84_TO_ALBERS, part) for part in parts])
    return None if area.is_empty or area.area <= 0 else area


def burned_cells(area: Optional[BaseGeometry], tile) -> np.ndarray:
    """Cells of the tile whose centre lies inside the burned area."""
    if area is None:
        return np.zeros((SIZE, SIZE), dtype=bool)
    return rasterize(
        [(area, 1)],
        out_shape=(SIZE, SIZE),
        transform=tile_affine(tile),
        all_touched=False,
        fill=0,
        dtype="uint8",
    ).astype(bool)
