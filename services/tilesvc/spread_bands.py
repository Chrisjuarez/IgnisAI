"""Fire spread as dated, nested polygons rather than a heat blur.

A per-day heatmap answers "how hot is this cell today", which is not the
question anyone looking at a fire map is asking. They want the shape: how far
has it come, and how far might it get. That is what the published fire
progression maps show - NASA's Palisades reconstruction, and the NWCG
progression standard - and it is what this produces.

Two conventions this follows, and one it deliberately breaks.

Follows: bands are CUMULATIVE and nested, so day 3 contains days 1 and 2. The
model emits a delta, so a naive per-day band makes a fire appear to shrink once
a cell has burned. And each band gets a crisp outline - the "isochron" that
Copernicus GC 8:167 (2025) recommends, because it keeps bands separable when
the colours themselves get hard to tell apart.

Breaks: NWCG puts the WARMEST colour on the most recent perimeter, which is
right for a retrospective map where the newest edge is the live one. This is a
forecast, and later days carry less confidence, so shouting loudest about day 6
would be backwards. Day 1 is the most saturated and confidence fades outward.
The palette is ColorBrewer YlOrRd reversed, which that same paper endorses for
fire over rainbow schemes like Turbo: it survives deuteranopia and protanopia
and still reads in greyscale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rasterio.features import shapes as raster_shapes
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

from .grid import PIX, tile_affine

#: ColorBrewer YlOrRd, reversed: day 1 deepest, later days fading out.
#: Sequential, colour-vision-deficient safe, and legible in greyscale.
BAND_COLORS = ("#bd0026", "#f03b20", "#fd8d3c", "#feb24c", "#fed976", "#ffffb2")

#: Below this calibrated probability a cell is not part of the burn shape.
DEFAULT_BAND_THRESHOLD = 0.10

#: Chaikin passes applied to each band outline. Two rounds the 500 m staircase
#: into something that reads as a fire perimeter; a third only adds vertices.
SMOOTHING_PASSES = 2

#: Drop specks smaller than this. A lone pixel is model noise, not a fire front,
#: and drawing it as a polygon gives it more authority than it has earned.
MIN_BAND_AREA_M2 = 4 * PIX * PIX


def band_color(day: int) -> str:
    return BAND_COLORS[min(max(day, 1), len(BAND_COLORS)) - 1]


def cumulative_masks(rollout: Sequence[Dict[str, Any]], threshold: float) -> List[np.ndarray]:
    """One boolean mask per day, each containing every earlier day.

    Running maximum, matching the AR feedback the rollout carries between
    steps, so the bands never contradict the model's own notion of burned.
    """
    masks: List[np.ndarray] = []
    running: Optional[np.ndarray] = None
    for step in rollout:
        prob = np.asarray(step["prob"], dtype=np.float32)
        running = prob if running is None else np.maximum(running, prob)
        masks.append(running >= threshold)
    return masks


def arrival_bands(masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Split nested masks into disjoint bands by first-arrival day.

    Cumulative masks stack: day 3 covers day 1, so drawing them as translucent
    layers muddies every colour where they overlap. A progression map instead
    gives each cell to the day it FIRST burned, which is both what the
    published maps show and what lets each band keep its own colour. The union
    of bands 1..N is still exactly cumulative mask N, so the semantics do not
    change - only which pixels each band is responsible for drawing.
    """
    bands: List[np.ndarray] = []
    claimed: Optional[np.ndarray] = None
    for mask in masks:
        band = mask if claimed is None else (mask & ~claimed)
        claimed = mask if claimed is None else (claimed | mask)
        bands.append(band)
    return bands


def _chaikin(ring: Sequence[Sequence[float]], iterations: int) -> List[tuple]:
    """Chaikin corner-cutting on a closed ring.

    simplify() removes vertices but leaves every corner square, so a
    rasterised edge stays a staircase with fewer steps. Chaikin replaces each
    corner with two points a quarter in from either side, which converges on a
    quadratic B-spline - the corners round off and the curve stays inside the
    original hull. Two passes is the useful range at 500 m: one still reads as
    angular, three costs vertices without a visible gain.
    """
    points = [tuple(p) for p in ring]
    if len(points) < 4:
        return points
    if points[0] != points[-1]:
        points.append(points[0])

    for _ in range(iterations):
        cut: List[tuple] = []
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            cut.append((0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by))
            cut.append((0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by))
        cut.append(cut[0])
        points = cut
    return points


def _smooth(polygon, iterations: int = SMOOTHING_PASSES):
    """Round a polygon's staircase edges, holes included.

    Returns the original if smoothing produced anything invalid - a rounded
    perimeter is a presentation improvement, never worth emitting a broken
    geometry for.
    """
    if polygon.is_empty or iterations < 1:
        return polygon

    smoothed = Polygon(
        _chaikin(polygon.exterior.coords, iterations),
        [_chaikin(hole.coords, iterations) for hole in polygon.interiors],
    )
    # Chaikin doubles the vertex count per pass and leaves near-collinear runs
    # on the straights; an eighth of a pixel trims those without visibly
    # re-cornering the curve.
    smoothed = smoothed.simplify(PIX / 8, preserve_topology=True)
    if not smoothed.is_valid:
        smoothed = smoothed.buffer(0)
    return smoothed if smoothed.is_valid and not smoothed.is_empty else polygon


def _mask_to_polygon(mask: np.ndarray, tile):
    """Polygonise one mask into a single smoothed geometry in tile CRS.

    Smoothing happens here, on the CUMULATIVE mask, rather than on the bands
    derived from it. Bands share boundaries; smoothing each one separately
    moves the two sides of a shared edge independently and opens seams between
    adjacent days. Smoothing first and differencing after means both sides of
    every shared edge come from the same curve.
    """
    if not mask.any():
        return None

    affine = tile_affine(tile)
    parts = []
    for geom, value in raster_shapes(mask.astype(np.uint8), mask=mask, transform=affine):
        if not value:
            continue
        polygon = shape(geom)
        if polygon.area < MIN_BAND_AREA_M2:
            continue
        # Half a pixel first: takes the staircase down to its corners without
        # inventing detail the 500 m grid cannot support. Rounding follows.
        polygon = polygon.simplify(PIX / 2, preserve_topology=True)
        if polygon.is_empty:
            continue
        smoothed = _repair(_smooth(polygon))
        if smoothed is not None:
            parts.append(smoothed)

    if not parts:
        return None
    return _repair(unary_union(parts))


def _repair(geometry):
    """Return a valid equivalent of `geometry`, or None if it cannot be saved.

    Chaikin crosses the ring wherever a band pinches to a single cell wide,
    which real fronts do constantly along their ragged edges. buffer(0) is the
    standard repair: it re-noses the ring and drops the zero-area lobe the
    self-intersection created.
    """
    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        return geometry
    repaired = geometry.buffer(0)
    return repaired if repaired.is_valid and not repaired.is_empty else None


def _to_features(geometry, to_wgs84) -> List[Dict[str, Any]]:
    """Split a band geometry into valid WGS84 GeoJSON polygons.

    Nothing invalid leaves this function. A self-intersecting ring is not a
    rendering curiosity - it makes fill and outline disagree about which side
    is inside, which shows up as a stray wedge across the band.
    """
    geometry = _repair(geometry)
    if geometry is None:
        return []

    parts = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", []))
    features = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area < MIN_BAND_AREA_M2:
            continue

        # Repair AFTER reprojecting, not before. Smoothing leaves vertices a
        # few tens of metres apart; projecting those into degrees can collapse
        # a near-degenerate spike into a crossing that did not exist in the
        # metric CRS. Validating the geometry we actually emit, in the CRS we
        # emit it in, is the only check that means anything to the renderer.
        projected = _repair(shapely_transform(to_wgs84, part))
        if projected is None:
            continue
        for piece in ([projected] if projected.geom_type == "Polygon"
                      else list(getattr(projected, "geoms", []))):
            if piece.geom_type == "Polygon" and not piece.is_empty:
                features.append(mapping(piece))
    return features


def spread_bands(
    rollout: Sequence[Dict[str, Any]],
    tile,
    to_wgs84,
    *,
    threshold: float = DEFAULT_BAND_THRESHOLD,
) -> Dict[str, Any]:
    """Nested day bands as GeoJSON, outermost (latest) day first.

    Bands are disjoint - each cell belongs to the day it first burned - so a
    renderer can paint them at full opacity without colours stacking. Later
    days are still emitted first so draw order stays correct even for a
    renderer that ignores the sort key.
    """
    masks = cumulative_masks(rollout, threshold)

    # Smooth the cumulative outlines, then difference them into bands. Doing it
    # the other way round - band, then smooth - moves the two sides of a shared
    # edge independently and leaves hairline seams between adjacent days.
    #
    # Each outline is unioned with the one before it so the sequence stays
    # nested after smoothing: Chaikin perturbs each curve on its own, and
    # without this a later day could fall a few metres inside an earlier one
    # and punch a hole through it on difference.
    cumulative = []
    previous = None
    for mask in masks:
        outline = _mask_to_polygon(mask, tile)
        if previous is not None:
            outline = previous if outline is None else unary_union([outline, previous])
        cumulative.append(outline)
        previous = outline

    bands = []
    for index, outline in enumerate(cumulative):
        if outline is None:
            bands.append(None)
            continue
        inner = cumulative[index - 1] if index else None
        bands.append(outline if inner is None else _repair(outline.difference(inner)))

    features: List[Dict[str, Any]] = []
    for index in range(len(bands) - 1, -1, -1):
        step = rollout[index]
        day = index + 1
        for geometry in _to_features(bands[index], to_wgs84):
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "day": day,
                    "lead_hours": int(step.get("lead_hours") or day * 24),
                    "label": step.get("label") or f"day {day}",
                    "color": band_color(day),
                    "threshold": float(threshold),
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "threshold": float(threshold),
            "days": len(masks),
            "disjoint": True,
            "palette": "YlOrRd-reversed",
            "note": ("Each cell belongs to the day it first burned, so bands do not "
                     "overlap; the union of days 1..N is the area burned by day N."),
        },
    }

#: The already-burned area is drawn as ground truth, not forecast, so it takes
#: a neutral char colour rather than a place on the forecast ramp. Mixing it
#: into the YlOrRd scale would imply it is another lead time.
OBSERVED_COLOR = "#3f3f46"

#: Where the fire is treated as having started, for the marker.
IGNITION_COLOR = "#111827"


def observed_polygons(observed: np.ndarray, tile, to_wgs84, *, threshold: float = 0.5) -> Dict[str, Any]:
    """The footprint that has already burned, as GeoJSON.

    Separating this from the forecast is the difference between "this is gone"
    and "this might go". Drawn together without distinction, a viewer cannot
    tell which part of the shape is observation and which is a model output -
    and only one of those is worth evacuating on.
    """
    mask = np.asarray(observed, dtype=np.float32) >= threshold
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {"kind": "observed", "color": OBSERVED_COLOR,
                               "label": "Already burned"},
            }
            # Smoothed on the same terms as the forecast bands: a scar drawn
            # with square corners beside rounded bands reads as a different
            # kind of object than it is.
            for geometry in _to_features(_mask_to_polygon(mask, tile), to_wgs84)
        ],
    }


def ignition_feature(lon: float, lat: float, *, label: str = "Ignition") -> Dict[str, Any]:
    """The point the forecast was seeded from."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": {"kind": "ignition", "color": IGNITION_COLOR, "label": label},
    }


def spread_scene(
    rollout: Sequence[Dict[str, Any]],
    observed: Optional[np.ndarray],
    tile,
    to_wgs84,
    *,
    ignition_lon: float,
    ignition_lat: float,
    threshold: float = DEFAULT_BAND_THRESHOLD,
) -> Dict[str, Any]:
    """The whole picture: where it started, what has burned, where it may go.

    Three layers rather than one, because they carry different authority.
    The ignition point is an input, the burned area is observation, and the
    bands are a model output - and a viewer deciding anything needs to know
    which is which.
    """
    return {
        "ignition": ignition_feature(ignition_lon, ignition_lat),
        "observed": (observed_polygons(observed, tile, to_wgs84)
                     if observed is not None
                     else {"type": "FeatureCollection", "features": []}),
        "forecast": spread_bands(rollout, tile, to_wgs84, threshold=threshold),
    }
