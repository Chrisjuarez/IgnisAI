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
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from .grid import PIX, tile_affine

#: ColorBrewer YlOrRd, reversed: day 1 deepest, later days fading out.
#: Sequential, colour-vision-deficient safe, and legible in greyscale.
BAND_COLORS = ("#bd0026", "#f03b20", "#fd8d3c", "#feb24c", "#fed976", "#ffffb2")

#: Below this calibrated probability a cell is not part of the burn shape.
DEFAULT_BAND_THRESHOLD = 0.10

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


def _polygonize(mask: np.ndarray, tile, to_wgs84) -> List[Dict[str, Any]]:
    if not mask.any():
        return []

    affine = tile_affine(tile)
    geometries = []
    for geom, value in raster_shapes(mask.astype(np.uint8), mask=mask, transform=affine):
        if not value:
            continue
        polygon = shape(geom)
        if polygon.area < MIN_BAND_AREA_M2:
            continue
        # Half a pixel: enough to take the staircase off a rasterised edge
        # without inventing detail the 500 m grid cannot support.
        polygon = polygon.simplify(PIX / 2, preserve_topology=True)
        if polygon.is_empty:
            continue
        geometries.append(mapping(shapely_transform(to_wgs84, polygon)))
    return geometries


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
    bands = arrival_bands(masks)

    features: List[Dict[str, Any]] = []
    for index in range(len(bands) - 1, -1, -1):
        step = rollout[index]
        day = index + 1
        for geometry in _polygonize(bands[index], tile, to_wgs84):
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
            for geometry in _polygonize(mask, tile, to_wgs84)
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
