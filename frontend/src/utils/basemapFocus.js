// frontend/src/utils/basemapFocus.js
//
// Drains colour out of the basemap while a forecast is on screen.
//
// The default style is satellite imagery, and the forecast bands are a
// ColorBrewer YlOrRd ramp drawn over it. Those are the same hues: dark greens
// and browns under translucent oranges reads as brown haze, and no amount of
// palette work on the bands fixes a basemap competing for the same part of the
// spectrum. The NWCG progression-map standard asks for a base that "does not
// distract from fire polygons"; Watch Duty gets there by using plain OSM.
//
// Desaturating rather than switching styles keeps the imagery — you can still
// see the ridge you are looking at — while surrendering the colour to the fire
// layer. Terrain reads as grey relief, and the ramp is then the only saturated
// thing on screen.

const FOCUS = {
  'raster-saturation': -0.85, // imagery to near-greyscale; shape survives, colour does not
  'raster-contrast': -0.12,   // pull the darkest greens up so orange separates from them
  'raster-brightness-max': 0.82,
};

// Set on the map instance rather than in module scope so two map instances
// (or a remount) cannot inherit each other's saved values.
const SAVED = Symbol('ignis.basemapFocus.saved');

function rasterLayerIds(map) {
  try {
    return (map.getStyle()?.layers || [])
      .filter(layer => layer.type === 'raster')
      .map(layer => layer.id);
  } catch (_) {
    // Style not ready. The caller re-runs on styledata, so this is a no-op now.
    return [];
  }
}

function readPaint(map, layerId, property) {
  try {
    return map.getPaintProperty(layerId, property);
  } catch (_) {
    return undefined;
  }
}

function writePaint(map, layerId, property, value) {
  try {
    map.setPaintProperty(layerId, property, value);
  } catch (_) {
    // A raster layer that does not accept one of these is left as it is;
    // a dimmed basemap is cosmetic and never worth throwing over.
  }
}

/**
 * Focus the map on the forecast by muting the basemap, or restore it.
 *
 * Safe to call repeatedly with the same value, and safe to call before the
 * style has loaded. Original paint values are captured once, on the first dim,
 * so a restore puts back exactly what the style shipped rather than a guess.
 */
export function setBasemapFocus(map, focused) {
  if (!map) return;

  const layerIds = rasterLayerIds(map);
  if (!layerIds.length) return;

  if (focused) {
    // Capture once. Re-capturing on a second dim would save the dimmed values
    // and make the restore a no-op.
    if (!map[SAVED]) {
      map[SAVED] = layerIds.reduce((saved, layerId) => {
        saved[layerId] = Object.keys(FOCUS).reduce((props, property) => {
          props[property] = readPaint(map, layerId, property);
          return props;
        }, {});
        return saved;
      }, {});
    }
    layerIds.forEach(layerId => {
      Object.entries(FOCUS).forEach(([property, value]) => writePaint(map, layerId, property, value));
    });
    return;
  }

  const saved = map[SAVED];
  if (!saved) return;
  Object.entries(saved).forEach(([layerId, props]) => {
    Object.entries(props).forEach(([property, value]) => {
      // undefined restores the style's own default rather than pinning a value.
      writePaint(map, layerId, property, value === undefined ? undefined : value);
    });
  });
  delete map[SAVED];
}

/**
 * Whether the basemap is currently muted. Lets a style reload re-apply the
 * effect without the caller tracking it separately.
 */
export function isBasemapFocused(map) {
  return Boolean(map && map[SAVED]);
}
