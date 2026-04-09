// frontend/src/utils/addPredictionOverlay.js
//
// Adds either:
// - Raster overlay (recommended): Mapbox image source using bounds + PNG (base64),
//   colorized to a heatmap so it looks like a probability field (not a grid).
// - Vector overlay (optional): GeoJSON source + fill/line layers
//
// IMPORTANT: Backwards compatible with BOTH call styles:
//   A) addPredictionOverlay(map, apiBase, lat, lon, opts)
//   B) addPredictionOverlay(map, { apiBase, lat, lon, mode, thr, Tseq })
//
// Also exports removePredictionOverlays(map) so you can add a “Clear overlay” UI later.

const IDS = {
  rasterSource: 'ignis-pred-raster-src',
  rasterLayer: 'ignis-pred-raster-layer',
  boundsSource: 'ignis-pred-bounds-src',
  boundsLayer: 'ignis-pred-bounds-layer',

  vectorSource: 'ignis-pred-vector-src',
  vectorFill: 'ignis-pred-vector-fill',
  vectorLine: 'ignis-pred-vector-line',
};

function waitForMapLoad(map) {
  if (!map) return Promise.reject(new Error('Map instance missing'));
  if (map.loaded()) return Promise.resolve();
  return new Promise((resolve) => map.once('load', resolve));
}

function sniffContentType(resp) {
  return (resp?.headers?.get('content-type') || '').toLowerCase();
}

async function safeReadBodyForDebug(resp, limit = 600) {
  try {
    const txt = await resp.text();
    return txt.slice(0, limit);
  } catch {
    return '';
  }
}

function removeIfExists(map, layerId, sourceId) {
  if (!map) return;
  if (layerId && map.getLayer(layerId)) {
    try { map.removeLayer(layerId); } catch (_) {}
  }
  if (sourceId && map.getSource(sourceId)) {
    try { map.removeSource(sourceId); } catch (_) {}
  }
}

function boundsToPolygonGeoJSON(bounds) {
  const [w, s, e, n] = bounds;
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [w, s],
          [e, s],
          [e, n],
          [w, n],
          [w, s],
        ]]
      }
    }]
  };
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

/**
 * Colorize a grayscale PNG (base64) into a heatmap PNG data URL.
 * - floor: values below floor become transparent
 * - gamma: contrast curve
 * - opacity: global opacity multiplier
 * - smooth: softens the "grid" feeling
 */
async function colorizeGrayscalePngToHeatmapDataUrl(base64Png, opts = {}) {
  const {
    gamma = 0.7,
    floor = 0.18,          // hide low values
    opacity = 0.60,        // slightly stronger after backend smoothing
    smooth = true,
    alphaThreshold = 0.18, // cut speckle
  } = opts;

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = `data:image/png;base64,${base64Png}`;

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error('failed to decode PNG image'));
  });

  // Base draw
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);

  // Optional smoothing step (reduces block appearance)
  if (smooth) {
    const tmp = document.createElement('canvas');
    tmp.width = img.width * 2;
    tmp.height = img.height * 2;
    const tctx = tmp.getContext('2d');
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(canvas, 0, 0, tmp.width, tmp.height);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  }

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Grayscale PNG comes in as R=G=B, A=255
  for (let i = 0; i < data.length; i += 4) {
    const g = data[i] / 255; // 0..1
    // Hard cut: hide anything below threshold (prevents "red tile at 0%")
    if (alphaThreshold != null && g < alphaThreshold) {
      data[i + 3] = 0;
      continue;
    }

    let v = (g - floor) / (1 - floor);
    v = clamp01(v);
    v = Math.pow(v, gamma);

    // Alpha by intensity
    const a = clamp01(v * opacity);

    // Heatmap ramp (simple, effective)
    let r, gg, b;
    if (v <= 0) {
      r = 0; gg = 0; b = 0;
    } else if (v < 0.33) {
      // dark -> yellow
      const t = v / 0.33;
      r = 255;
      gg = Math.round(255 * t);
      b = 0;
    } else if (v < 0.66) {
      // yellow -> orange
      const t = (v - 0.33) / 0.33;
      r = 255;
      gg = Math.round(255 - 90 * t);
      b = 0;
    } else {
      // orange -> red
      const t = (v - 0.66) / 0.34;
      r = 255;
      gg = Math.round(165 - 165 * t);
      b = 0;
    }

    data[i] = r;
    data[i + 1] = gg;
    data[i + 2] = b;
    data[i + 3] = Math.round(a * 255);
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

/**
 * Raster overlay: calls backend raster endpoint and draws heatmap image layer.
 * Expects backend JSON:
 *   { bounds:[w,s,e,n], image_base64, threshold, prob_min/max/mean, area_fraction }
 */
export async function addRasterOverlay(map, apiBase, lat, lon, opts = {}) {
  await waitForMapLoad(map);

  const qs = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    ...(opts.Tseq != null ? { Tseq: String(opts.Tseq) } : { Tseq: "1" }),
    ...(opts.thr != null ? { thr: String(opts.thr) } : { thr: "0.1" }),
  });

  const url = `${apiBase}/predict-fire-spread/raster?${qs.toString()}`;
  const resp = await fetch(url);
  const ct = sniffContentType(resp);

  if (ct.includes('text/html')) {
    const body = await safeReadBodyForDebug(resp);
    throw new Error(
      `Raster returned text/html (likely /api proxy issue). Body: ${body}`
    );
  }

  if (!resp.ok) {
    const body = await safeReadBodyForDebug(resp);
    throw new Error(`Raster endpoint failed (${resp.status}). Body: ${body}`);
  }

  if (!ct.includes('application/json')) {
    const body = await safeReadBodyForDebug(resp);
    throw new Error(`Unexpected raster content-type: ${ct}. Body: ${body}`);
  }

  const payload = await resp.json();

  if (typeof payload === 'string') {
    throw new Error(
      'Raster endpoint returned JSON string (binary PNG). Backend must return {bounds,image_base64}.'
    );
  }

  const bounds = payload?.bounds;
  const b64 = payload?.image_base64;

  if (!Array.isArray(bounds) || bounds.length !== 4) {
    throw new Error(`Bad raster bounds: ${JSON.stringify(bounds)}`);
  }
  if (!b64 || typeof b64 !== 'string') {
    throw new Error('Missing image_base64 from raster endpoint');
  }

  // Colorize grayscale to heatmap
  const heatmapUrl = await colorizeGrayscalePngToHeatmapDataUrl(b64, {
    gamma: opts.gamma ?? 0.85,
    floor: opts.floor ?? 0.02,
    opacity: opts.opacity ?? 0.85,
    smooth: opts.smooth ?? true,
    alphaThreshold: (opts.thr != null ? Number(opts.thr) : null),
  });

  const [west, south, east, north] = bounds;
  const coords = [
    [west, north],
    [east, north],
    [east, south],
    [west, south],
  ];

  // Remove any existing raster overlay (stable IDs)
  removeIfExists(map, IDS.rasterLayer, IDS.rasterSource);

  map.addSource(IDS.rasterSource, {
    type: 'image',
    url: heatmapUrl,
    coordinates: coords,
  });

  map.addLayer({
    id: IDS.rasterLayer,
    type: 'raster',
    source: IDS.rasterSource,
    paint: {
      'raster-opacity': 1.0,
      'raster-resampling': 'linear',
    },
  });

  // Add/update bounds outline for alignment/debug (optional)
  if (opts.showBounds) {
    const boundsGeo = boundsToPolygonGeoJSON(bounds);
    if (!map.getSource(IDS.boundsSource)) {
      map.addSource(IDS.boundsSource, { type: 'geojson', data: boundsGeo });
    } else {
      map.getSource(IDS.boundsSource).setData(boundsGeo);
    }

    if (!map.getLayer(IDS.boundsLayer)) {
      map.addLayer({
        id: IDS.boundsLayer,
        type: 'line',
        source: IDS.boundsSource,
        paint: {
          'line-width': 2,
          'line-opacity': 0.6,
        },
      });
    }
  } else {
    removeIfExists(map, IDS.boundsLayer, IDS.boundsSource);
  }

  return { kind: 'raster', bounds, meta: payload };
}

/**
 * Vector overlay: calls backend vector endpoint.
 * WARNING: vector can be huge (thousands of polygons) and looks "grid-like".
 * We keep it, but we auto-block insanely large feature counts.
 */
export async function addVectorOverlay(map, apiBase, lat, lon, opts = {}) {
  await waitForMapLoad(map);

  const qs = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    ...(opts.Tseq != null ? { Tseq: String(opts.Tseq) } : { Tseq: "1" }),
    ...(opts.thr != null ? { thr: String(opts.thr) } : { thr: "0.1" }),
  });

  const url = `${apiBase}/predict-fire-spread/vector?${qs.toString()}`;
  const resp = await fetch(url);
  const ct = sniffContentType(resp);

  if (ct.includes('text/html')) {
    const body = await safeReadBodyForDebug(resp);
    throw new Error(`Vector returned text/html (/api proxy issue). Body: ${body}`);
  }

  if (!resp.ok) {
    const body = await safeReadBodyForDebug(resp);
    throw new Error(`Vector endpoint failed (${resp.status}). Body: ${body}`);
  }

  const data = await resp.json();
  const geojson =
    data?.type === 'FeatureCollection' ? data :
    data?.geojson?.type === 'FeatureCollection' ? data.geojson :
    null;

  if (!geojson) {
    throw new Error(
      `Vector endpoint did not return FeatureCollection. Got: ${JSON.stringify(data).slice(0, 400)}`
    );
  }

  const count = geojson.features?.length ?? 0;
  if (count > (opts.maxFeatures ?? 2000)) {
    throw new Error(
      `Vector has ${count} polygons (too many). Use raster mode for a clean overlay.`
    );
  }

  // Remove existing vector overlay (stable IDs)
  removeIfExists(map, IDS.vectorFill, null);
  removeIfExists(map, IDS.vectorLine, null);
  if (map.getSource(IDS.vectorSource)) {
    try { map.removeSource(IDS.vectorSource); } catch (_) {}
  }

  map.addSource(IDS.vectorSource, { type: 'geojson', data: geojson });

  map.addLayer({
    id: IDS.vectorFill,
    type: 'fill',
    source: IDS.vectorSource,
    paint: {
      'fill-opacity': 0.25,
      // You can pick a nicer fill color later; leaving color default is okay for correctness.
      // If you want explicit:
      // 'fill-color': 'rgba(255, 80, 0, 0.7)'
    },
  });

  map.addLayer({
    id: IDS.vectorLine,
    type: 'line',
    source: IDS.vectorSource,
    paint: {
      'line-width': 2,
      'line-opacity': 0.9,
    },
  });

  return { kind: 'vector', featureCount: count, meta: data };
}

/**
 * BACKWARDS COMPAT EXPORT:
 * - old style: addPredictionOverlay(map, apiBase, lat, lon, opts)
 * - new style: addPredictionOverlay(map, { apiBase, lat, lon, mode, thr, Tseq })
 */
export async function addPredictionOverlay(map, apiBaseOrParams, latMaybe, lonMaybe, optsMaybe = {}) {
  // New style: addPredictionOverlay(map, { apiBase, lat, lon, mode, thr, ... })
  if (typeof apiBaseOrParams === 'object' && apiBaseOrParams) {
    const p = apiBaseOrParams;
    const apiBase = p.apiBase || p.baseURL || p.baseUrl || '/api';
    const lat = p.lat ?? p.latitude;
    const lon = p.lon ?? p.lng ?? p.longitude;
    const mode = (p.mode || 'raster').toLowerCase();

    if (lat == null || lon == null) throw new Error(`lat/lon missing (lat=${lat}, lon=${lon})`);

    if (mode === 'vector') return addVectorOverlay(map, apiBase, lat, lon, p);
    return addRasterOverlay(map, apiBase, lat, lon, p);
  }

  // Old style: addPredictionOverlay(map, apiBase, lat, lon, opts)
  const apiBase = apiBaseOrParams || '/api';
  const lat = latMaybe;
  const lon = lonMaybe;
  const opts = optsMaybe || {};
  const mode = (opts.mode || 'raster').toLowerCase();

  if (lat == null || lon == null) throw new Error(`lat/lon missing (lat=${lat}, lon=${lon})`);

  if (mode === 'vector') return addVectorOverlay(map, apiBase, lat, lon, opts);
  return addRasterOverlay(map, apiBase, lat, lon, opts);
}

export function removePredictionOverlays(map) {
  if (!map) return;
  removeIfExists(map, IDS.rasterLayer, IDS.rasterSource);
  removeIfExists(map, IDS.vectorFill, null);
  removeIfExists(map, IDS.vectorLine, null);
  if (map.getSource(IDS.vectorSource)) {
    try { map.removeSource(IDS.vectorSource); } catch (_) {}
  }
  if (map.getLayer(IDS.boundsLayer)) {
    try { map.removeLayer(IDS.boundsLayer); } catch (_) {}
  }
  if (map.getSource(IDS.boundsSource)) {
    try { map.removeSource(IDS.boundsSource); } catch (_) {}
  }
}
