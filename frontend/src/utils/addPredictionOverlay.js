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
 *
 * The backend now normalizes the probability map per-tile (min-max stretch)
 * so grayscale values span 0-255 even when absolute probabilities are low.
 * Pixels at exactly 0 are below-threshold and should be transparent.
 *
 * - gamma: contrast curve (< 1 brightens, > 1 darkens)
 * - opacity: global opacity multiplier
 * - smooth: softens the "grid" feeling
 */
async function colorizeGrayscalePngToHeatmapDataUrl(base64Png, opts = {}) {
  const {
    gamma = 0.65,
    opacity = 0.80,
    smooth = true,
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
  // Backend normalizes per-tile: 0 = transparent (below threshold), 1-255 = probability range
  for (let i = 0; i < data.length; i += 4) {
    const raw = data[i]; // 0..255
    const sourceAlpha = (data[i + 3] ?? 255) / 255;

    // Pixel value 0 means below threshold — fully transparent
    if (raw === 0 || sourceAlpha === 0) {
      data[i + 3] = 0;
      continue;
    }

    // Normalized intensity (backend already stretched min-max to 0-255)
    let v = raw / 255;
    v = Math.pow(v, gamma);

    // Alpha ramps up with intensity
    const a = clamp01(0.3 + v * 0.7) * opacity;

    // Inferno-inspired color ramp (matches notebook's matplotlib look)
    let r, g, b;
    if (v < 0.2) {
      // dark purple to deep red
      const t = v / 0.2;
      r = Math.round(40 + 160 * t);
      g = Math.round(10 + 10 * t);
      b = Math.round(80 - 60 * t);
    } else if (v < 0.45) {
      // deep red to orange
      const t = (v - 0.2) / 0.25;
      r = Math.round(200 + 55 * t);
      g = Math.round(20 + 100 * t);
      b = Math.round(20 - 20 * t);
    } else if (v < 0.7) {
      // orange to yellow
      const t = (v - 0.45) / 0.25;
      r = 255;
      g = Math.round(120 + 120 * t);
      b = Math.round(0 + 20 * t);
    } else {
      // yellow to bright white-yellow (hottest)
      const t = (v - 0.7) / 0.3;
      r = 255;
      g = Math.round(240 + 15 * t);
      b = Math.round(20 + 180 * t);
    }

    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = Math.round(a * sourceAlpha * 255);
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

function rasterBoundsToCoordinates(bounds) {
  const [west, south, east, north] = bounds;
  return [
    [west, north],
    [east, north],
    [east, south],
    [west, south],
  ];
}

function isFiniteCoordinatePair(pair) {
  return Array.isArray(pair)
    && pair.length === 2
    && Number.isFinite(Number(pair[0]))
    && Number.isFinite(Number(pair[1]));
}

function resolveRasterCoordinates(frame) {
  const coordinates = frame?.coordinates;
  if (Array.isArray(coordinates) && coordinates.length === 4 && coordinates.every(isFiniteCoordinatePair)) {
    return coordinates.map(([lon, lat]) => [Number(lon), Number(lat)]);
  }
  return rasterBoundsToCoordinates(frame?.bounds);
}

export async function renderPredictionRasterFrame(map, frame, opts = {}) {
  await waitForMapLoad(map);

  const bounds = frame?.bounds;
  const imageUrl = frame?.heatmapUrl;
  const coordinates = resolveRasterCoordinates(frame);
  if (!Array.isArray(bounds) || bounds.length !== 4) {
    throw new Error(`Bad raster bounds: ${JSON.stringify(bounds)}`);
  }
  if (!imageUrl || typeof imageUrl !== 'string') {
    throw new Error('Missing heatmapUrl for raster frame');
  }

  removeIfExists(map, IDS.rasterLayer, IDS.rasterSource);
  map.addSource(IDS.rasterSource, {
    type: 'image',
    url: imageUrl,
    coordinates,
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

  return { kind: 'raster', bounds, meta: frame?.meta || null };
}

export async function prepareMultistepRasterFrames(payload, opts = {}) {
  const bounds = payload?.bounds;
  const steps = Array.isArray(payload?.steps) ? payload.steps : [];
  if (!Array.isArray(bounds) || bounds.length !== 4) {
    throw new Error(`Bad multistep bounds: ${JSON.stringify(bounds)}`);
  }
  if (!steps.length) {
    throw new Error('Multistep payload did not include any steps');
  }

  const frames = await Promise.all(
    steps.map(async step => {
      if (!step?.image_base64 || typeof step.image_base64 !== 'string') {
        throw new Error('Multistep step missing image_base64');
      }
      const heatmapUrl = await colorizeGrayscalePngToHeatmapDataUrl(step.image_base64, {
        gamma: opts.gamma ?? 0.85,
        opacity: opts.opacity ?? 0.85,
        smooth: opts.smooth ?? true,
      });
      return {
        ...step,
        bounds,
        coordinates: payload?.coordinates,
        heatmapUrl,
        meta: {
          threshold: payload?.threshold,
          step_hours: payload?.step_hours,
          prob_min: step?.prob_min,
          prob_mean: step?.prob_mean,
          prob_max: step?.prob_max,
          area_fraction: step?.area_fraction,
        },
      };
    })
  );

  return {
    bounds,
    threshold: payload?.threshold,
    stepHours: payload?.step_hours,
    frames,
  };
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
    ...(opts.thr != null ? { thr: String(opts.thr) } : { thr: "0.01" }),
    ...(opts.date ? { date: opts.date } : {}),
  });

  const url = `${apiBase}/predict-fire-spread/raster?${qs.toString()}`;

  // Retry up to 3 times for cold-start timeouts (Render spins down idle services)
  let resp = null;
  let lastErr = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      resp = await fetch(url);
      if (resp.ok) break;
      // 502/504 often means tilesvc is still waking up — retry
      if (resp.status === 502 || resp.status === 504) {
        console.warn(`Raster attempt ${attempt + 1} got ${resp.status}, retrying...`);
        resp = null;
        continue;
      }
      break; // other errors, don't retry
    } catch (err) {
      lastErr = err;
      console.warn(`Raster attempt ${attempt + 1} failed:`, err.message);
    }
  }
  if (!resp) {
    throw lastErr || new Error('Raster endpoint unreachable after retries');
  }

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

  await renderPredictionRasterFrame(map, {
    bounds,
    coordinates: payload?.coordinates,
    heatmapUrl,
    meta: payload,
  }, opts);

  return {
    kind: 'raster',
    bounds,
    coordinates: payload?.coordinates,
    meta: payload
  };
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
    ...(opts.thr != null ? { thr: String(opts.thr) } : { thr: "0.01" }),
    ...(opts.date ? { date: opts.date } : {}),
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
