// backend/routes/fireData.js
const express = require('express');
const router = express.Router();
const axios = require('axios');
const Wildfire = require('../models/Wildfire');
require('dotenv').config();

const rateLimit = require('express-rate-limit');

// Lightweight, test-safe limiter for this route
const wildfireLimiter = rateLimit({
  windowMs: Number(process.env.WILDFIRES_WINDOW_MS || 30_000), // 30s
  max: Number(process.env.WILDFIRES_MAX || 3),                 // 3 reqs / window
  standardHeaders: true,
  legacyHeaders: false,
  skip: () => process.env.NODE_ENV === 'test',                 // don’t affect Jest
});

// --- FIRMS config -----------------------------------------------------------
const FIRMS_PRODUCTS = [
  'VIIRS_NOAA21_NRT',
  'VIIRS_NOAA20_NRT',
  'MODIS_NRT',          // Terra + Aqua combined
];
const FIRMS_DAYS = Math.max(1, Math.min(Number(process.env.FIRMS_DAYS || 2), 5)); // clamp 1..5
const FIRMS_BBOX = process.env.FIRMS_BBOX || '-125.0,24.0,-66.0,49.0';           // CONUS default
const FIRMS_HEADER = 'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight';
const EARTH_RADIUS_M = 6_378_137;

// --- Helpers ---------------------------------------------------------------

function clampDays(days) {
  const n = Number(days || FIRMS_DAYS);
  if (!Number.isFinite(n)) return FIRMS_DAYS;
  return Math.max(1, Math.min(Math.round(n), 5));
}

function parseBbox(raw = FIRMS_BBOX) {
  const parts = String(raw || '')
    .split(',')
    .map(v => Number(v.trim()));
  if (parts.length !== 4 || parts.some(v => !Number.isFinite(v))) {
    return FIRMS_BBOX;
  }
  const [w, s, e, n] = parts;
  if (w >= e || s >= n || w < -180 || e > 180 || s < -90 || n > 90) {
    return FIRMS_BBOX;
  }
  return `${w},${s},${e},${n}`;
}

async function fetchFirmsProduct(product, opts = {}) {
  const NASA_KEY = (process.env.NASA_API_KEY || process.env.NASA_KEY || '').trim();
  const keyPart = NASA_KEY ? `${NASA_KEY}/` : '';
  const bbox = parseBbox(opts.bbox || FIRMS_BBOX);
  const days = clampDays(opts.days);
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${keyPart}${product}/${bbox}/${days}`;
  const { data } = await axios.get(url, {
    headers: { 'User-Agent': 'ignis-ai (chrisjuarez1596@gmail.com)' },
    responseType: 'text',
    timeout: 20_000,
  });
  return { product, data };
}

// Confidence to 0..100
function asConfidencePct(conf) {
  const s = String(conf ?? '').trim().toLowerCase();
  if (!s) return 60; // neutral default, not 0
  const n = Number(s);
  if (!Number.isNaN(n)) {
    if (n <= 1) return Math.round(n * 100); // 0..1 → 0..100
    return Math.max(0, Math.min(100, Math.round(n)));
  }
  if (s === 'l' || s === 'low') return 25;
  if (s === 'n' || s === 'nominal' || s === 'med' || s === 'medium') return 60;
  if (s === 'h' || s === 'high') return 90;
  return 60;
}

// Heuristic to drop likely gas flares
function isLikelyFlare({ daynight, frp, brightness, confidence, confidencePct }) {
  const conf = Number(confidence ?? confidencePct ?? NaN);
  return (
    daynight === 'N' &&
    !Number.isNaN(conf) &&
    conf <= 60 &&
    Number(frp) < 25 &&
    Number(brightness) < 330
  );
}

function asFiniteNumber(value, fallback = null) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// Parse FIRMS area CSV (VIIRS/MODIS area endpoint)
function parseCSV(csvText, opts = {}) {
  const { excludeFlares = true, predictableOnly = false, product = null } = opts;
  const lines = String(csvText || '').trim().split('\n');
  const dataLines = lines.slice(1);

  const rows = dataLines
    .map(line => {
      const cols = line.split(',');
      if (cols.length < 14) return null;

      const [
        lat, lon, bright_ti4, scan, track,
        acq_date, acq_time, satellite, instrument,
        confidence, version, bright_ti5, frp, daynight
      ] = cols;

      const hhmm = String(acq_time || '').padStart(4, '0');
      const iso = `${acq_date}T${hhmm.slice(0, 2)}:${hhmm.slice(2)}:00Z`;

      const latitude = asFiniteNumber(lat);
      const longitude = asFiniteNumber(lon);
      const brightness = asFiniteNumber(bright_ti4);
      const scanKm = asFiniteNumber(scan);
      const trackKm = asFiniteNumber(track);
      const frpVal = asFiniteNumber(frp);
      const brightTi5 = asFiniteNumber(bright_ti5);
      const confidencePct = asConfidencePct(confidence);
      if (latitude == null || longitude == null || brightness == null) return null;

      const brightnessCat =
        brightness >= 375 ? 'Extreme' :
        brightness >= 350 ? 'Severe'  :
        brightness >= 325 ? 'Moderate' : 'Small';

      const predictable = brightness >= 325 && confidencePct >= 50;

      return {
        latitude,
        longitude,
        brightness,
        confidence: confidencePct, // store as number 0..100
        satellite,
        instrument,
        product,
        scan: scanKm,
        track: trackKm,
        frp: frpVal,
        daynight,
        version,
        brightTi5,
        brightnessCat,
        predictable,
        timestamp: new Date(iso),
      };
    })
    .filter(Boolean);

  return rows.filter(f => {
    if (excludeFlares && isLikelyFlare(f)) return false;
    if (predictableOnly && !f.predictable) return false;
    return true;
  });
}

function stripCsvRows(csvText = '') {
  const lines = String(csvText || '')
    .trim()
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean);
  return lines.length > 1 ? lines.slice(1) : [];
}

function dedupeFires(fires = []) {
  const unique = new Map();
  for (const fire of fires) {
    const timestamp = fire.timestamp instanceof Date
      ? fire.timestamp.toISOString()
      : new Date(fire.timestamp).toISOString();
    const key = [
      fire.latitude,
      fire.longitude,
      timestamp,
      fire.satellite || '',
      fire.instrument || '',
    ].join('|');
    if (!unique.has(key)) {
      unique.set(key, fire);
    }
  }
  return Array.from(unique.values());
}

function decorateFire(fire = {}) {
  const brightness = Number(fire.brightness ?? 0);
  const confidence = asConfidencePct(fire.confidence);
  return {
    ...fire,
    brightness,
    confidence,
    scan: asFiniteNumber(fire.scan),
    track: asFiniteNumber(fire.track),
    frp: asFiniteNumber(fire.frp),
    instrument: fire.instrument,
    satellite: fire.satellite,
    product: fire.product,
    daynight: fire.daynight,
    version: fire.version,
    brightTi5: asFiniteNumber(fire.brightTi5),
    brightnessCat:
      brightness >= 375 ? 'Extreme' :
      brightness >= 350 ? 'Severe'  :
      brightness >= 325 ? 'Moderate' : 'Small',
    predictable: fire.predictable === true || (brightness >= 325 && confidence >= 50),
  };
}

async function loadCachedFires({ predictableOnly = false } = {}) {
  if (typeof Wildfire.find !== 'function') {
    return [];
  }

  try {
    let query = Wildfire.find({});
    if (query && typeof query.sort === 'function') query = query.sort({ timestamp: -1 });
    if (query && typeof query.limit === 'function') query = query.limit(5000);
    if (query && typeof query.lean === 'function') query = query.lean();

    const docs = await query;
    const normalized = Array.isArray(docs) ? docs.map(decorateFire) : [];
    return predictableOnly ? normalized.filter(f => f.predictable) : normalized;
  } catch (err) {
    console.warn('⚠️  Failed to load cached wildfire data:', err.message);
    return [];
  }
}

function productPixelDefaultsKm(fire = {}) {
  const product = String(fire.product || '').toUpperCase();
  const instrument = String(fire.instrument || '').toUpperCase();
  if (product.includes('MODIS') || instrument.includes('MODIS')) {
    return { scanKm: 1.0, trackKm: 1.0, nominal: 'MODIS_1KM' };
  }
  return { scanKm: 0.375, trackKm: 0.375, nominal: 'VIIRS_375M' };
}

function fireToFootprintFeature(fire = {}) {
  const lat = Number(fire.latitude);
  const lon = Number(fire.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const defaults = productPixelDefaultsKm(fire);
  const scanKm = Number.isFinite(Number(fire.scan)) && Number(fire.scan) > 0
    ? Number(fire.scan)
    : defaults.scanKm;
  const trackKm = Number.isFinite(Number(fire.track)) && Number(fire.track) > 0
    ? Number(fire.track)
    : defaults.trackKm;

  const halfWidthM = (scanKm * 1000) / 2;
  const halfHeightM = (trackKm * 1000) / 2;
  const latDelta = (halfHeightM / EARTH_RADIUS_M) * (180 / Math.PI);
  const cosLat = Math.max(Math.cos(lat * Math.PI / 180), 0.01);
  const lonDelta = (halfWidthM / (EARTH_RADIUS_M * cosLat)) * (180 / Math.PI);

  const west = lon - lonDelta;
  const east = lon + lonDelta;
  const south = lat - latDelta;
  const north = lat + latDelta;

  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [west, north],
        [east, north],
        [east, south],
        [west, south],
        [west, north],
      ]],
    },
    properties: {
      latitude: lat,
      longitude: lon,
      brightness: Number(fire.brightness ?? 0),
      brightnessCat: fire.brightnessCat,
      confidence: Number(fire.confidence ?? 0),
      confidencePct: Number(fire.confidence ?? 0),
      predictable: fire.predictable === true,
      satellite: fire.satellite || null,
      instrument: fire.instrument || null,
      product: fire.product || null,
      scan: scanKm,
      track: trackKm,
      frp: Number.isFinite(Number(fire.frp)) ? Number(fire.frp) : null,
      daynight: fire.daynight || null,
      version: fire.version || null,
      timestamp: fire.timestamp instanceof Date ? fire.timestamp.toISOString() : fire.timestamp,
      footprint_source: (fire.scan && fire.track) ? 'firms_scan_track' : defaults.nominal,
    },
  };
}

function firesToFootprintGeoJSON(fires = []) {
  return {
    type: 'FeatureCollection',
    features: fires.map(fireToFootprintFeature).filter(Boolean),
  };
}

async function fetchCurrentFires(opts = {}) {
  const excludeFlares = opts.excludeFlares !== false;
  const predictableOnly = opts.predictableOnly === true;
  const bbox = parseBbox(opts.bbox || FIRMS_BBOX);
  const days = clampDays(opts.days);

  const results = await Promise.allSettled(FIRMS_PRODUCTS.map(product => fetchFirmsProduct(product, { bbox, days })));
  const successfulBodies = results
    .filter(r => r.status === 'fulfilled')
    .map(r => r.value);

  if (!successfulBodies.length) {
    const cached = await loadCachedFires({ predictableOnly });
    return {
      fires: cached,
      stale: true,
      inserted: 0,
      fetched: false,
      message: cached.length
        ? 'FIRMS fetch failed; returning cached wildfire data'
        : 'FIRMS fetch failed; no cached wildfire data',
    };
  }

  const rowsByProduct = successfulBodies.flatMap(({ product, data }) => (
    stripCsvRows(data).map(row => ({ product, row }))
  ));

  if (!rowsByProduct.length) {
    return { fires: [], stale: false, inserted: 0, fetched: true, message: 'No valid fire data' };
  }

  const fires = dedupeFires(
    rowsByProduct.flatMap(({ product, row }) => (
      parseCSV(`${FIRMS_HEADER}\n${row}`, { excludeFlares, predictableOnly, product })
    ))
  );

  return { fires, stale: false, inserted: null, fetched: true, message: 'Wildfire data fetched' };
}

// --- Route ----------------------------------------------------------------

router.get('/wildfires', wildfireLimiter, async (req, res) => {
  try {
    const excludeFlares   = (req.query.excludeFlares ?? 'true') !== 'false';
    const predictableOnly = (req.query.predictableOnly ?? 'false') === 'true';

    const result = await fetchCurrentFires({ excludeFlares, predictableOnly });
    if (!result.fetched) {
      return res.status(200).json({
        message: result.message,
        count: result.fires.length,
        data: result.fires,
        stale: true,
      });
    }

    if (!result.fires.length) {
      return res.status(200).json({ message: 'No valid fire data', count: 0, data: [] });
    }

    // Insert rows; ignore duplicate errors
    let inserted = [];
    try {
      inserted = await Wildfire.insertMany(result.fires, { ordered: false });
    } catch (_) { /* ignore dup key errors */ }

    const insertedCount = Array.isArray(inserted) ? inserted.length : 0;
    // ok to log counts
    console.log(`🔥 Parsed ${result.fires.length} (inserted ${insertedCount})`);

    const message = insertedCount > 0
      ? 'Wildfire data fetched & stored'
      : 'Wildfire data fetched';

    return res.status(200).json({
      message,
      count: result.fires.length,
      inserted: insertedCount,
      data: result.fires,              // ✅ what the frontend expects
    });
  } catch (err) {
    console.error('❌ Error fetching wildfire data:', err);
    return res.status(500).json({ error: err.message });
  }
});

router.get('/wildfires/footprints', wildfireLimiter, async (req, res) => {
  try {
    const excludeFlares = (req.query.excludeFlares ?? 'true') !== 'false';
    const predictableOnly = (req.query.predictableOnly ?? 'false') === 'true';
    const result = await fetchCurrentFires({
      excludeFlares,
      predictableOnly,
      bbox: req.query.bbox,
      days: req.query.days,
    });

    const geojson = firesToFootprintGeoJSON(result.fires);
    return res.status(200).json({
      geojson,
      count: geojson.features.length,
      stale: result.stale === true,
      message: result.message,
      caveat: 'FIRMS detections are satellite pixel footprints, not exact fire perimeters.',
    });
  } catch (err) {
    console.error('❌ Error building wildfire footprints:', err);
    return res.status(500).json({ error: err.message });
  }
});

module.exports = router;
module.exports._private = {
  parseCSV,
  fireToFootprintFeature,
  firesToFootprintGeoJSON,
  parseBbox,
};
