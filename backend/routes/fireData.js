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

// --- Helpers ---------------------------------------------------------------

async function fetchFirmsProduct(product) {
const NASA_KEY = (process.env.NASA_API_KEY || process.env.NASA_KEY || '').trim();
const keyPart = NASA_KEY ? `${NASA_KEY}/` : '';
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${keyPart}${product}/${FIRMS_BBOX}/${FIRMS_DAYS}`;
  const { data } = await axios.get(url, {
    headers: { 'User-Agent': 'ignis-ai (chrisjuarez1596@gmail.com)' },
    responseType: 'text',
    timeout: 20_000,
  });
  return data;
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

// Parse FIRMS area CSV (VIIRS/MODIS area endpoint)
function parseCSV(csvText, opts = {}) {
  const { excludeFlares = true, predictableOnly = false } = opts;
  const lines = csvText.trim().split('\n');
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

      const latitude = parseFloat(lat);
      const longitude = parseFloat(lon);
      const brightness = parseFloat(bright_ti4);
      const frpVal = parseFloat(frp);
      const confidencePct = asConfidencePct(confidence);

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
        frp: frpVal,
        daynight,
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

// --- Route ----------------------------------------------------------------

router.get('/wildfires', wildfireLimiter, async (req, res) => {
  try {
    const excludeFlares   = (req.query.excludeFlares ?? 'true') !== 'false';
    const predictableOnly = (req.query.predictableOnly ?? 'false') === 'true';

    // Fetch multiple FIRMS products, merge, drop headers
    const results = await Promise.allSettled(FIRMS_PRODUCTS.map(fetchFirmsProduct));
    const bodies = results
      .filter(r => r.status === 'fulfilled')
      .map(r => r.value.split('\n').slice(1).join('\n')) // strip header
      .filter(Boolean);

    if (!bodies.length) {
      return res.status(502).json({ message: 'FIRMS fetch failed', count: 0, data: [] });
    }

    const mergedCsv = 'latitude,longitude,acq_date,acq_time\n' + bodies.join('\n');
    const fires = parseCSV(mergedCsv, { excludeFlares, predictableOnly });
    const parsedCount = fires.length;

    // Header-only / no rows
    if (parsedCount === 0) {
      return res.status(200).json({ message: 'No valid fire data', count: 0, data: [] });
    }

    // Insert rows; ignore duplicate errors
    let inserted = [];
    try {
      inserted = await Wildfire.insertMany(fires, { ordered: false });
    } catch (_) { /* ignore dup key errors */ }

    const insertedCount = Array.isArray(inserted) ? inserted.length : 0;
    // ok to log counts
    console.log(`🔥 Parsed ${parsedCount} (inserted ${insertedCount})`);

    const message = insertedCount > 0
      ? 'Wildfire data fetched & stored'
      : 'Wildfire data fetched';

    return res.status(200).json({
      message,
      count: fires.length,
      inserted: insertedCount,
      data: fires,              // ✅ what the frontend expects
    });
  } catch (err) {
    console.error('❌ Error fetching wildfire data:', err);
    return res.status(500).json({ error: err.message });
  }
});

module.exports = router;
