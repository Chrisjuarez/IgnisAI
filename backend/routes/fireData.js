// backend/routes/fireData.js
const express = require('express');
const router = express.Router();
const axios  = require('axios');
const Wildfire = require('../models/Wildfire');
require('dotenv').config();

// Robust confidence parser
function asConfidencePct(conf) {
  const s = String(conf ?? '').trim().toLowerCase();
  if (!s) return 60;                        // neutral default instead of 0
  const n = Number(s);
  if (!Number.isNaN(n)) {
    if (n <= 1) return Math.round(n * 100); // 0..1 → 0..100
    return Math.max(0, Math.min(100, Math.round(n)));
  }
  if (s === 'l' || s === 'low') return 25;
  if (s === 'n' || s === 'nominal' || s === 'med' || s === 'medium') return 60;
  if (s === 'h' || s === 'high') return 90;
  return 60;                                // unknown categorical → neutral
}

// Simple heuristics for likely gas flares
function isLikelyFlare({ daynight, frp, brightness, confidencePct }) {
  return (daynight === 'N') && (confidencePct <= 60) && (Number(frp) < 25) && (Number(brightness) < 330);
}

// Parse FIRMS area CSV (VIIRS 14 columns)
function parseCSV(csvText, opts = {}) {
  const { excludeFlares = true, predictableOnly = false } = opts;
  const lines = csvText.trim().split('\n');
  const dataLines = lines.slice(1);

  const rows = dataLines.map(line => {
    const cols = line.split(',');
    if (cols.length < 14) return null;

    const [
      lat, lon, bright_ti4, scan, track,
      acq_date, acq_time, satellite, instrument,
      confidence, version, bright_ti5, frp, daynight
    ] = cols;

    const hhmm = String(acq_time || '').padStart(4, '0');
    const iso = `${acq_date}T${hhmm.slice(0,2)}:${hhmm.slice(2)}:00Z`;

    const latitude   = parseFloat(lat);
    const longitude  = parseFloat(lon);
    const brightness = parseFloat(bright_ti4);
    const frpVal     = parseFloat(frp);
    const confidencePct = asConfidencePct(confidence);

    const brightnessCat =
      brightness >= 375 ? 'Extreme' :
      brightness >= 350 ? 'Severe'  :
      brightness >= 325 ? 'Moderate': 'Small';

    const predictable = (brightness >= 325) && (confidencePct >= 50);

    return {
      latitude,
      longitude,
      brightness,
      confidence: confidencePct,     // store as 0..100 number
      satellite,
      instrument,
      frp: frpVal,
      daynight,
      brightnessCat,
      predictable,
      timestamp: new Date(iso)
    };
  }).filter(Boolean);

  const filtered = rows.filter(f => {
    if (excludeFlares && isLikelyFlare(f)) return false;
    if (predictableOnly && !f.predictable) return false;
    return true;
  });

  return filtered;
}

router.get('/wildfires', async (req, res) => {
  try {
    const excludeFlares   = (req.query.excludeFlares ?? 'true') !== 'false';
    const predictableOnly = (req.query.predictableOnly ?? 'false') === 'true';

    const NASA_KEY = process.env.NASA_API_KEY || process.env.NASA_KEY || '';
    const keyPart  = NASA_KEY ? `${NASA_KEY}/` : '';
    const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${keyPart}VIIRS_NOAA21_NRT/-125.0,24.0,-66.0,49.0/2`;

    const redactedUrl = NASA_KEY ? url.replace(NASA_KEY, '[REDACTED_KEY]') : url;
    console.log(`Fetching FIRMS area data from:\n  ${redactedUrl}`);
    const { data: csvText } = await axios.get(url, {
      headers: { 'User-Agent': 'ignis-ai (chrisjuarez1596@gmail.com)' },
      responseType: 'text',
      timeout: 20000
    });

    const fires = parseCSV(csvText, { excludeFlares, predictableOnly });
    const parsedCount = fires.length;

    if (parsedCount === 0) {
      console.log('⚠️  No valid wildfire rows parsed.');
    }

    // insert only if we have rows; ignore duplicate-key errors
    let inserted = [];
    try {
      if (parsedCount > 0) {
        inserted = await Wildfire.insertMany(fires, { ordered: false });
      }
    } catch (_) { /* ignore dup errors */ }

    const insertedCount = Array.isArray(inserted) ? inserted.length : 0;
    console.log(`🔥 Parsed ${parsedCount} (inserted ${insertedCount})`);

    // Keep the exact message your CI test expects
    const message = insertedCount > 0 ? 'Wildfire data fetched & stored' : 'Wildfire data fetched';
    return res.status(200).json({ message, count: insertedCount });

  } catch (err) {
    console.error('❌ Error fetching wildfire data:', err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
