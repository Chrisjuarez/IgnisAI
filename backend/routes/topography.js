// routes/topography.js
// Fetch topography (elevation + approx slope/aspect) and store in Mongo
const express = require('express');
const axios = require('axios');
const Topography = require('../models/Topography');
const router = express.Router();

// --- Robust Elevation lookup (USGS EPQS) ---
async function fetchElevation(lat, lon) {
  // 1) New EPQS endpoint (most locations)
  try {
    const url = 'https://epqs.nationalmap.gov/v1/json';
    const { data } = await axios.get(url, {
      params: { x: lon, y: lat, units: 'Meters', wkid: 4326 },
      timeout: 10000,
      // follow redirects just in case
      maxRedirects: 5
    });

    // Possible new shapes:
    // { elevation: 123.4, units: "Meters", location: {...} }
    // { value: 123.4,  units: "Meters", location: {...} }
    const candidates = [
      data?.elevation,
      data?.value
    ];
    const picked = candidates.find(v =>
      typeof v === 'number' || (typeof v === 'string' && !Number.isNaN(parseFloat(v)))
    );
    if (picked != null) return typeof picked === 'number' ? picked : parseFloat(picked);
  } catch (e) {
    // swallow and fall through to legacy
  }

  // 2) Legacy EPQS (some regions still serve this)
  try {
    const legacyUrl = 'https://nationalmap.gov/epqs/pqs.php';
    const { data } = await axios.get(legacyUrl, {
      params: { x: lon, y: lat, units: 'Meters', output: 'json' },
      timeout: 10000,
      maxRedirects: 5
    });
    const v = data?.USGS_Elevation_Point_Query_Service?.Elevation_Query?.Elevation;
    if (typeof v === 'number') return v;
    if (typeof v === 'string' && !Number.isNaN(parseFloat(v))) return parseFloat(v);
  } catch (e) {
    // fall through
  }

  return null; // couldn’t resolve elevation
}

// --- Finite-difference slope/aspect from 3×3 neighborhood ---
async function slopeAspect(lat, lon, deltaMeters = 30) {
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(lat * Math.PI / 180);
  const dLat = deltaMeters / metersPerDegLat;
  const dLon = deltaMeters / metersPerDegLon;

  const [zC, zN, zS, zE, zW] = await Promise.all([
    fetchElevation(lat, lon),
    fetchElevation(lat + dLat, lon),
    fetchElevation(lat - dLat, lon),
    fetchElevation(lat, lon + dLon),
    fetchElevation(lat, lon - dLon),
  ]);

  // If center is missing, return all nulls
  if (zC == null) return { elevation: null, slope_deg: null, aspect_deg: null };

  // If any neighbor missing, return elevation only (avoid NaN math)
  if ([zN, zS, zE, zW].some(z => z == null)) {
    return { elevation: zC, slope_deg: null, aspect_deg: null };
  }

  const dzdx = (zE - zW) / (2 * deltaMeters); // east-west gradient
  const dzdy = (zN - zS) / (2 * deltaMeters); // north-south gradient

  const slope_rad = Math.atan(Math.hypot(dzdx, dzdy));
  const slope_deg = slope_rad * 180 / Math.PI;

  // Aspect: 0° = North, clockwise
  let aspect_rad = Math.atan2(dzdx, dzdy);
  if (aspect_rad < 0) aspect_rad += 2 * Math.PI;
  const aspect_deg = aspect_rad * 180 / Math.PI;

  return { elevation: zC, slope_deg, aspect_deg };
}

// GET /api/topography/point?lat=..&lon=..
router.get('/point', async (req, res) => {
  try {
    const lat = parseFloat(req.query.lat);
    const lon = parseFloat(req.query.lon);
    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      return res.status(400).json({ error: 'lat and lon are required numbers' });
    }

    const topo = await slopeAspect(lat, lon, 30);
    const doc = await Topography.create({
      latitude: lat,
      longitude: lon,
      fetchedAt: new Date(),
      ...topo
    });

    res.json({ ok: true, data: doc });
  } catch (err) {
    console.error('❌ topography/point error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;