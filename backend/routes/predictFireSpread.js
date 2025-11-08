// backend/routes/predictFireSpread.js
const express = require('express');
const axios = require('axios');

const router = express.Router();
const TILE_SVC = process.env.TILE_SVC || 'http://localhost:8008';

async function getJSON(url, params = {}, timeout = 20000) {
  const r = await axios.get(url, { params, timeout });
  return r.data;
}

// PNG+bounds helper (kept for raster overlay)
async function rasterViaPng(lat, lon, Tseq = 3) {
  const [pngResp, bndResp] = await Promise.all([
    axios.get(`${TILE_SVC}/predict`, {
      params: { lat, lon, Tseq },
      responseType: 'arraybuffer',
      timeout: 20000
    }),
    axios.get(`${TILE_SVC}/tile-bounds`, {
      params: { lat, lon },
      timeout: 10000
    })
  ]);
  const image_base64 = Buffer.from(pngResp.data, 'binary').toString('base64');
  const bounds = bndResp.data?.bounds;
  if (!Array.isArray(bounds) || bounds.length !== 4) {
    throw new Error('bad tile-bounds response');
  }
  return { bounds, image_base64 };
}

// GET /api/predict-fire-spread/raster?lat=&lon=&Tseq=
router.get('/raster', async (req, res) => {
  try {
    const { lat, lon, Tseq } = req.query;
    if (!lat || !lon) return res.status(400).json({ error: 'lat and lon are required' });

    try {
      // tilesvc native JSON (bounds + b64 PNG)
      const data = await getJSON(`${TILE_SVC}/predict_raster`, {
        lat, lon, Tseq: Tseq || 3
      }, 20000);
      return res.json(data);
    } catch {
      // fallback to image endpoint + bounds
      const payload = await rasterViaPng(lat, lon, Tseq);
      return res.json(payload);
    }
  } catch (err) {
    console.error('raster error:', err.message);
    res.status(502).json({ error: 'tilesvc_raster_failed', detail: err.message });
  }
});

// GET /api/predict-fire-spread/vector?lat=&lon=&thr=
router.get('/vector', async (req, res) => {
  try {
    const { lat, lon, Tseq, thr } = req.query;
    if (!lat || !lon) return res.status(400).json({ error: 'lat and lon are required' });

    // 1) GeoJSON polygons from tilesvc
    const geojson = await getJSON(`${TILE_SVC}/predict_geojson`, {
      lat, lon, Tseq: Tseq || 3, thr
    }, 20000);

    // 2) Area fraction & threshold (png=false returns meta)
    const meta = await getJSON(`${TILE_SVC}/predict`, {
      lat, lon, Tseq: Tseq || 3, png: false
    }, 20000);
    const area_fraction = typeof meta?.area_fraction === 'number' ? meta.area_fraction : 0.0;
    const best_thr = typeof meta?.threshold === 'number' ? meta.threshold : (thr ? Number(thr) : 0.5);

    // 3) Weather (reuse your backend route for consistency)
    let wx = null;
    try {
      const wr = await getJSON(`${req.protocol}://${req.get('host')}/api/weather/current`, { lat, lon }, 15000);
      wx = wr?.data?.current || null;
    } catch (_) { /* ignore */ }

    const wind_kmh = wx?.wind_speed_10m != null ? wx.wind_speed_10m * 3.6 : null; // m/s -> km/h
    const env = {
      wind_speed: wind_kmh,
      wind_direction: wx?.wind_direction_10m ?? null,
      temperature: wx?.temperature_2m ?? null,
      humidity: wx?.relative_humidity_2m ?? null,
      data_source: wx ? 'weather_api' : 'unknown'
    };

    // rough, human-friendly “distance” proxy based on area fraction
    const spread_distance_km = Math.max(0, 64 * Math.sqrt(Math.max(0, area_fraction))); // 64km tile

    return res.json({
      geojson,
      spread_probability: area_fraction,   // 0..1
      spread_direction: env.wind_direction,
      spread_distance_km,
      environmental_data: env,
      threshold: best_thr
    });
  } catch (err) {
    console.error('vector error:', err?.response?.data || err.message);
    res.status(501).json({ error: 'vector_not_available', detail: 'tilesvc/predict_geojson missing' });
  }
});

// POST /api/predict-fire-spread  (back-compat: return vector-style payload)
router.post('/', async (req, res) => {
  try {
    const { lat, lng: lon, Tseq, thr } = req.body || {};
    if (!lat || !lon) return res.status(400).json({ error: 'lat and lon are required' });
    const data = await getJSON(`${req.protocol}://${req.get('host')}/api/predict-fire-spread/vector`, {
      lat, lon, Tseq: Tseq || 3, thr
    }, 25000);
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: 'tilesvc_vector_failed', detail: err.message });
  }
});

module.exports = router;