// backend/routes/weather.js
const express = require('express');
const axios = require('axios');
const mongoose = require('mongoose');
const Weather = require('../models/Weather');
const router = express.Router();

/**
 * GET /api/weather/current?lat=..&lon=..
 * Returns current + next 24h hourly weather.
 * Persists a snapshot only when Mongo is connected.
 */
router.get('/current', async (req, res) => {
  try {
    const lat = parseFloat(req.query.lat);
    const lon = parseFloat(req.query.lon);
    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      return res.status(400).json({ error: 'lat and lon are required numbers' });
    }

    const url = 'https://api.open-meteo.com/v1/forecast';
    const params = {
      latitude: lat,
      longitude: lon,
      current: [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'wind_direction_10m',
        'wind_gusts_10m',
        'precipitation'
      ],
      hourly: [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'wind_direction_10m',
        'wind_gusts_10m',
        'precipitation'
      ],
      timezone: 'UTC',
    };

    const { data } = await axios.get(url, { params });

    const hourly = {
      time: data.hourly?.time?.slice(0, 24) || [],
      temperature_2m: data.hourly?.temperature_2m?.slice(0, 24) || [],
      relative_humidity_2m: data.hourly?.relative_humidity_2m?.slice(0, 24) || [],
      wind_speed_10m: data.hourly?.wind_speed_10m?.slice(0, 24) || [],
      wind_gusts_10m: data.hourly?.wind_gusts_10m?.slice(0, 24) || [],
      precipitation: data.hourly?.precipitation?.slice(0, 24) || [],
    };
    const hourlyWindDirection = data.hourly?.wind_direction_10m?.slice(0, 24);
    if (Array.isArray(hourlyWindDirection) && hourlyWindDirection.length > 0) {
      hourly.wind_direction_10m = hourlyWindDirection;
    }

    const snapshot = {
      latitude: lat,
      longitude: lon,
      fetchedAt: new Date(),
      current: data.current,
      hourly,
    };

    // Only write if Mongo is actually connected (prevents buffering timeouts in tests)
    if (mongoose.connection.readyState === 1) {
      const doc = await Weather.create(snapshot);
      return res.json({ ok: true, data: doc });
    }

    // Not connected: return snapshot without persisting
    return res.json({ ok: true, data: snapshot, persisted: false });
  } catch (err) {
    console.error('❌ weather/current error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
