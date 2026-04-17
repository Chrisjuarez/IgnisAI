// backend/routes/predictFireSpread.js
const express = require("express");
const axios = require("axios");

const router = express.Router();

// tilesvc inside docker network should be http://tilesvc:8008 if you set it,
// but keep default for local dev.
const TILE_SVC = process.env.TILE_SVC || "http://localhost:8008";

const SELF_BASE =
  process.env.INTERNAL_SELF_BASE ||
  `http://127.0.0.1:${process.env.PORT || 5000}`;

async function getJSON(url, params = {}, timeout = 80000, retries = 2) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const r = await axios.get(url, { params, timeout });
      return r.data;
    } catch (err) {
      lastErr = err;
      const status = err?.response?.status;
      // Retry on connection errors or 502/504 (tilesvc cold start)
      if (i < retries && (!status || status === 502 || status === 504)) {
        console.warn(`getJSON attempt ${i + 1} failed (${status || err.code}), retrying...`);
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

// GET /api/predict-fire-spread/raster?lat=&lon=&Tseq=&thr=&date=
router.get("/raster", async (req, res) => {
  try {
    const { lat, lon, Tseq, thr, crop_frac, date } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const params = {
      lat,
      lon,
      Tseq: Tseq || 1,
      ...(thr ? { thr } : {}),
      crop_frac: crop_frac != null ? crop_frac : 0.5,
      ...(date ? { date, ignition: true } : {}),
    };

    // tilesvc returns base64 PNG + bounds; crop centers the output on the fire point
    const raster = await getJSON(`${TILE_SVC}/predict_raster_json`, params, 80000);

    const bounds = raster?.bounds;
    const image_base64 = raster?.image_base64;
    if (!Array.isArray(bounds) || bounds.length !== 4 || !image_base64) {
      throw new Error(`tilesvc raster missing bounds/image: ${JSON.stringify(raster)?.slice(0, 200)}`);
    }

    return res.json({
      bounds,
      coordinates: Array.isArray(raster?.coordinates) ? raster.coordinates : undefined,
      image_base64,
      threshold: raster?.threshold,
      prob_min: raster?.prob_min,
      prob_max: raster?.prob_max,
      prob_mean: raster?.prob_mean,
      area_fraction: raster?.area_fraction,
    });
  } catch (err) {
    console.error("raster error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_raster_failed", detail: err.message });
  }
});

// GET /api/predict-fire-spread/vector?lat=&lon=&Tseq=&thr=&date=
router.get("/vector", async (req, res) => {
  try {
    const { lat, lon, Tseq, thr, crop_frac, date } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const crop = crop_frac != null ? crop_frac : 0.5;
    const dateParams = date ? { date, ignition: true } : {};

    // 1) GeoJSON polygons from tilesvc (honor thr!)
    const geojson = await getJSON(
      `${TILE_SVC}/predict_geojson`,
      { lat, lon, Tseq: Tseq || 1, ...(thr ? { thr } : {}), crop_frac: crop, ...dateParams },
      20000
    );

    // 2) Meta (area_fraction + threshold + bounds) from tilesvc
    const meta = await getJSON(
      `${TILE_SVC}/predict`,
      { lat, lon, Tseq: Tseq || 1, png: false, ...(thr ? { thr } : {}), crop_frac: crop, ...dateParams },
      80000
    );

    const area_fraction =
      typeof meta?.area_fraction === "number" ? meta.area_fraction : 0.0;

    const best_thr =
      typeof meta?.threshold === "number"
        ? meta.threshold
        : (thr ? Number(thr) : 0.1);

    // 3) Weather (reuse backend route for consistency)
    let wx = null;
    try {
      const wr = await getJSON(
        new URL("/api/weather/current", SELF_BASE).toString(),
        { lat, lon },
        15000
      );
      wx = wr?.data?.current || null;
    } catch (_) {
      wx = null;
    }

    const wind_kmh = wx?.wind_speed_10m != null ? wx.wind_speed_10m * 3.6 : null; // m/s -> km/h

    const env = {
      wind_speed: wind_kmh,
      wind_direction: wx?.wind_direction_10m ?? null,
      temperature: wx?.temperature_2m ?? null,
      humidity: wx?.relative_humidity_2m ?? null,
      data_source: wx ? "weather_api" : "unknown",
    };

    // simple distance proxy based on area_fraction
    const spread_distance_km = Math.max(0, 64 * Math.sqrt(Math.max(0, area_fraction)));

    // Enrich features with direction so frontend can render it easily
    if (geojson?.features?.length) {
      geojson.features = geojson.features.map((f) => {
        const props = { ...(f.properties || {}) };
        props.direction = env.wind_direction; // simple proxy; replace with model-derived direction if available
        props.spread_probability = area_fraction;
        return { ...f, properties: props };
      });
    }

    return res.json({
      geojson,
      spread_probability: area_fraction, // 0..1
      spread_direction: env.wind_direction,
      spread_distance_km,
      environmental_data: env,
      threshold: best_thr,
      bounds: meta?.bounds ?? null,
      prob_min: meta?.prob_min,
      prob_max: meta?.prob_max,
      prob_mean: meta?.prob_mean,
    });
  } catch (err) {
    console.error("vector error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_vector_failed", detail: err.message });
  }
});

// GET /api/predict-fire-spread/multistep?lat=&lon=&steps=&step_hours=&Tseq=&thr=&date=&debug=
router.get("/multistep", async (req, res) => {
  try {
    const { lat, lon, steps, step_hours, Tseq, thr, crop_frac, date, debug } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const params = {
      lat,
      lon,
      steps: steps || 6,
      step_hours: step_hours || 6,
      Tseq: Tseq || 1,
      ...(thr ? { thr } : {}),
      crop_frac: crop_frac != null ? crop_frac : 0.5,
      ...(date ? { date, ignition: true } : {}),
      // Forward diagnostic modes (e.g. "solid", "dump", "solid,dump")
      // so we can exercise the tilesvc debug harness through the normal
      // stack without hitting the python service directly.
      ...(debug ? { debug } : {}),
    };

    const forecast = await getJSON(`${TILE_SVC}/predict_multistep`, params, 80000);
    if (!Array.isArray(forecast?.bounds) || forecast.bounds.length !== 4 || !Array.isArray(forecast?.steps)) {
      throw new Error(`tilesvc multistep missing bounds/steps: ${JSON.stringify(forecast)?.slice(0, 200)}`);
    }

    return res.json({
      bounds: forecast.bounds,
      coordinates: Array.isArray(forecast?.coordinates) ? forecast.coordinates : undefined,
      threshold: forecast?.threshold,
      step_hours: forecast?.step_hours,
      steps: forecast.steps,
      ...(forecast?.debug ? { debug: forecast.debug } : {}),
    });
  } catch (err) {
    console.error("multistep error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_multistep_failed", detail: err.message });
  }
});

module.exports = router;
