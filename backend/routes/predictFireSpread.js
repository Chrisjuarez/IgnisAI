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

const PREDICT_RETRY_BACKOFF_MS = Number(
  process.env.PREDICT_RETRY_BACKOFF_MS || (process.env.NODE_ENV === "test" ? 0 : 750)
);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function getJSON(url, params = {}, timeout = 80000, retries = 2) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const r = await axios.get(url, { params, timeout });
      return r.data;
    } catch (err) {
      lastErr = err;
      const status = err?.response?.status;
      // Retry on connection errors or 502/504 (tilesvc cold start).
      // NOTE: for long-running endpoints (e.g. /predict_multistep) pass
      // retries=0 — each retry restarts the rollout from step 1 and
      // wastes minutes of compute. Use a generous single-attempt timeout
      // instead.
      if (i < retries && (!status || status === 502 || status === 504)) {
        console.warn(`getJSON attempt ${i + 1} failed (${status || err.code}), retrying...`);
        if (PREDICT_RETRY_BACKOFF_MS > 0) {
          await sleep(PREDICT_RETRY_BACKOFF_MS * (i + 1));
        }
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

// tilesvc cold-start (boot + FIRMS + weather + model load) + 6-step rollout
// can take 3-4 minutes. A shorter timeout aborts the client before the model
// finishes and forces a full restart from step 1.
const MULTISTEP_TIMEOUT_MS = Number(process.env.MULTISTEP_TIMEOUT_MS || 240000);

function pickModelFields(payload = {}) {
  return {
    ...(payload?.model_meta ? { model_meta: payload.model_meta } : {}),
    ...(payload?.input_summary ? { input_summary: payload.input_summary } : {}),
    ...(payload?.probability_scale ? { probability_scale: payload.probability_scale } : {}),
    ...(payload?.quality ? { quality: payload.quality } : {}),
    ...(payload?.data_sources ? { data_sources: payload.data_sources } : {}),
    ...(payload?.p_new_burn ? { p_new_burn: payload.p_new_burn } : {}),
    ...(payload?.p_next_fire ? { p_next_fire: payload.p_next_fire } : {}),
    ...(payload?.observed_fire ? { observed_fire: payload.observed_fire } : {}),
    ...(payload?.display_score ? { display_score: payload.display_score } : {}),
    ...(payload?.risk_class ? { risk_class: payload.risk_class } : {}),
    ...(payload?.layer_images ? { layer_images: payload.layer_images } : {}),
    ...(payload?.display_mask ? { display_mask: payload.display_mask } : {}),
    ...(payload?.threshold_override != null ? { threshold_override: payload.threshold_override } : {}),
  };
}

function predictionsEnabled() {
  return !["0", "false", "no", "off"].includes(String(process.env.PREDICTIONS_ENABLED ?? "true").trim().toLowerCase());
}

function sendPredictionsDisabled(res) {
  return res.status(503).json({
    error: "predictions_disabled",
    detail: "Model predictions are disabled; observed FIRMS/perimeter layers remain available.",
    quality: {
      status: "unavailable",
      degraded: false,
      reasons: ["predictions_disabled"],
    },
  });
}

// Normalize the optional `ignition` query param. Defaults to true so live
// clicks always seed a starting fire — FIRMS has lag and misses small fires,
// so without ignition the model is handed an empty fire channel and produces
// a blank forecast. Callers can still opt out with `ignition=false`.
function resolveIgnition(raw) {
  if (raw == null) return true;
  const v = String(raw).trim().toLowerCase();
  if (["0", "false", "no", "off"].includes(v)) return false;
  return true;
}

// GET /api/predict-fire-spread/raster?lat=&lon=&Tseq=&thr=&date=
router.get("/raster", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const { lat, lon, Tseq, thr, display_floor, crop_frac, date, ignition } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const params = {
      lat,
      lon,
      ...(Tseq ? { Tseq } : {}),
      ...(thr ? { thr } : {}),
      ...(display_floor ? { display_floor } : {}),
      crop_frac: crop_frac != null ? crop_frac : 0.5,
      ...(date ? { date } : {}),
      ignition: resolveIgnition(ignition),
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
      display_floor: raster?.display_floor,
      prob_min: raster?.prob_min,
      prob_max: raster?.prob_max,
      prob_mean: raster?.prob_mean,
      area_fraction: raster?.area_fraction,
      display_area_fraction: raster?.display_area_fraction,
      ...pickModelFields(raster),
    });
  } catch (err) {
    console.error("raster error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_raster_failed", detail: err.message });
  }
});

// GET /api/predict-fire-spread/vector?lat=&lon=&Tseq=&thr=&date=
router.get("/vector", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const { lat, lon, Tseq, thr, crop_frac, date, ignition } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const crop = crop_frac != null ? crop_frac : 0.5;
    const sharedParams = {
      ...(date ? { date } : {}),
      ignition: resolveIgnition(ignition),
    };

    // 1) GeoJSON polygons from tilesvc (honor thr!)
    const geojson = await getJSON(
      `${TILE_SVC}/predict_geojson`,
      { lat, lon, ...(Tseq ? { Tseq } : {}), ...(thr ? { thr } : {}), crop_frac: crop, ...sharedParams },
      20000
    );

    // 2) Meta (area_fraction + threshold + bounds) from tilesvc
    const meta = await getJSON(
      `${TILE_SVC}/predict`,
      { lat, lon, ...(Tseq ? { Tseq } : {}), png: false, ...(thr ? { thr } : {}), crop_frac: crop, ...sharedParams },
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
      ...pickModelFields(meta),
    });
  } catch (err) {
    console.error("vector error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_vector_failed", detail: err.message });
  }
});

// GET /api/predict-fire-spread/multistep?lat=&lon=&steps=&step_hours=&Tseq=&thr=&date=&debug=
router.get("/multistep", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const { lat, lon, steps, step_hours, Tseq, thr, display_floor, crop_frac, date, debug, ignition } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const params = {
      lat,
      lon,
      steps: steps || 6,
      ...(step_hours ? { step_hours } : {}),
      ...(Tseq ? { Tseq } : {}),
      ...(thr ? { thr } : {}),
      ...(display_floor ? { display_floor } : {}),
      crop_frac: crop_frac != null ? crop_frac : 0.5,
      ...(date ? { date } : {}),
      ignition: resolveIgnition(ignition),
      // Forward diagnostic modes (e.g. "solid", "dump", "solid,dump")
      // so we can exercise the tilesvc debug harness through the normal
      // stack without hitting the python service directly.
      ...(debug ? { debug } : {}),
    };

    // retries=0: a multistep retry restarts the whole rollout on tilesvc's
    // side, which usually guarantees the next attempt also times out. One
    // generous attempt is strictly better.
    const forecast = await getJSON(`${TILE_SVC}/predict_multistep`, params, MULTISTEP_TIMEOUT_MS, 0);
    if (!Array.isArray(forecast?.bounds) || forecast.bounds.length !== 4 || !Array.isArray(forecast?.steps)) {
      throw new Error(`tilesvc multistep missing bounds/steps: ${JSON.stringify(forecast)?.slice(0, 200)}`);
    }

    return res.json({
      bounds: forecast.bounds,
      coordinates: Array.isArray(forecast?.coordinates) ? forecast.coordinates : undefined,
      threshold: forecast?.threshold,
      display_floor: forecast?.display_floor,
      step_hours: forecast?.step_hours,
      steps: forecast.steps,
      ...pickModelFields(forecast),
      ...(forecast?.debug ? { debug: forecast.debug } : {}),
    });
  } catch (err) {
    console.error("multistep error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_multistep_failed", detail: err.message });
  }
});

// GET /api/predict-fire-spread/input-audit?lat=&lon=&date=&Tseq=&step_hours=
router.get("/input-audit", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const { lat, lon, Tseq, date, step_hours, ignition, include_npz } = req.query;
    if (lat == null || lon == null) {
      return res.status(400).json({ error: "lat and lon are required" });
    }

    const audit = await getJSON(
      `${TILE_SVC}/input_audit`,
      {
        lat,
        lon,
        ...(Tseq ? { Tseq } : {}),
        ...(date ? { date } : {}),
        ignition: resolveIgnition(ignition),
        ...(step_hours ? { step_hours } : {}),
        ...(include_npz ? { include_npz } : {}),
      },
      80000
    );

    return res.json(audit);
  } catch (err) {
    console.error("input-audit error:", err?.response?.data || err.message);
    return res.status(502).json({ error: "tilesvc_input_audit_failed", detail: err.message });
  }
});

module.exports = router;
