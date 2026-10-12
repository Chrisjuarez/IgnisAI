// backend/routes/predictFireSpread.js
const express = require("express");
const axios = require("axios");
const mapData = require("./mapData");
const { forecastErrorResponse } = require("../utils/forecastErrorResponse");
const { logSafe } = require("../utils/logSafe");

const router = express.Router();

// tilesvc inside docker network should be http://tilesvc:8008 if you set it,
// but keep default for local dev.
const TILE_SVC = process.env.TILE_SVC || "http://localhost:8008";

// tilesvc cold-start (boot + FIRMS + weather + model load) + 6-step rollout
// can take 3-4 minutes. A shorter timeout aborts the client before the model
// finishes and forces a full restart from step 1.
const MULTISTEP_TIMEOUT_MS = Number(process.env.MULTISTEP_TIMEOUT_MS || 240000);

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

async function burnedAreaOrNull(incidentId) {
  if (!incidentId) return null;
  try {
    return await mapData.burnedAreaForIncident(String(incidentId));
  } catch (err) {
    console.warn(`burned area lookup failed for ${logSafe(incidentId)}: ${err.message}`);
    return null;
  }
}

// GET /api/predict-fire-spread/multistep?lat=&lon=&steps=&step_hours=&Tseq=&thr=&date=&debug=&incident_id=
router.get("/multistep", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const {
      lat, lon, steps, step_hours, Tseq, thr, display_floor, crop_frac, date, debug, ignition, incident_id,
    } = req.query;
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

    // For an incident with a mapped perimeter, the forecast grows from it:
    // tilesvc treats it as burned and draws the bands beyond it. Sent as a
    // body because a perimeter does not fit in a query string.
    //
    // No retry: a multistep retry restarts the whole rollout on tilesvc's
    // side, which usually guarantees the next attempt also times out.
    const burnedArea = await burnedAreaOrNull(incident_id);
    const { data: forecast } = await axios.post(
      `${TILE_SVC}/predict_multistep`,
      { burned_area: burnedArea },
      { params, timeout: MULTISTEP_TIMEOUT_MS },
    );
    if (!Array.isArray(forecast?.bounds) || forecast.bounds.length !== 4 || !Array.isArray(forecast?.steps)) {
      throw new Error(`tilesvc multistep missing bounds/steps: ${JSON.stringify(forecast)?.slice(0, 200)}`);
    }

    // Forward the forecast whole. This route used to rebuild the response from
    // a field whitelist, so every field tilesvc added later was dropped without
    // a trace - `scene` among them. The arrival-band view, its labels and the
    // hybrid layer all shipped in tilesvc and never reached a browser. The
    // route adds nothing to the payload: validate the shape, pass it on, and a
    // new field needs one change instead of two.
    return res.json(forecast);
  } catch (err) {
    console.error("multistep error:", err?.response?.data || err.message);
    const { status, body } = forecastErrorResponse(err);
    return res.status(status).json(body);
  }
});

// GET /api/predict-fire-spread/site-exposure?site_lat=&site_lon=&ignition_lat=&ignition_lon=&days=
//
// The map answers "where will this fire go"; an asset owner asks "does it reach
// my site, and when". Same forecast, read at one coordinate. Thin proxy: the
// projection and sampling live in tilesvc, which owns the grid.
router.get("/site-exposure", async (req, res) => {
  try {
    if (!predictionsEnabled()) return sendPredictionsDisabled(res);
    const { site_lat, site_lon, ignition_lat, ignition_lon, days, Tseq, thr, arrival_threshold, date } = req.query;
    if (site_lat == null || site_lon == null) {
      return res.status(400).json({ error: "site_lat and site_lon are required" });
    }

    const params = {
      site_lat,
      site_lon,
      ...(ignition_lat != null ? { ignition_lat } : {}),
      ...(ignition_lon != null ? { ignition_lon } : {}),
      ...(days != null ? { days } : {}),
      ...(Tseq ? { Tseq } : {}),
      ...(thr ? { thr } : {}),
      ...(arrival_threshold != null ? { arrival_threshold } : {}),
      ...(date ? { date } : {}),
    };

    // A rollout per request, so this carries the multistep timeout and does not
    // retry: a retry restarts the whole rollout and re-spends the same minutes.
    const { data: exposure } = await axios.get(`${TILE_SVC}/site_exposure`, { params, timeout: MULTISTEP_TIMEOUT_MS });
    return res.json(exposure);
  } catch (err) {
    console.error("site-exposure error:", err?.response?.data || err.message);
    const { status, body } = forecastErrorResponse(err);
    return res.status(status).json(body);
  }
});

module.exports = router;
