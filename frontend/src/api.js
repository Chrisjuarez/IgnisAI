// frontend/src/api.js
import axios from "axios";

const API_BASE =
  (typeof window !== "undefined" && window._env_?.REACT_APP_API_URL) ||
  process.env.REACT_APP_API_URL ||
  "/api";

// Axios client
// 80s was too tight: on Render cold-starts tilesvc needs ~1-2 min to boot
// (FIRMS + weather + model load) before it even starts the 6-step rollout,
// so the browser would abort ~halfway through the first attempt and leave
// Express retrying from step 1 forever. 300s gives enough headroom for a
// full cold-start + rollout while still bounding hangs.
const DEFAULT_TIMEOUT_MS = 300_000;
// Multistep rollout specifically can legitimately take 3-4 minutes on a
// cold tilesvc instance. Callers can still override.
const MULTISTEP_TIMEOUT_MS = 300_000;

const api = axios.create({
  baseURL: API_BASE,
  timeout: DEFAULT_TIMEOUT_MS,
});

// -----------------------------
// Existing calls
// -----------------------------
export const getWildfireData = (opts = {}) => api.get("/wildfires", { params: opts });
export const getWildfireFootprints = (opts = {}) => api.get("/wildfires/footprints", { params: opts });
export const getFirePerimeters = (opts = {}) => api.get("/fire-perimeters", { params: opts });
export const getMapBootstrap = (opts = {}) => api.get("/map/bootstrap", { params: opts });
export const getIncidents = (opts = {}) => api.get("/incidents", { params: opts });
export const getIncident = (id, opts = {}) => api.get(`/incidents/${encodeURIComponent(id)}`, { params: opts });
export const getIncidentUpdates = (id, opts = {}) => api.get(`/incidents/${encodeURIComponent(id)}/updates`, { params: opts });
export const getAlerts = (opts = {}) => api.get("/alerts", { params: opts });
export const getLayers = () => api.get("/layers");

// -----------------------------
// Prediction endpoints
// -----------------------------

/**
 * Vector prediction (GeoJSON + spread_probability + environmental_data)
 * Backend expects: /predict-fire-spread/vector?lat=&lon=&thr=
 */
export const predictFireSpreadVector = async ({ lat, lng, lon, thr, Tseq, date } = {}) => {
  const latitude = lat != null ? Number(lat) : null;

  // accept either lng or lon
  const longitude =
    lon != null ? Number(lon) : (lng != null ? Number(lng) : null);

  if (latitude == null || Number.isNaN(latitude) || longitude == null || Number.isNaN(longitude)) {
    throw new Error(`predictFireSpreadVector missing lat/lon (lat=${lat}, lng=${lng}, lon=${lon})`);
  }

  const threshold = thr == null ? null : Number(thr);

  const { data } = await api.get("/predict-fire-spread/vector", {
    params: {
      lat: latitude,
      lon: longitude,
      ...(threshold != null && !Number.isNaN(threshold) ? { thr: threshold } : {}),
      ...(Tseq ? { Tseq } : {}),
      ...(date ? { date } : {}),
    },
  });

  return data;
};

/**
 * Raster overlay (bounds + image_base64)
 * Backend expects: /predict-fire-spread/raster?lat=&lon=
 */
export const predictFireSpreadRaster = async ({ lat, lng, lon, thr, displayFloor, Tseq, date } = {}) => {
  const latitude = lat != null ? Number(lat) : null;
  const longitude =
    lon != null ? Number(lon) : (lng != null ? Number(lng) : null);

  if (latitude == null || Number.isNaN(latitude) || longitude == null || Number.isNaN(longitude)) {
    throw new Error(`predictFireSpreadRaster missing lat/lon (lat=${lat}, lng=${lng}, lon=${lon})`);
  }

  const threshold = thr == null ? null : Number(thr);

  const { data } = await api.get("/predict-fire-spread/raster", {
    params: {
      lat: latitude,
      lon: longitude,
      ...(Tseq ? { Tseq } : {}),
      ...(threshold != null && !Number.isNaN(threshold) ? { thr: threshold } : {}),
      ...(displayFloor != null && !Number.isNaN(Number(displayFloor)) ? { display_floor: Number(displayFloor) } : {}),
      ...(date ? { date } : {}),
    },
  });

  return data; // { bounds, image_base64, ... }
};

/**
 * Multistep raster forecast timeline.
 * Backend expects: /predict-fire-spread/multistep?lat=&lon=&steps=&step_hours=
 */
export const predictFireSpreadMultistep = async ({
  lat,
  lng,
  lon,
  thr,
  Tseq,
  date,
  steps,
  stepHours,
  displayFloor,
  ignition,
} = {}) => {
  const latitude = lat != null ? Number(lat) : null;
  const longitude =
    lon != null ? Number(lon) : (lng != null ? Number(lng) : null);

  if (latitude == null || Number.isNaN(latitude) || longitude == null || Number.isNaN(longitude)) {
    throw new Error(`predictFireSpreadMultistep missing lat/lon (lat=${lat}, lng=${lng}, lon=${lon})`);
  }

  const threshold = thr == null ? null : Number(thr);
  const ignitionParam =
    typeof ignition === "boolean" ? ignition : (ignition == null ? null : String(ignition));

  const { data } = await api.get("/predict-fire-spread/multistep", {
    // Cold-start tilesvc + 6-step rollout can take 3-4 minutes; explicit
    // override so a future lower default on `api` doesn't silently regress.
    timeout: MULTISTEP_TIMEOUT_MS,
    params: {
      lat: latitude,
      lon: longitude,
      ...(steps ? { steps } : {}),
      ...(stepHours ? { step_hours: stepHours } : {}),
      ...(Tseq ? { Tseq } : {}),
      ...(threshold != null && !Number.isNaN(threshold) ? { thr: threshold } : {}),
      ...(displayFloor != null && !Number.isNaN(Number(displayFloor)) ? { display_floor: Number(displayFloor) } : {}),
      ...(date ? { date } : {}),
      ...(ignitionParam != null ? { ignition: ignitionParam } : {}),
    },
  });

  return data;
};

/**
 * Fire exposure at a fixed asset.
 * Backend expects: /predict-fire-spread/site-exposure?site_lat=&site_lon=&ignition_lat=&ignition_lon=&days=
 *
 * The ignition defaults to the site itself, which answers "what if it starts
 * here"; passing an ignition answers "what if it starts over there".
 */
export const getSiteExposure = async ({
  siteLat,
  siteLon,
  ignitionLat,
  ignitionLon,
  days,
  arrivalThreshold,
  date,
} = {}) => {
  const lat = siteLat != null ? Number(siteLat) : null;
  const lon = siteLon != null ? Number(siteLon) : null;
  if (lat == null || Number.isNaN(lat) || lon == null || Number.isNaN(lon)) {
    throw new Error(`getSiteExposure missing site coordinates (lat=${siteLat}, lon=${siteLon})`);
  }

  const { data } = await api.get("/predict-fire-spread/site-exposure", {
    // One rollout per request, same cost profile as multistep.
    timeout: MULTISTEP_TIMEOUT_MS,
    params: {
      site_lat: lat,
      site_lon: lon,
      ...(ignitionLat != null && !Number.isNaN(Number(ignitionLat)) ? { ignition_lat: Number(ignitionLat) } : {}),
      ...(ignitionLon != null && !Number.isNaN(Number(ignitionLon)) ? { ignition_lon: Number(ignitionLon) } : {}),
      ...(days ? { days } : {}),
      ...(arrivalThreshold != null ? { arrival_threshold: Number(arrivalThreshold) } : {}),
      ...(date ? { date } : {}),
    },
  });

  return data;
};

// Keep your old name if the UI calls it:
export const predictFireSpread = async ({ lat, lng, thr, date } = {}) =>
  predictFireSpreadVector({ lat, lng, thr, date });

export default api;
