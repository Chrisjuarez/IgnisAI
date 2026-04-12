// frontend/src/api.js
import axios from "axios";

const API_BASE =
  (typeof window !== "undefined" && window._env_?.REACT_APP_API_URL) ||
  process.env.REACT_APP_API_URL ||
  "/api";

// Axios client
const api = axios.create({
  baseURL: API_BASE,
  timeout: 80000,
});

// Use something realistic given your model prob_max (~0.25).
// Tune later; 0.07–0.15 tends to show structure.
const DEFAULT_THR =
  (typeof window !== "undefined" &&
    window._env_?.REACT_APP_MODEL_THR &&
    Number(window._env_?.REACT_APP_MODEL_THR)) ||
  (process.env.REACT_APP_MODEL_THR ? Number(process.env.REACT_APP_MODEL_THR) : 0.01);

// -----------------------------
// Existing calls
// -----------------------------
export const getWildfireData = (opts = {}) => api.get("/wildfires", { params: opts });

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

  const threshold = thr == null ? DEFAULT_THR : Number(thr);

  const { data } = await api.get("/predict-fire-spread/vector", {
    params: {
      lat: latitude,
      lon: longitude,
      thr: threshold,
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
export const predictFireSpreadRaster = async ({ lat, lng, lon, thr, Tseq, date } = {}) => {
  const latitude = lat != null ? Number(lat) : null;
  const longitude =
    lon != null ? Number(lon) : (lng != null ? Number(lng) : null);

  if (latitude == null || Number.isNaN(latitude) || longitude == null || Number.isNaN(longitude)) {
    throw new Error(`predictFireSpreadRaster missing lat/lon (lat=${lat}, lng=${lng}, lon=${lon})`);
  }

  const threshold = thr == null ? DEFAULT_THR : Number(thr);

  const { data } = await api.get("/predict-fire-spread/raster", {
    params: {
      lat: latitude,
      lon: longitude,
      ...(Tseq ? { Tseq } : {}),
      ...(threshold != null ? { thr: threshold } : {}),
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
} = {}) => {
  const latitude = lat != null ? Number(lat) : null;
  const longitude =
    lon != null ? Number(lon) : (lng != null ? Number(lng) : null);

  if (latitude == null || Number.isNaN(latitude) || longitude == null || Number.isNaN(longitude)) {
    throw new Error(`predictFireSpreadMultistep missing lat/lon (lat=${lat}, lng=${lng}, lon=${lon})`);
  }

  const threshold = thr == null ? DEFAULT_THR : Number(thr);

  const { data } = await api.get("/predict-fire-spread/multistep", {
    params: {
      lat: latitude,
      lon: longitude,
      ...(steps ? { steps } : {}),
      ...(stepHours ? { step_hours: stepHours } : {}),
      ...(Tseq ? { Tseq } : {}),
      ...(threshold != null ? { thr: threshold } : {}),
      ...(date ? { date } : {}),
    },
  });

  return data;
};

// Keep your old name if the UI calls it:
export const predictFireSpread = async ({ lat, lng, thr, date } = {}) =>
  predictFireSpreadVector({ lat, lng, thr, date });

export default api;
