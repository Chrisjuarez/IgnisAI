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

// A failed request carries a message a person can read, so callers show
// `error.message` as is.
const NETWORK_MESSAGES = {
  ECONNABORTED: 'The request took too long. Try again in a minute.',
  ETIMEDOUT: 'The request took too long. Try again in a minute.',
  ERR_NETWORK: 'Ignis could not be reached. Check your connection.',
};

/**
 * What to tell a person about a failed request: the API's own message when it
 * sent one, a general one when the server failed without one - Render's own
 * error page while the backend restarts carries none - or what went wrong on
 * the way there. Undefined when there is nothing better than axios's text.
 */
export function readableMessage(error) {
  const apiMessage = error.response?.data?.message;
  if (apiMessage) return apiMessage;
  if (error.response?.status >= 500) return 'Ignis is having trouble right now. Try again shortly.';
  return NETWORK_MESSAGES[error.code];
}

/** Replaces the failed request's `message` with the readable one, in place. */
export function attachReadableMessage(error) {
  const message = readableMessage(error);
  if (message) error.message = message;
  return error;
}

api.interceptors.response.use(undefined, (error) => Promise.reject(attachReadableMessage(error)));

// -----------------------------
// Existing calls
// -----------------------------
export const getWildfireData = (opts = {}) => api.get("/wildfires", { params: opts });
export const getWildfireFootprints = (opts = {}) => api.get("/wildfires/footprints", { params: opts });
export const getFirePerimeters = (opts = {}) => api.get("/fire-perimeters", { params: opts });
export const getMapBootstrap = (opts = {}) => api.get("/map/bootstrap", { params: opts });
export const getIncident = (id, opts = {}) => api.get(`/incidents/${encodeURIComponent(id)}`, { params: opts });
export const getIncidentUpdates = (id, opts = {}) => api.get(`/incidents/${encodeURIComponent(id)}/updates`, { params: opts });

// -----------------------------
// Prediction endpoints
// -----------------------------

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
  incidentId,
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
      // Lets the backend send the incident's mapped perimeter with the request,
      // so the forecast grows from it rather than over it.
      ...(incidentId ? { incident_id: incidentId } : {}),
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

export default api;
