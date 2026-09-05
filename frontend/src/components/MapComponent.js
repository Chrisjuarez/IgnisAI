// src/components/MapComponent.js
/* eslint-disable react-hooks/exhaustive-deps */
import React, {
  useEffect,
  useRef,
  useState,
  useImperativeHandle,
  forwardRef,
  useCallback
} from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  getWildfireData,
  getWildfireFootprints,
  getFirePerimeters,
  predictFireSpreadMultistep
} from '../api';
import {
  addPredictionOverlay,
  prepareMultistepRasterFrames,
  removePredictionOverlays,
  removePredictionRaster,
  removePredictionScene,
  renderPredictionRasterFrame,
  renderPredictionScene
} from '../utils/addPredictionOverlay';

// ---- Tokens / Base URLs -----------------------------------------------------
const MAPBOX_TOKEN =
  (typeof window !== 'undefined' && window._env_?.REACT_APP_MAPBOX_TOKEN) ||
  process.env.REACT_APP_MAPBOX_TOKEN ||
  '';

mapboxgl.accessToken = MAPBOX_TOKEN;

const API_BASE =
  (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) ||
  process.env.REACT_APP_API_URL ||
  '/api';

const R_EARTH = 3958.8; // miles
const OBSERVED_CELL_MAX_KM = 0.42;
const OBSERVED_CELL_MIN_KM = 0.18;
// Hard ceiling on how long we'll wait for a multistep forecast before giving
// up and clearing the loading spinner. 180s matches a generous warm-path
// rollout (~45s) + render pipeline (~10s) with headroom, but stops short of
// axios's 300s so a UI-side hang after the response returns is surfaced
// before the user thinks the app is dead. If you see "Forecast timed out"
// *after* the '[forecast] 2/5 HTTP response received' breadcrumb, the bug
// is in colorize / render — not the network.
const FORECAST_TIMEOUT_MS = 180_000;
const FORECAST_RENDER_TIMEOUT_MS = 15_000;
// Matches the checkpoint's training cadence (model_config.default.json ->
// step_hours: 24). Forwarding explicitly keeps the autoregressive rollout in
// the training distribution; tilesvc will otherwise fall back to its own
// default which may drift if re-tuned.
const DEFAULT_STEP_HOURS = 24;

const FOOTPRINT_FILL_COLOR = [
  'match', ['get', 'brightnessCat'],
  'Extreme', '#d34823',
  'Severe',  '#ff6b35',
  'Moderate', '#ff9b45',
  '#ffd977'
];
const FOOTPRINT_FILL_OPACITY = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.02,
  50, 0.04,
  100, 0.07
];
const FOOTPRINT_FILL_OPACITY_FORECAST = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.0,
  100, 0.0
];
const FOOTPRINT_LINE_COLOR = [
  'match', ['get', 'instrument'],
  'MODIS', '#ffe0b2',
  '#fff3cf'
];
const FOOTPRINT_LINE_WIDTH = [
  'match', ['get', 'footprint_source'],
  'firms_scan_track', 0.9,
  0.45
];
const OBSERVED_CELL_FILL_OPACITY = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.14,
  50, 0.28,
  100, 0.48
];
const OBSERVED_CELL_FILL_OPACITY_FORECAST = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.05,
  100, 0.16
];
const POINT_CIRCLE_OPACITY = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.08,
  50, 0.18,
  100, 0.28
];
const POINT_CIRCLE_OPACITY_FORECAST = [
  'interpolate', ['linear'], ['get', 'confidencePct'],
  0, 0.03,
  100, 0.10
];
const PERIMETER_FILL_OPACITY = 0.06;
const PERIMETER_FILL_OPACITY_FORECAST = 0.035;

// ---- Helpers ----------------------------------------------------------------
function getBrightnessCategory(b) {
  if (b >= 375) return 'Extreme';
  if (b >= 350) return 'Severe';
  if (b >= 325) return 'Moderate';
  return 'Small';
}

function confidenceToPercent(raw) {
  const n = Number(raw);
  if (Number.isNaN(n)) return 0;
  return n <= 1 ? Math.round(n * 100) : Math.round(n);
}

function bracketConfidence(pct) {
  if (pct < 40) return 'Low';
  if (pct < 85) return 'Medium';
  return 'High';
}

function getSeverityIcon(cat) {
  if (cat === 'Extreme') return '🔥🔥🔥';
  if (cat === 'Severe') return '🔴🔥';
  if (cat === 'Moderate') return '⚠️';
  return '🌡️';
}

function haversineDistance(lat1, lon1, lat2, lon2) {
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R_EARTH * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

async function reverseGeocode(lat, lon, token) {
  try {
    const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lon},${lat}.json?access_token=${token}&limit=1`;
    const resp = await fetch(url);
    const { features } = await resp.json();
    return features?.[0]?.place_name || 'Unknown';
  } catch {
    return 'Unknown';
  }
}

const toCompass = d => {
  if (d == null || Number.isNaN(Number(d))) return '—';
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(((d % 360) / 22.5)) % 16];
};

const msToMph = v => (v == null ? null : v * 2.23694);
const fmt = (v, digits = 1) => (v == null || Number.isNaN(+v) ? '—' : (+v).toFixed(digits));
const pctStr = v => (v == null || Number.isNaN(+v) ? '—' : `${Math.round(+v)}%`);
const withTimeout = (promise, ms, message) => {
  let timeoutHandle = null;
  const timeout = new Promise((_, reject) => {
    timeoutHandle = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timeoutHandle) clearTimeout(timeoutHandle);
  });
};

const getDirectionName = deg => toCompass(deg);

// ---- Historical fire presets for model testing ----
const HISTORICAL_FIRES = [
  { name: 'Camp/Paradise Fire',  date: '2018-11-08', lat: 39.80, lon: -121.44, zoom: 10 },
  { name: 'Eaton Fire',          date: '2025-01-07', lat: 34.19, lon: -118.06, zoom: 11 },
  // Palisades Fire (2025-01-07): real ignition was at the head of Skull
  // Rock / Temescal Canyon (~34.078°N, -118.555°W), not at the coastline.
  // The previous lat=34.05 sat on PCH/beach where the synthetic ignition
  // landed on impervious or non-burnable pixels that get masked out.
  // ignition:true so the model gets a seed even when FIRMS snapshots
  // haven't been backfilled for January 2025 — without a seed the rollout
  // has no input fire signal and the heat defaults to the inland leeward
  // ridge (Beverly Crest / Bel Air) where baseline weather/fuel risk peaks.
  { name: 'Palisades Fire',      incidentName: 'Palisades', date: '2025-01-07T18:30:00Z', displayDate: '2025-01-07 10:30 PT', lat: 34.078, lon: -118.555, zoom: 11, ignition: true },
  { name: 'Dixie Fire',         date: '2021-07-14', lat: 40.05, lon: -121.38, zoom: 10 },
  { name: 'Caldor Fire',        date: '2021-08-14', lat: 38.75, lon: -120.30, zoom: 10 },
];

function emptyFeatureCollection() {
  return { type: 'FeatureCollection', features: [] };
}

function mapBoundsBbox(map) {
  const bounds = map?.getBounds?.();
  if (!bounds) return null;
  const west = bounds.getWest?.();
  const south = bounds.getSouth?.();
  const east = bounds.getEast?.();
  const north = bounds.getNorth?.();
  if ([west, south, east, north].some(v => !Number.isFinite(Number(v)))) return null;
  return [west, south, east, north].map(v => Number(v).toFixed(5)).join(',');
}

function bboxAround(lon, lat, spanDeg = 1.25) {
  return [
    Number(lon) - spanDeg,
    Number(lat) - spanDeg,
    Number(lon) + spanDeg,
    Number(lat) + spanDeg,
  ].map(v => v.toFixed(5)).join(',');
}

function addDaysIso(dateValue, days) {
  const d = new Date(dateValue);
  if (Number.isNaN(d.getTime())) return undefined;
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString();
}

function normalizedFootprintFeature(feature) {
  const props = feature?.properties || {};
  const brightness = Number(props.brightness ?? 0);
  const confidencePct = confidenceToPercent(props.confidencePct ?? props.confidence);
  return {
    ...feature,
    properties: {
      ...props,
      brightness,
      brightnessCat: props.brightnessCat || getBrightnessCategory(brightness),
      confidencePct,
      confidence: confidencePct,
      predictable: props.predictable === true || props.predictable === 'true',
    },
  };
}

function filterFootprintCollection(collection, brightnessFilter, confidenceFilter) {
  const features = Array.isArray(collection?.features) ? collection.features : [];
  return {
    type: 'FeatureCollection',
    features: features
      .map(normalizedFootprintFeature)
      .filter(feature => {
        const props = feature.properties || {};
        const b = props.brightnessCat || getBrightnessCategory(props.brightness);
        const c = bracketConfidence(confidenceToPercent(props.confidencePct ?? props.confidence));
        return (!brightnessFilter || brightnessFilter === b)
          && (!confidenceFilter || confidenceFilter === c);
      }),
  };
}

function fireMatchesFilters(fire, brightnessFilter, confidenceFilter) {
  const b = getBrightnessCategory(fire.brightness);
  const pct = confidenceToPercent(fire.confidence);
  const c = bracketConfidence(pct);
  return (!brightnessFilter || brightnessFilter === b)
    && (!confidenceFilter || confidenceFilter === c);
}

function normalizedFireProperties(fire) {
  const pct = confidenceToPercent(fire.confidence);
  const brightnessCat = getBrightnessCategory(fire.brightness);
  const predictable = (fire.predictable === true) || (fire.brightness >= 325 && pct >= 50);
  return {
    brightnessCat,
    confidencePct: pct,
    confidence: pct,
    predictable,
    timestamp: fire.timestamp,
    brightness: fire.brightness,
    frp: fire.frp,
    scan: fire.scan,
    track: fire.track,
    satellite: fire.satellite,
    instrument: fire.instrument,
    product: fire.product,
    daynight: fire.daynight,
    latitude: fire.latitude,
    longitude: fire.longitude
  };
}

function observedCellSizeKm(fire) {
  const scan = Number(fire.scan);
  const track = Number(fire.track);
  const nominal = String(fire.product || fire.instrument || '').toUpperCase().includes('MODIS') ? 0.50 : 0.32;
  const sourceSize = Math.max(
    Number.isFinite(scan) && scan > 0 ? scan : nominal,
    Number.isFinite(track) && track > 0 ? track : nominal,
  );
  return Math.max(OBSERVED_CELL_MIN_KM, Math.min(OBSERVED_CELL_MAX_KM, sourceSize));
}

function fireToObservedCellFeature(fire) {
  const lat = Number(fire.latitude);
  const lon = Number(fire.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const sizeKm = observedCellSizeKm(fire);
  const halfLat = (sizeKm / 111.32) / 2;
  const lonKm = Math.max(0.1, 111.32 * Math.cos((lat * Math.PI) / 180));
  const halfLon = (sizeKm / lonKm) / 2;
  const props = normalizedFireProperties(fire);
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [lon - halfLon, lat + halfLat],
        [lon + halfLon, lat + halfLat],
        [lon + halfLon, lat - halfLat],
        [lon - halfLon, lat - halfLat],
        [lon - halfLon, lat + halfLat],
      ]]
    },
    properties: {
      ...props,
      observed_cell_km: sizeKm,
      display_geometry: 'firms_detection_cell',
    }
  };
}

function observedCellCollection(dataArray, brightnessFilter, confidenceFilter) {
  const fires = Array.isArray(dataArray) ? dataArray : [];
  return {
    type: 'FeatureCollection',
    features: fires
      .filter(fire => fireMatchesFilters(fire, brightnessFilter, confidenceFilter))
      .map(fireToObservedCellFeature)
      .filter(Boolean),
  };
}

function incidentFeatureCollection(incidents = []) {
  return {
    type: 'FeatureCollection',
    features: (Array.isArray(incidents) ? incidents : [])
      .filter(incident => Number.isFinite(Number(incident.lat)) && Number.isFinite(Number(incident.lon)))
      .map(incident => ({
        type: 'Feature',
        geometry: incident.geometry || { type: 'Point', coordinates: [Number(incident.lon), Number(incident.lat)] },
        properties: {
          id: incident.id,
          name: incident.name || 'Unnamed incident',
          status: incident.status || 'unknown',
          type: incident.type || 'wildfire',
          acres: Number(incident.acres || 0),
          county: incident.county || '',
          state: incident.state || '',
          updatedAt: incident.updatedAt || '',
          hasPrediction: Boolean(incident.hasPrediction),
          hasPerimeter: Boolean(incident.hasPerimeter),
          hasHotspots: Boolean(incident.hasHotspots),
          lat: Number(incident.lat),
          lon: Number(incident.lon),
        },
      })),
  };
}

function alertFeatureCollection(alerts = []) {
  return {
    type: 'FeatureCollection',
    features: (Array.isArray(alerts) ? alerts : [])
      .filter(alert => alert.geometry)
      .map(alert => ({
        type: 'Feature',
        geometry: alert.geometry,
        properties: {
          id: alert.id,
          event: alert.event || 'Fire Weather Alert',
          headline: alert.headline || alert.event || 'Fire Weather Alert',
          areaDesc: alert.areaDesc || '',
          status: alert.status || '',
          effective: alert.effective || '',
          expires: alert.expires || alert.ends || '',
        },
      })),
  };
}

function featureCoordinateBounds(geometry) {
  const bounds = { w: Infinity, s: Infinity, e: -Infinity, n: -Infinity };
  const visit = coords => {
    if (!Array.isArray(coords)) return;
    if (typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      bounds.w = Math.min(bounds.w, coords[0]);
      bounds.e = Math.max(bounds.e, coords[0]);
      bounds.s = Math.min(bounds.s, coords[1]);
      bounds.n = Math.max(bounds.n, coords[1]);
      return;
    }
    coords.forEach(visit);
  };
  visit(geometry?.coordinates);
  if ([bounds.w, bounds.s, bounds.e, bounds.n].some(v => !Number.isFinite(v))) return null;
  return bounds;
}

function featureCollectionBounds(featureCollection) {
  const features = Array.isArray(featureCollection?.features) ? featureCollection.features : [];
  const merged = { w: Infinity, s: Infinity, e: -Infinity, n: -Infinity };
  features.forEach((feature) => {
    const bounds = featureCoordinateBounds(feature?.geometry);
    if (!bounds) return;
    merged.w = Math.min(merged.w, bounds.w);
    merged.s = Math.min(merged.s, bounds.s);
    merged.e = Math.max(merged.e, bounds.e);
    merged.n = Math.max(merged.n, bounds.n);
  });
  if ([merged.w, merged.s, merged.e, merged.n].some((value) => !Number.isFinite(value))) return null;
  return merged;
}

function centerFromBounds(bounds) {
  if (!bounds) return null;
  return {
    lon: (bounds.w + bounds.e) / 2,
    lat: (bounds.s + bounds.n) / 2,
  };
}

function hotspotCenterForIncident(fires = [], incident = {}) {
  const lat = Number(incident.lat);
  const lon = Number(incident.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const nearby = (Array.isArray(fires) ? fires : []).filter((fire) => {
    const fireLat = Number(fire.latitude);
    const fireLon = Number(fire.longitude);
    if (!Number.isFinite(fireLat) || !Number.isFinite(fireLon)) return false;
    return Math.abs(fireLat - lat) <= 0.35 && Math.abs(fireLon - lon) <= 0.35;
  });

  if (!nearby.length) return null;

  let totalWeight = 0;
  let weightedLat = 0;
  let weightedLon = 0;

  nearby.forEach((fire) => {
    const fireLat = Number(fire.latitude);
    const fireLon = Number(fire.longitude);
    const confidence = confidenceToPercent(fire.confidencePct ?? fire.confidence ?? 0);
    const frp = Number(fire.frp || 0);
    const weight = Math.max(1, frp, confidence / 8);
    totalWeight += weight;
    weightedLat += fireLat * weight;
    weightedLon += fireLon * weight;
  });

  if (!totalWeight) return null;

  return {
    lat: weightedLat / totalWeight,
    lon: weightedLon / totalWeight,
    sampleCount: nearby.length,
  };
}

// ---- Component --------------------------------------------------------------
const MapComponent = forwardRef(({
  brightnessFilter,
  confidenceFilter,
  onFiresUpdated,
  setIsFetching,
  mapStyle,
  userLocation,
  range,
  onNearbyFiresUpdate,
  incidents = [],
  alerts = [],
  layerVisibility = {},
  selectedIncident = null,
  selectedAlert = null,
  onIncidentSelect,
  onAlertSelect,
  watchShell = false
}, ref) => {
  const mapContainerRef = useRef();
  const mapRef          = useRef();
  const clickHandlerRef = useRef();
  const perimeterClickHandlerRef = useRef();
  const incidentClickHandlerRef = useRef();
  const alertClickHandlerRef = useRef();
  const incidentsRef = useRef([]);
  const alertsRef = useRef([]);

  const [wildfires, setWildfires]       = useState([]);
  const [wildfireFootprints, setWildfireFootprints] = useState(emptyFeatureCollection);
  const [firePerimeters, setFirePerimeters] = useState(emptyFeatureCollection);
  const [isPredicting, setIsPredicting] = useState(false);
  // Synchronous re-entry guard so rapid double-clicks on the "predict" action
  // can't fire two concurrent tilesvc rollouts before React re-renders the
  // state flag. This is what was showing up in the tilesvc logs as duplicate
  // step 1..6 lines for the same (lat, lon).
  const isPredictingRef = useRef(false);
  const [activePopup, setActivePopup]   = useState(null);
  const [forecastFrames, setForecastFrames] = useState([]);
  const [activeForecastIndex, setActiveForecastIndex] = useState(0);
  const [forecastVisible, setForecastVisible] = useState(false);
  const [isForecastPlaying, setIsForecastPlaying] = useState(false);
  const [forecastTitle, setForecastTitle] = useState('Ignis Forecast Timeline');
  const [isForecastLoading, setIsForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);
  const [observedLayersVisible, setObservedLayersVisibleState] = useState(true);
  const [forecastLayerMode, setForecastLayerMode] = useState('new_burn');
  // 'bands' draws cumulative day-by-day arrival polygons; 'heat' keeps the
  // per-cell probability raster for reading confidence inside a single day.
  const [spreadView, setSpreadView] = useState('bands');
  const [forecastScene, setForecastScene] = useState(null);

  // Historical fire testing state
  const [showHistPanel, setShowHistPanel] = useState(false);
  const [histDate, setHistDate]           = useState('');
  const [histRunning, setHistRunning]     = useState(false);
  // Same story as isPredictingRef — guard the historical preset clicks
  // against double-fire while React state is still flushing.
  const histRunningRef = useRef(false);

  // NDVI overlay state
  const [ndviOn, setNdviOn] = useState(false);
  const ndviTemplateRef     = useRef(null);
  const NDVI_SOURCE_ID      = 'ndvi-tiles';
  const NDVI_LAYER_ID       = 'ndvi-layer';

  // ---------- Data: FIRMS ----------
  const loadFirePerimeters = useCallback(async (opts = {}) => {
    const map = mapRef.current;
    try {
      const bbox = opts.bbox || mapBoundsBbox(map);
      const response = await getFirePerimeters({
        ...(bbox ? { bbox } : {}),
        ...(opts.incidentName ? { incidentName: opts.incidentName } : {}),
        ...(opts.from ? { from: opts.from } : {}),
        ...(opts.to ? { to: opts.to } : {}),
      });
      const payload = response?.data ?? response;
      const geojson = payload?.geojson?.type === 'FeatureCollection'
        ? payload.geojson
        : emptyFeatureCollection();
      setFirePerimeters(geojson);
      updatePerimeterSource(geojson);
      return geojson;
    } catch (err) {
      console.warn('fire perimeter fetch failed', err?.message || err);
      const empty = emptyFeatureCollection();
      setFirePerimeters(empty);
      updatePerimeterSource(empty);
      return empty;
    }
  }, []);

  const fetchWildfires = useCallback(async () => {
    setIsFetching?.(true);
    try {
      // You can pass { predictableOnly:true } if you want to hide weak signals by default.
      const map = mapRef.current;
      const bbox = mapBoundsBbox(map);
      const [firesResult, footprintsResult] = await Promise.allSettled([
        getWildfireData(),
        getWildfireFootprints({ ...(bbox ? { bbox } : {}), days: 2 }),
      ]);

      if (firesResult.status === 'fulfilled') {
        const payload = firesResult.value?.data ?? firesResult.value;
        const arr = Array.isArray(payload?.data)
          ? payload.data
          : Array.isArray(payload)
            ? payload
            : [];
        setWildfires(arr);
        onFiresUpdated?.(arr.length);
        updateWildfireSource(arr);
      } else {
        console.warn('wildfire point fetch failed', firesResult.reason?.message || firesResult.reason);
      }

      if (footprintsResult.status === 'fulfilled') {
        const payload = footprintsResult.value?.data ?? footprintsResult.value;
        const geojson = payload?.geojson?.type === 'FeatureCollection'
          ? payload.geojson
          : emptyFeatureCollection();
        setWildfireFootprints(geojson);
        updateWildfireFootprintsSource(geojson);
      } else {
        console.warn('wildfire footprint fetch failed', footprintsResult.reason?.message || footprintsResult.reason);
      }

      loadFirePerimeters({ ...(bbox ? { bbox } : {}) });
    } finally {
      setIsFetching?.(false);
    }
  }, [loadFirePerimeters, onFiresUpdated, setIsFetching]);

  // ---------- Update sources ----------
  function updateWildfireSource(dataArray = wildfires) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('wildfires-source');
    if (!src) return;

    const filtered = dataArray.filter(f => fireMatchesFilters(f, brightnessFilter, confidenceFilter));
    onFiresUpdated?.(filtered.length);

    src.setData({
      type: 'FeatureCollection',
      features: filtered.map(f => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [f.longitude, f.latitude] },
        properties: normalizedFireProperties(f)
      }))
    });
    updateObservedFireCellsSource(dataArray);
  }

  function updateObservedFireCellsSource(dataArray = wildfires) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('observed-fire-cells-source');
    if (!src) return;
    src.setData(observedCellCollection(dataArray, brightnessFilter, confidenceFilter));
  }

  function updateWildfireFootprintsSource(data = wildfireFootprints) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('wildfire-footprints-source');
    if (!src) return;
    src.setData(filterFootprintCollection(data, brightnessFilter, confidenceFilter));
  }

  function updatePerimeterSource(data = firePerimeters) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('fire-perimeters-source');
    if (!src) return;
    const geojson = data?.type === 'FeatureCollection' ? data : emptyFeatureCollection();
    src.setData(geojson);
  }

  function updateIncidentSource(data = incidentsRef.current) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('ignis-incidents-source');
    if (!src) return;
    src.setData(incidentFeatureCollection(data));
  }

  function updateAlertSource(data = alertsRef.current) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('nws-alerts-source');
    if (!src) return;
    src.setData(alertFeatureCollection(data));
  }

  function updateSelectedIncidentSource(incident = selectedIncident) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('selected-incident-source');
    if (!src) return;
    if (!incident) {
      src.setData(emptyFeatureCollection());
      return;
    }
    src.setData(incidentFeatureCollection([incident]));
  }

  function applyLayerVisibility(visibility = layerVisibility) {
    const map = mapRef.current;
    if (!map) return;
    const groups = {
      hotspots: [
        'observed-fire-cells-fill',
        'observed-fire-cells-outline',
        'wildfire-footprints-fill',
        'wildfire-footprints-outline',
        'wildfires-layer',
      ],
      perimeters: ['fire-perimeters-fill', 'fire-perimeters-outline'],
      incidents: ['ignis-incidents-layer', 'ignis-incidents-label', 'selected-incident-layer'],
      warnings: ['nws-alerts-fill', 'nws-alerts-outline'],
      prediction: [
        'ignis-pred-raster-layer',
        'ignis-pred-contour-line',
        'ignis-pred-contour50-line',
        'ignis-pred-bounds-layer',
        'ignis-pred-vector-fill',
        'ignis-pred-vector-line',
      ],
      ndvi: [NDVI_LAYER_ID],
    };
    Object.entries(groups).forEach(([key, layerIds]) => {
      const visible = visibility[key] !== false;
      layerIds.forEach(layerId => {
        if (map.getLayer(layerId)) {
          try { map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none'); } catch (_) {}
        }
      });
    });
  }

  function setObservedLayerMode(mode = 'normal') {
    const map = mapRef.current;
    if (!map) return;
    const isForecast = mode === 'forecast';
    const paint = [
      ['observed-fire-cells-fill', 'fill-opacity', isForecast ? OBSERVED_CELL_FILL_OPACITY_FORECAST : OBSERVED_CELL_FILL_OPACITY],
      ['observed-fire-cells-outline', 'line-opacity', isForecast ? 0.35 : 0.78],
      ['wildfire-footprints-fill', 'fill-opacity', isForecast ? FOOTPRINT_FILL_OPACITY_FORECAST : FOOTPRINT_FILL_OPACITY],
      ['wildfire-footprints-outline', 'line-opacity', isForecast ? 0.0 : 0.30],
      ['wildfires-layer', 'circle-opacity', isForecast ? POINT_CIRCLE_OPACITY_FORECAST : POINT_CIRCLE_OPACITY],
      ['fire-perimeters-fill', 'fill-opacity', isForecast ? PERIMETER_FILL_OPACITY_FORECAST : PERIMETER_FILL_OPACITY],
      ['fire-perimeters-outline', 'line-opacity', isForecast ? 0.72 : 0.90],
    ];
    paint.forEach(([layerId, property, value]) => {
      if (map.getLayer(layerId)) {
        try { map.setPaintProperty(layerId, property, value); } catch (_) {}
      }
    });
  }

  function setObservedLayersVisible(visible) {
    const map = mapRef.current;
    setObservedLayersVisibleState(Boolean(visible));
    if (!map) return;
    [
      'observed-fire-cells-fill',
      'observed-fire-cells-outline',
      'wildfire-footprints-fill',
      'wildfire-footprints-outline',
      'wildfires-layer',
      'fire-perimeters-fill',
      'fire-perimeters-outline',
    ].forEach(layerId => {
      if (map.getLayer(layerId)) {
        try { map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none'); } catch (_) {}
      }
    });
  }

  function updateUserSource() {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('user-source');
    if (!src) return;

    if (!userLocation) {
      src.setData({ type: 'FeatureCollection', features: [] });
      return;
    }
    src.setData({
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [userLocation.lng, userLocation.lat] },
        properties: {}
      }]
    });
    const z = map.getZoom();
    map.setCenter([userLocation.lng, userLocation.lat]);
    map.setZoom(z);
  }

  // ---------- NDVI overlay ----------
  async function ensureNdviLayer() {
    const map = mapRef.current;
    if (!map) return;

    if (!ndviTemplateRef.current) {
      try {
        const r = await fetch(`${API_BASE}/ndvi/tile`);
        const j = await r.json();
        ndviTemplateRef.current = j.template;
      } catch (e) {
        console.warn('NDVI template fetch failed', e);
        return;
      }
    }

    if (!map.getSource(NDVI_SOURCE_ID)) {
      map.addSource(NDVI_SOURCE_ID, {
        type: 'raster',
        tiles: [ndviTemplateRef.current],
        tileSize: 256,
        attribution: 'NASA GIBS MODIS NDVI (16-day)'
      });
    }
    if (!map.getLayer(NDVI_LAYER_ID)) {
      const before = map.getStyle()?.layers?.find(l => l.id === 'wildfires-layer') ? 'wildfires-layer' : undefined;
      map.addLayer({
        id: NDVI_LAYER_ID,
        type: 'raster',
        source: NDVI_SOURCE_ID,
        paint: { 'raster-opacity': 0.6 }
      }, before);
    }
  }

  const toggleNdvi = useCallback(async () => {
    const map = mapRef.current;
    if (!map) return;

    if (!ndviOn) {
      await ensureNdviLayer();
      setNdviOn(true);
      if (map.getLayer(NDVI_LAYER_ID)) {
        map.setLayoutProperty(NDVI_LAYER_ID, 'visibility', 'visible');
      }
    } else {
      if (map.getLayer(NDVI_LAYER_ID)) {
        map.setLayoutProperty(NDVI_LAYER_ID, 'visibility', 'none');
      }
      setNdviOn(false);
    }
  }, [ndviOn]);

  const clearForecastTimeline = useCallback(() => {
    setIsForecastPlaying(false);
    setForecastVisible(false);
    setForecastFrames([]);
    setForecastScene(null);
    setActiveForecastIndex(0);
    setForecastLayerMode('new_burn');
    removePredictionOverlays(mapRef.current);
    setObservedLayerMode('normal');
  }, []);

  const loadForecastTimeline = useCallback(async ({
    lat,
    lon,
    date,
    title = 'Ignis Forecast Timeline',
    steps = 6,
    stepHours,
    ignition,
    fitBounds = true,
  }) => {
    setIsForecastLoading(true);
    setForecastError(null);

    // Safety net: if anything in the pipeline (axios, image decode, map render)
    // hangs, force-clear the spinner after FORECAST_TIMEOUT_MS and surface an
    // error. Without this the UI can get stuck on "LOADING FORECAST…" forever
    // even after the network request completes.
    let timeoutHandle = null;
    const timeoutPromise = new Promise((_, reject) => {
      timeoutHandle = setTimeout(() => {
        console.warn('[forecast] safety-net timeout fired — pipeline stalled past FORECAST_TIMEOUT_MS');
        reject(new Error(`Forecast timed out after ${Math.round(FORECAST_TIMEOUT_MS / 1000)}s`));
      }, FORECAST_TIMEOUT_MS);
    });

    const work = (async () => {
      // Breadcrumbs so the console pinpoints which await is hanging. Keep
      // these until the forecast render pipeline is provably reliable —
      // silent hangs are the expensive bug here, not log noise.
      console.info('[forecast] 1/5 calling predictFireSpreadMultistep', { lat, lon, date, steps, stepHours, ignition });
      // Express proxy already retries upstream; don't double up from the
      // browser or tilesvc queues up redundant expensive predictions.
      const response = await predictFireSpreadMultistep({
        lat,
        lon,
        date,
        steps,
        ...(stepHours ? { stepHours } : {}),
        ...(ignition != null ? { ignition } : {}),
      });
      const payload = response?.data ?? response;
      console.info('[forecast] 2/5 HTTP response received', {
        stepCount: payload?.steps?.length,
        hasBounds: Array.isArray(payload?.bounds),
        firstStepHasImage: !!payload?.steps?.[0]?.image_base64,
      });

      const prepared = await prepareMultistepRasterFrames(payload, {
        opacity: 0.75,
        smooth: true,
      });
      console.info('[forecast] 3/5 frames prepared', {
        frames: prepared?.frames?.length,
        firstHeatmapUrlLen: prepared?.frames?.[0]?.heatmapUrl?.length,
      });

      const firstFrame = prepared.frames[0];
      setObservedLayerMode('forecast');
      console.info('[forecast] 4/5 rendering first frame on map', {
        hasMap: !!mapRef.current,
        mapLoaded: typeof mapRef.current?.loaded === 'function' ? mapRef.current.loaded() : undefined,
        styleLoaded: typeof mapRef.current?.isStyleLoaded === 'function' ? mapRef.current.isStyleLoaded() : undefined,
        bounds: firstFrame?.bounds,
        coordinateCount: Array.isArray(firstFrame?.coordinates) ? firstFrame.coordinates.length : 0,
      });
      await withTimeout(
        renderPredictionRasterFrame(mapRef.current, firstFrame, { layerMode: forecastLayerMode }),
        FORECAST_RENDER_TIMEOUT_MS,
        `Forecast map render timed out after ${Math.round(FORECAST_RENDER_TIMEOUT_MS / 1000)}s`,
      );
      console.info('[forecast] 4/5 first frame rendered on map');
      if (fitBounds && firstFrame?.bounds) {
        const [w, s, e, n] = firstFrame.bounds;
        mapRef.current?.fitBounds?.([[w, s], [e, n]], { padding: 40, duration: 800 });
      }

      setForecastTitle(title);
      setForecastScene(prepared.scene || null);
      setForecastFrames(prepared.frames);
      setActiveForecastIndex(0);
      setIsForecastPlaying(false);
      setForecastVisible(true);
      console.info('[forecast] 5/5 timeline state flushed, returning prepared');
      return prepared;
    })();

    try {
      return await Promise.race([work, timeoutPromise]);
    } finally {
      if (timeoutHandle) clearTimeout(timeoutHandle);
      setIsForecastLoading(false);
    }
  }, []);

  // ---------- Prediction ----------
  const handlePredictFireSpread = async (fireProps) => {
    // Synchronous check + set on the ref blocks a second click that fires
    // before React flushes the isPredicting state update from a first click.
    if (isPredictingRef.current) return;
    isPredictingRef.current = true;
    setIsPredicting(true);
    try {
      if (activePopup) activePopup.remove();
      clearForecastTimeline();

      const prepared = await loadForecastTimeline({
        lat: fireProps.latitude,
        lon: fireProps.longitude,
        title: 'Ignis Forecast Timeline',
        stepHours: DEFAULT_STEP_HOURS,
      });
      showPredictionPopup(fireProps, forecastSummaryFromPrepared(prepared));
    } catch (error) {
      console.error('Forecast timeline failed:', error);
      setObservedLayerMode('normal');
      setForecastError(error?.message || 'Forecast failed. Please try again.');
      const map = mapRef.current;
      if (map) {
        const popup = new mapboxgl.Popup()
          .setLngLat([fireProps.longitude, fireProps.latitude])
          .setHTML(`
            <div class="wildfire-popup">
              <h4>${getSeverityIcon(fireProps.brightnessCat)} ${fireProps.brightnessCat} Fire</h4>
              <p>Error predicting fire spread. Please try again.</p>
            </div>
          `)
          .addTo(map);
        setActivePopup(popup);
      }
    } finally {
      isPredictingRef.current = false;
      setIsPredicting(false);
    }
  };

  const runPredictionForIncident = useCallback(async (incident) => {
    if (!incident || isPredictingRef.current) return null;
    const lat = Number(incident.lat);
    const lon = Number(incident.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

    isPredictingRef.current = true;
    setIsPredicting(true);
    try {
      if (activePopup) activePopup.remove();
      clearForecastTimeline();
      const map = mapRef.current;
      const perimeterGeojson = await loadFirePerimeters({
        bbox: bboxAround(lon, lat, 1.25),
        incidentName: incident.name,
      });
      const perimeterBounds = featureCollectionBounds(perimeterGeojson);
      const perimeterCenter = centerFromBounds(perimeterBounds);
      const hotspotCenter = hotspotCenterForIncident(wildfires, incident);
      const targetCenter = perimeterCenter || hotspotCenter || { lat, lon };

      if (map && perimeterBounds) {
        map.fitBounds(
          [[perimeterBounds.w, perimeterBounds.s], [perimeterBounds.e, perimeterBounds.n]],
          {
            padding: watchShell
              ? { top: 86, right: 430, bottom: 90, left: 60 }
              : { top: 60, right: 60, bottom: 70, left: 60 },
            duration: 900,
          }
        );
      } else {
        map?.flyTo?.({
          center: [targetCenter.lon, targetCenter.lat],
          zoom: Math.max(map?.getZoom?.() || 0, 10),
          duration: 900,
        });
      }

      const useSyntheticIgnition = !(incident.hasHotspots || incident.hasPerimeter);
      return await loadForecastTimeline({
        lat: targetCenter.lat,
        lon: targetCenter.lon,
        title: `${incident.name || 'Incident'} Ignis Advisory`,
        stepHours: DEFAULT_STEP_HOURS,
        ignition: useSyntheticIgnition,
      });
    } catch (error) {
      console.error('Incident forecast failed:', error);
      setObservedLayerMode('normal');
      setForecastError(error?.message || 'Incident forecast failed. Please try again.');
      return null;
    } finally {
      isPredictingRef.current = false;
      setIsPredicting(false);
    }
  }, [activePopup, clearForecastTimeline, loadFirePerimeters, loadForecastTimeline, watchShell, wildfires]);

  const forecastSummaryFromPrepared = (prepared) => {
    const frames = Array.isArray(prepared?.frames) ? prepared.frames : [];
    const peakFrame = frames.reduce((best, frame) => {
      if (!best) return frame;
      return Number(frame?.prob_max ?? 0) > Number(best?.prob_max ?? 0) ? frame : best;
    }, null);
    const firstFrame = frames[0] || peakFrame || {};
    const meta = peakFrame?.meta || firstFrame?.meta || {};
    return {
      prob_max: peakFrame?.prob_max ?? firstFrame?.prob_max ?? meta.prob_max ?? 0,
      prob_mean: peakFrame?.prob_mean ?? firstFrame?.prob_mean ?? meta.prob_mean ?? 0,
      area_fraction: peakFrame?.area_fraction ?? firstFrame?.area_fraction ?? meta.area_fraction ?? 0,
      display_area_fraction: peakFrame?.display_area_fraction ?? firstFrame?.display_area_fraction ?? meta.display_area_fraction,
      threshold: meta.threshold ?? prepared?.threshold,
      display_floor: meta.display_floor ?? prepared?.displayFloor,
      step_hours: meta.step_hours ?? prepared?.stepHours,
      label: peakFrame?.label ?? firstFrame?.label,
      environmental_data: {
        data_source: 'model_input_audit',
      },
    };
  };

  const showPredictionPopup = (fireProps, prediction) => {
    const map = mapRef.current;
    if (!map) return;

    const env = prediction?.environmental_data || {};
    // Use prob_max for the headline "will it spread" assessment (max probability in the tile)
    const probMax = prediction?.prob_max ?? prediction?.spread_probability ?? 0;
    const probMean = prediction?.prob_mean ?? 0;
    const maxPct = Math.round(probMax * 100);
    const meanPct = (probMean * 100).toFixed(1);
    const threshold = Number(prediction?.threshold);
    const displayFloor = Number(prediction?.display_floor);
    const willSpread = Number.isFinite(threshold) && probMax >= threshold
      ? 'Yes'
      : probMax >= Math.max(Number.isFinite(displayFloor) ? displayFloor : 0.02, 0.15)
        ? 'Possible'
        : 'Unlikely';
    // Spread distance based on prob_max rather than area_fraction
    const spreadKm = prediction?.spread_distance_km ?? (64 * Math.sqrt(Math.max(0, probMax)));

    const popup = new mapboxgl.Popup()
      .setLngLat([fireProps.longitude, fireProps.latitude])
      .setHTML(`
        <div class="wildfire-popup">
          <h4>${getSeverityIcon(fireProps.brightnessCat)} ${fireProps.brightnessCat} Fire</h4>
          <div class="prediction-results">
            <h5>Ignis Advisory Risk</h5>
            <p style="font-size:0.88em;color:#666;">Modeled fire-spread risk, not an official perimeter.</p>
            <p><strong>Spread risk:</strong> ${willSpread}</p>
            <p><strong>Peak probability:</strong> ${pctStr(maxPct)}</p>
            <p><strong>Mean probability:</strong> ${meanPct}%</p>
            ${Number.isFinite(threshold) ? `<p><strong>Decision threshold:</strong> ${(threshold * 100).toFixed(0)}%</p>` : ''}
            ${Number.isFinite(displayFloor) ? `<p><strong>Display floor:</strong> ${(displayFloor * 100).toFixed(0)}%</p>` : ''}
            <p><strong>Est. spread distance:</strong> ${fmt(spreadKm, 1)} km</p>
            <div class="env-data">
              <h6>Environmental Factors</h6>
              <p>Wind: ${fmt(env.wind_speed, 1)} km/h ${getDirectionName(env.wind_direction)}</p>
              <p>Temp: ${fmt(env.temperature, 1)}°C, Humidity: ${fmt(env.humidity, 0)}%</p>
              <p>Data source: ${env.data_source || '—'}</p>
            </div>
          </div>
        </div>
      `)
      .addTo(map);

    setActivePopup(popup);

    // Optional enrichment: live topo + weather details from your backend
    (async () => {
      try {
        const lat = fireProps.latitude;
        const lon = fireProps.longitude;

        const [topoRes, wxRes] = await Promise.all([
          fetch(`${API_BASE}/topography/point?lat=${lat}&lon=${lon}`),
          fetch(`${API_BASE}/weather/current?lat=${lat}&lon=${lon}`)
        ]);
        const { data: tp } = topoRes.ok ? await topoRes.json() : { data: null };
        const wxJSON = wxRes.ok ? await wxRes.json() : null;
        const wx = wxJSON?.data?.current;

        const windMph = msToMph(wx?.wind_speed_10m);
        const gustMph = msToMph(wx?.wind_gusts_10m);

        const extra = `
          <div class="prediction-results">
            <h5>Real-Time Context</h5>
            <div class="env-data">
              <p><strong>Weather</strong></p>
              <p>Temp: ${wx?.temperature_2m ?? '—'} °C</p>
              <p>RH: ${wx?.relative_humidity_2m ?? '—'}%</p>
              <p>Wind: ${windMph != null ? windMph.toFixed(1) : '—'} mph
                 ${gustMph != null ? `(gust ${gustMph.toFixed(1)} mph)` : ''}</p>

              <p style="margin-top:8px;"><strong>Topography</strong></p>
              <p>Elevation: ${tp?.elevation != null ? tp.elevation.toFixed(0) : '—'} m</p>
              <p>Slope: ${tp?.slope_deg != null ? tp.slope_deg.toFixed(1) : '—'}°</p>
              <p>Aspect: ${tp?.aspect_deg != null ? `${tp.aspect_deg.toFixed(0)}° (${toCompass(tp.aspect_deg)})` : '—'}</p>
            </div>
          </div>
        `;
        const node = document.createElement('div');
        node.innerHTML = extra;
        const popupEl = document.querySelector('.mapboxgl-popup-content .wildfire-popup');
        if (popupEl) popupEl.appendChild(node);
      } catch (e) {
        console.warn('context fetch failed', e);
      }
    })();
  };

  const addIgnisOverlayAt = async ({ latitude, longitude }, mode = 'raster') => {
    try {
        const map = mapRef.current;
      if (!map) return;

      const result = await addPredictionOverlay(map, {
        apiBase: API_BASE,
        lat: latitude,
      lon: longitude,
      mode,
      gamma: 0.7,
      opacity: 0.75,
      smooth: true,
      alphaThreshold: 0.01,
      });

      // Fit to tile bounds so overlay doesn’t look “random”
      if (result?.bounds) {
        const [w, s, e, n] = result.bounds;
        map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 800 });
      }
    } catch (e) {
      console.error('Ignis overlay error:', e);
    }
  };

  // ---------- Historical fire test ----------
  const runHistoricalPrediction = async (preset) => {
    const map = mapRef.current;
    // Synchronous check + set on the ref blocks a second click that fires
    // before React flushes the histRunning state update from a first click.
    if (!map || histRunningRef.current) return;
    histRunningRef.current = true;
    setHistRunning(true);
    try {
      const { lat, lon, date, zoom, name, incidentName, displayDate, ignition } = preset;
      const usesIgnitionPoint = ignition == null ? true : Boolean(ignition);
      const historicalInputNote = usesIgnitionPoint
        ? 'Uses historical date inputs; this preset includes an ignition-point seed when persisted fire history is unavailable.'
        : 'Uses cached historical FIRMS/weather inputs for this incident; no synthetic ignition-point seed is added.';

      if (activePopup) activePopup.remove();
      map.flyTo?.({ center: [lon, lat], zoom: zoom || 10, duration: 1200 });
      clearForecastTimeline();
      await loadFirePerimeters({
        bbox: bboxAround(lon, lat, 1.5),
        incidentName: incidentName || name.replace(/\s+Fire$/i, ''),
        from: date,
        to: addDaysIso(date, 8),
      });

      const prepared = await loadForecastTimeline({
        lat,
        lon,
        date,
        title: `${name} Forecast Timeline`,
        stepHours: DEFAULT_STEP_HOURS,
        ignition: usesIgnitionPoint,
      });

      // Show info popup
      const firstFrame = prepared?.frames?.[0];
      const probMax = firstFrame?.prob_max;
      const probMean = firstFrame?.prob_mean;
      const cadence = prepared?.stepHours;
      const popup = new mapboxgl.Popup()
        .setLngLat([lon, lat])
        .setHTML(`
          <div class="wildfire-popup">
            <h4>${name}</h4>
            <div class="prediction-results">
              <h5>Historical Prediction (${displayDate || date})</h5>
              <p><strong>Max probability:</strong> ${probMax != null ? (probMax * 100).toFixed(1) + '%' : '--'}</p>
              <p><strong>Mean probability:</strong> ${probMean != null ? (probMean * 100).toFixed(2) + '%' : '--'}</p>
              <p><strong>Timeline:</strong> every ${cadence || '--'} hours</p>
              <p><strong>Observed overlay:</strong> FIRIS/WFIGS perimeter where available, plus FIRMS footprints.</p>
              <p style="margin-top:8px;font-size:0.85em;color:#888;">
                ${historicalInputNote}
              </p>
            </div>
          </div>
        `)
        .addTo(map);
      setActivePopup(popup);
    } catch (e) {
      console.error('Historical prediction error:', e);
      setObservedLayerMode('normal');
      setForecastError(e?.message || 'Historical forecast failed. Please try again.');
    } finally {
      histRunningRef.current = false;
      setHistRunning(false);
    }
  };

  const runCustomHistorical = async () => {
    const map = mapRef.current;
    if (!map || !histDate) return;
    const center = map.getCenter();
    await runHistoricalPrediction({
      name: `Custom (${histDate})`,
      date: histDate,
      lat: center.lat,
      lon: center.lng,
      zoom: map.getZoom(),
    });
  };

  const flyToIncident = useCallback((incident) => {
    const map = mapRef.current;
    const lat = Number(incident?.lat);
    const lon = Number(incident?.lon);
    if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
    map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom?.() || 0, 9.5), duration: 900 });
  }, []);

  const flyToAlert = useCallback((alert) => {
    const map = mapRef.current;
    if (!map || !alert?.geometry) return;
    const bounds = featureCoordinateBounds(alert.geometry);
    if (!bounds) return;
    map.fitBounds([[bounds.w, bounds.s], [bounds.e, bounds.n]], { padding: 60, duration: 900, maxZoom: 8 });
  }, []);

  // ---------- Expose actions to parent ----------
  useImperativeHandle(ref, () => ({
    refreshWildfires: fetchWildfires,
    toggleNdvi,
    toggleHistoryPanel: () => setShowHistPanel(v => !v),
    runPredictionForIncident,
    flyToIncident,
    flyToAlert,
  }), [fetchWildfires, toggleNdvi, runPredictionForIncident, flyToIncident, flyToAlert]);

  // ---------- Layers setup ----------
  function setupLayers() {
    const map = mapRef.current;
    if (!map) return;

    // Observed fire shape: small per-detection cells are primary; raw scan/track footprints stay subdued.
    [
      'selected-incident-layer',
      'ignis-incidents-label',
      'ignis-incidents-layer',
      'nws-alerts-outline',
      'nws-alerts-fill',
      'wildfires-layer',
      'observed-fire-cells-outline',
      'observed-fire-cells-fill',
      'wildfire-footprints-outline',
      'wildfire-footprints-fill',
      'fire-perimeters-outline',
      'fire-perimeters-fill',
    ].forEach(id => {
      if (map.getLayer(id)) map.removeLayer(id);
    });
    [
      'selected-incident-source',
      'ignis-incidents-source',
      'nws-alerts-source',
      'wildfires-source',
      'observed-fire-cells-source',
      'wildfire-footprints-source',
      'fire-perimeters-source',
    ].forEach(id => {
      if (map.getSource(id)) map.removeSource(id);
    });

    map.addSource('nws-alerts-source', {
      type: 'geojson',
      data: emptyFeatureCollection(),
    });
    map.addLayer({
      id: 'nws-alerts-fill',
      type: 'fill',
      source: 'nws-alerts-source',
      paint: {
        'fill-color': '#aa6095',
        'fill-opacity': 0.16,
      }
    });
    map.addLayer({
      id: 'nws-alerts-outline',
      type: 'line',
      source: 'nws-alerts-source',
      paint: {
        'line-color': '#aa6095',
        'line-width': 1.8,
        'line-dasharray': [2, 1.5],
        'line-opacity': 0.78,
      }
    });

    map.addSource('observed-fire-cells-source', {
      type: 'geojson',
      data: emptyFeatureCollection(),
    });
    map.addLayer({
      id: 'observed-fire-cells-fill',
      type: 'fill',
      source: 'observed-fire-cells-source',
      paint: {
        'fill-color': FOOTPRINT_FILL_COLOR,
        'fill-opacity': OBSERVED_CELL_FILL_OPACITY,
      }
    });
    map.addLayer({
      id: 'observed-fire-cells-outline',
      type: 'line',
      source: 'observed-fire-cells-source',
      paint: {
        'line-color': [
          'match', ['get', 'brightnessCat'],
          'Extreme', '#fff1cc',
          'Severe', '#ffd6b0',
          'Moderate', '#ffe6a4',
          '#fff6cc'
        ],
        'line-width': [
          'interpolate', ['linear'], ['zoom'],
          4, 0.25,
          8, 0.55,
          12, 1.0
        ],
        'line-opacity': 0.78,
      }
    });

    map.addSource('wildfire-footprints-source', {
      type: 'geojson',
      data: emptyFeatureCollection(),
    });
    map.addLayer({
      id: 'wildfire-footprints-fill',
      type: 'fill',
      source: 'wildfire-footprints-source',
      paint: {
        'fill-color': FOOTPRINT_FILL_COLOR,
        'fill-opacity': FOOTPRINT_FILL_OPACITY,
      }
    });
    map.addLayer({
      id: 'wildfire-footprints-outline',
      type: 'line',
      source: 'wildfire-footprints-source',
      paint: {
        'line-color': FOOTPRINT_LINE_COLOR,
        'line-width': FOOTPRINT_LINE_WIDTH,
        'line-opacity': 0.30,
      }
    });

    map.addSource('fire-perimeters-source', {
      type: 'geojson',
      data: emptyFeatureCollection(),
    });
    map.addLayer({
      id: 'fire-perimeters-fill',
      type: 'fill',
      source: 'fire-perimeters-source',
      paint: {
        'fill-color': '#d65a31',
        'fill-opacity': PERIMETER_FILL_OPACITY,
      }
    });
    map.addLayer({
      id: 'fire-perimeters-outline',
      type: 'line',
      source: 'fire-perimeters-source',
      paint: {
        'line-color': '#e16a38',
        'line-width': 2.2,
        'line-opacity': 0.9,
      }
    });

    map.addSource('wildfires-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
      id: 'wildfires-layer',
      type: 'circle',
      source: 'wildfires-source',
      paint: {
        'circle-radius': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 5,
          'Severe', 4,
          'Moderate', 3.5,
          3
        ],
        'circle-color': [
          'match', ['get', 'brightnessCat'],
          'Extreme', '#d94b21',
          'Severe',  '#ff6b35',
          'Moderate', '#ff9440',
          '#ffc766'
        ],
        'circle-opacity': POINT_CIRCLE_OPACITY,
        'circle-stroke-width': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 0.9,
          'Severe', 0.7,
          0.5
        ],
        'circle-stroke-color': '#fff2d6',
        'circle-blur': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 0.92,
          0.52
        ]
      }
    });

    map.addSource('ignis-incidents-source', { type: 'geojson', data: emptyFeatureCollection() });
    map.addLayer({
      id: 'ignis-incidents-layer',
      type: 'circle',
      source: 'ignis-incidents-source',
      paint: {
        'circle-radius': [
          'interpolate', ['linear'], ['coalesce', ['get', 'acres'], 0],
          0, 7,
          100, 10,
          1000, 14,
          10000, 20,
        ],
        'circle-color': [
          'case',
          ['!=', ['get', 'status'], 'active'], '#8f949b',
          ['>=', ['coalesce', ['get', 'acres'], 0], 1000], '#d44a22',
          '#ff7d3e',
        ],
        'circle-opacity': 0.92,
        'circle-stroke-width': 2.2,
        'circle-stroke-color': '#fff0d1',
        'circle-blur': [
          'case',
          ['!=', ['get', 'status'], 'active'], 0.15,
          0.45,
        ],
      }
    });
    map.addLayer({
      id: 'ignis-incidents-label',
      type: 'symbol',
      source: 'ignis-incidents-source',
      minzoom: 5,
      layout: {
        'text-field': ['get', 'name'],
        'text-size': 12,
        'text-offset': [0, 1.25],
        'text-anchor': 'top',
        'text-allow-overlap': false,
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#1b1f24',
        'text-halo-width': 1.4,
      }
    });

    map.addSource('selected-incident-source', { type: 'geojson', data: emptyFeatureCollection() });
    map.addLayer({
      id: 'selected-incident-layer',
      type: 'circle',
      source: 'selected-incident-source',
      paint: {
        'circle-radius': 18,
        'circle-color': 'rgba(0,0,0,0)',
        'circle-stroke-color': '#ffd8a6',
        'circle-stroke-width': 3,
        'circle-opacity': 1,
      }
    });

    // Click handler (re-bind safely)
    if (clickHandlerRef.current) {
      ['observed-fire-cells-fill', 'wildfire-footprints-fill', 'wildfires-layer'].forEach(layerId => {
        try { map.off('click', layerId, clickHandlerRef.current); } catch (_) {}
      });
    }
    if (perimeterClickHandlerRef.current) {
      ['fire-perimeters-fill', 'fire-perimeters-outline'].forEach(layerId => {
        try { map.off('click', layerId, perimeterClickHandlerRef.current); } catch (_) {}
      });
    }
    if (incidentClickHandlerRef.current) {
      try { map.off('click', 'ignis-incidents-layer', incidentClickHandlerRef.current); } catch (_) {}
    }
    if (alertClickHandlerRef.current) {
      ['nws-alerts-fill', 'nws-alerts-outline'].forEach(layerId => {
        try { map.off('click', layerId, alertClickHandlerRef.current); } catch (_) {}
      });
    }
    clickHandlerRef.current = async e => {
      const f = e.features?.[0];
      if (!f) return;

      const p = f.properties || {};
      const pointCoords = Array.isArray(f.geometry?.coordinates) && f.geometry?.type === 'Point'
        ? f.geometry.coordinates
        : null;
      const lon = Number(p.longitude ?? pointCoords?.[0]);
      const lat = Number(p.latitude ?? pointCoords?.[1]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const coords = [lon, lat];
      const timeStr = p.timestamp ? new Date(p.timestamp).toLocaleString() : 'Unknown';
      const addr   = await reverseGeocode(lat, lon, mapboxgl.accessToken);
      const icon   = getSeverityIcon(p.brightnessCat);
      const confPct = Number(p.confidencePct ?? 0);
      const canPredict = String(p.predictable) === 'true' || p.predictable === true;

      if (activePopup) activePopup.remove();

      const fireProps = {
        brightnessCat: p.brightnessCat,
        brightness: Number(p.brightness),
        confidencePct: confPct,
        frp: Number(p.frp),
        latitude: lat,
        longitude: lon
      };

      const popup = new mapboxgl.Popup()
        .setLngLat(coords)
        .setHTML(`
          <div class="wildfire-popup">
            <h4>${icon} ${p.brightnessCat} Fire</h4>
            <p><strong>Address:</strong> ${addr}</p>
            <p><strong>Confidence:</strong> ${pctStr(confPct)}</p>
            ${p.frp != null && p.frp !== '' ? `<p><strong>FRP:</strong> ${fmt(p.frp, 1)} MW</p>` : ''}
            ${p.product ? `<p><strong>Product:</strong> ${p.product}</p>` : ''}
            ${p.scan && p.track ? `<p><strong>Pixel:</strong> ${fmt(p.scan, 2)} km x ${fmt(p.track, 2)} km</p>` : ''}
            <p><strong>Captured at:</strong> ${timeStr}</p>
            ${canPredict ? `
              <button id="predict-spread-btn" class="predict-spread-btn">
                ${isPredicting ? 'Predicting...' : 'Predict (Ignis model)'}
              </button>
            ` : `
              <div style="margin:8px 0;color:#777;">
                Not enough signal for a reliable prediction (low brightness/confidence).
              </div>
            `}
            <button id="ignis-overlay-raster" class="predict-spread-btn" style="margin-top:8px;background:#444;">
              Add Ignis Overlay (raster)
            </button>
            <button id="ignis-overlay-vector" class="predict-spread-btn" style="margin-top:8px;background:#222;">
              Add Ignis Overlay (vector)
            </button>
          </div>
        `)
        .addTo(map);

      setActivePopup(popup);

      const btn = document.getElementById('predict-spread-btn');
      if (btn && canPredict) btn.addEventListener('click', () => handlePredictFireSpread(fireProps));
      document.getElementById('ignis-overlay-raster')?.addEventListener('click', () => addIgnisOverlayAt(fireProps, 'raster'));
      document.getElementById('ignis-overlay-vector')?.addEventListener('click', () => addIgnisOverlayAt(fireProps, 'vector'));
    };
    perimeterClickHandlerRef.current = e => {
      const f = e.features?.[0];
      if (!f) return;
      const p = f.properties || {};
      const center = map.getCenter?.();
      const coords = e.lngLat
        ? [e.lngLat.lng, e.lngLat.lat]
        : [center?.lng ?? center?.lon ?? -98, center?.lat ?? 38];
      const name = p.poly_IncidentName || p.attr_IncidentName || p.IncidentName || p.FIRE_NAME || p.FireName || p.incidentName || 'Fire perimeter';
      const source = p.poly_Source || p.Source || p.source || (p.irwin_InitialLatitude ? 'WFIGS/FIRIS' : 'Authoritative perimeter');
      const date = p.poly_CreateDate || p.attr_ModifiedOnDateTime || p.ModifiedOnDateTime || p.CreateDate || p.date || p.perimeterDate;
      if (activePopup) activePopup.remove();
      const popup = new mapboxgl.Popup()
        .setLngLat(coords)
        .setHTML(`
          <div class="wildfire-popup">
            <h4>Observed Fire Perimeter</h4>
            <p><strong>Incident:</strong> ${name}</p>
            <p><strong>Source:</strong> ${source}</p>
            ${date ? `<p><strong>Observed at:</strong> ${new Date(date).toLocaleString()}</p>` : ''}
            <p style="margin-top:8px;color:#666;font-size:0.9em;">
              This is an observed perimeter overlay, not the Ignis forecast.
            </p>
          </div>
        `)
        .addTo(map);
      setActivePopup(popup);
    };
    incidentClickHandlerRef.current = e => {
      const f = e.features?.[0];
      if (!f) return;
      const id = f.properties?.id;
      const incident = incidentsRef.current.find(item => item.id === id) || f.properties;
      onIncidentSelect?.(incident);
    };
    alertClickHandlerRef.current = e => {
      const f = e.features?.[0];
      if (!f) return;
      const id = f.properties?.id;
      const alert = alertsRef.current.find(item => item.id === id) || {
        id,
        event: f.properties?.event,
        headline: f.properties?.headline,
        areaDesc: f.properties?.areaDesc,
        geometry: f.geometry,
      };
      onAlertSelect?.(alert);
    };
    map.on('click', 'observed-fire-cells-fill', clickHandlerRef.current);
    map.on('click', 'wildfire-footprints-fill', clickHandlerRef.current);
    map.on('click', 'wildfires-layer', clickHandlerRef.current);
    map.on('click', 'fire-perimeters-fill', perimeterClickHandlerRef.current);
    map.on('click', 'fire-perimeters-outline', perimeterClickHandlerRef.current);
    map.on('click', 'ignis-incidents-layer', incidentClickHandlerRef.current);
    map.on('click', 'nws-alerts-fill', alertClickHandlerRef.current);
    map.on('click', 'nws-alerts-outline', alertClickHandlerRef.current);
    [
      'observed-fire-cells-fill',
      'wildfire-footprints-fill',
      'wildfires-layer',
      'fire-perimeters-fill',
      'fire-perimeters-outline',
      'ignis-incidents-layer',
      'nws-alerts-fill',
      'nws-alerts-outline',
    ].forEach(layerId => {
      map.on('mouseenter', layerId, () => {
        const canvas = map.getCanvas?.();
        if (canvas?.style) canvas.style.cursor = 'pointer';
      });
      map.on('mouseleave', layerId, () => {
        const canvas = map.getCanvas?.();
        if (canvas?.style) canvas.style.cursor = '';
      });
    });

    // User marker
    if (map.getLayer('user-layer'))  map.removeLayer('user-layer');
    if (map.getSource('user-source')) map.removeSource('user-source');
    map.addSource('user-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
      id: 'user-layer',
      type: 'circle',
      source: 'user-source',
      paint: {
        'circle-radius':       6,
        'circle-color':        '#1E90FF',
        'circle-stroke-width': 2,
        'circle-stroke-color': '#fff'
      }
    });

    if (ndviOn) ensureNdviLayer();
    updateIncidentSource();
    updateAlertSource();
    applyLayerVisibility();
  }

  // ---------- Lifecycle ----------
  useEffect(() => {
    const m = new mapboxgl.Map({
      container: mapContainerRef.current,
      style:     mapStyle,
      center:    [-98, 38],
      zoom:      4,
      maxBounds: [[-130,22],[-66,50]]
    });
    m.addControl(new mapboxgl.NavigationControl());
    mapRef.current = m;

    m.on('load', () => {
      setupLayers();
      fetchWildfires();
      updateWildfireSource();
      updateObservedFireCellsSource();
      updateWildfireFootprintsSource();
      updatePerimeterSource();
      updateIncidentSource();
      updateAlertSource();
      updateSelectedIncidentSource();
      applyLayerVisibility();
      updateUserSource();
    });
    return () => m.remove();
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.setStyle(mapStyle);
    map.once('styledata', () => {
      setupLayers();
      updateWildfireSource();
      updateObservedFireCellsSource();
      updateWildfireFootprintsSource();
      updatePerimeterSource();
      updateIncidentSource();
      updateAlertSource();
      updateSelectedIncidentSource();
      updateUserSource();
      setObservedLayerMode(forecastVisible ? 'forecast' : 'normal');
      setObservedLayersVisible(observedLayersVisible);
      applyLayerVisibility();
      if (forecastVisible && forecastFrames[activeForecastIndex]) {
        renderActiveSpread(map, forecastFrames[activeForecastIndex]);
      }
    });
  }, [mapStyle]);

  useEffect(() => { updateWildfireSource(); }, [wildfires, brightnessFilter, confidenceFilter]);
  useEffect(() => { updateObservedFireCellsSource(); }, [wildfires, brightnessFilter, confidenceFilter]);
  useEffect(() => { updateWildfireFootprintsSource(); }, [wildfireFootprints, brightnessFilter, confidenceFilter]);
  useEffect(() => { updatePerimeterSource(); }, [firePerimeters]);
  useEffect(() => { updateUserSource(); }, [userLocation]);
  useEffect(() => {
    incidentsRef.current = incidents;
    updateIncidentSource(incidents);
  }, [incidents]);
  useEffect(() => {
    alertsRef.current = alerts;
    updateAlertSource(alerts);
  }, [alerts]);
  useEffect(() => {
    updateSelectedIncidentSource(selectedIncident);
    if (selectedIncident) flyToIncident(selectedIncident);
  }, [selectedIncident, flyToIncident]);
  useEffect(() => {
    if (selectedAlert) flyToAlert(selectedAlert);
  }, [selectedAlert, flyToAlert]);
  useEffect(() => {
    applyLayerVisibility(layerVisibility);
  }, [layerVisibility, ndviOn, forecastVisible, forecastFrames.length]);

  const renderActiveSpread = useCallback((map, frame) => {
    if (!map || !frame) return;
    if (spreadView === 'bands' && forecastScene?.forecast?.features?.length) {
      removePredictionRaster(map);
      renderPredictionScene(map, forecastScene, { activeLeadHours: frame.lead_hours }).catch(err => {
        console.error('Forecast scene render error:', err);
      });
      return;
    }
    removePredictionScene(map);
    renderPredictionRasterFrame(map, frame, { layerMode: forecastLayerMode }).catch(err => {
      console.error('Forecast frame render error:', err);
    });
  }, [spreadView, forecastScene, forecastLayerMode]);

  useEffect(() => {
    if (!forecastVisible || !forecastFrames.length) return;
    renderActiveSpread(mapRef.current, forecastFrames[activeForecastIndex]);
  }, [forecastVisible, forecastFrames, activeForecastIndex, renderActiveSpread]);

  useEffect(() => {
    if (!forecastVisible || !isForecastPlaying || forecastFrames.length < 2) return undefined;
    const id = window.setInterval(() => {
      setActiveForecastIndex(idx => (idx + 1) % forecastFrames.length);
    }, 1200);
    return () => window.clearInterval(id);
  }, [forecastVisible, isForecastPlaying, forecastFrames.length]);

  useEffect(() => {
    const onKey = e => { if (e.key.toLowerCase() === 'n') toggleNdvi(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggleNdvi]);

  // Nearby fires panel feed
  useEffect(() => {
    if (!onNearbyFiresUpdate) return;
    const t = setTimeout(async () => {
      if (!userLocation || wildfires.length === 0 || range <= 0) {
        onNearbyFiresUpdate([]);
        return;
      }
      const inRange = wildfires
        .map(f => {
          const d = haversineDistance(userLocation.lat, userLocation.lng, f.latitude, f.longitude);
          return {
            ...f,
            distance: d,
            brightnessCat: getBrightnessCategory(f.brightness),
            confidencePct: confidenceToPercent(f.confidence)
          };
        })
        .filter(f => f.distance <= range);

      const enriched = await Promise.all(
        inRange.map(async f => {
          const confCat = bracketConfidence(f.confidencePct);
          const cityName = await reverseGeocode(f.latitude, f.longitude, mapboxgl.accessToken);
          return { cityName, distance: f.distance, brightnessCat: f.brightnessCat, confidenceCat: confCat };
        })
      );
      onNearbyFiresUpdate(enriched);
    }, 500);
    return () => clearTimeout(t);
  }, [userLocation, range, wildfires, onNearbyFiresUpdate]);

  // Popup button CSS
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .predict-spread-btn {
        background: linear-gradient(135deg, #ff8d4d 0%, #ff6b35 52%, #e35424 100%);
        color: #1a120d;
        padding: 8px 12px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 10px;
        width: 100%;
        font-weight: bold;
        box-shadow: 0 10px 22px rgba(227, 84, 36, 0.26);
      }
      .predict-spread-btn:hover { filter: brightness(1.04); }
      .prediction-results { margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; }
      .prediction-results h5 { margin-top: 0; margin-bottom: 8px; color: #FF4500; }
      .env-data { margin-top: 8px; padding: 8px; background-color: #f8f8f8; border-radius: 4px; font-size: 0.9em; }
      .env-data h6 { margin: 0 0 5px 0; color: #666; }
      .env-data p { margin: 3px 0; }
      .hist-panel {
        position: absolute; top: 10px; right: 50px; z-index: 10;
        background: rgba(30,30,30,0.92); color: #eee;
        border-radius: 8px; padding: 12px 14px; min-width: 220px;
        font-size: 13px; box-shadow: 0 2px 12px rgba(0,0,0,0.4);
      }
      .hist-panel h4 { margin: 0 0 8px; font-size: 14px; color: #FF6600; }
      .hist-btn {
        display: block; width: 100%; padding: 6px 8px; margin: 4px 0;
        border: none; border-radius: 4px; cursor: pointer;
        font-size: 12px; font-weight: 600; text-align: left;
        background: #444; color: #fff;
      }
      .hist-btn:hover { background: #666; }
      .hist-btn:disabled { opacity: 0.5; cursor: wait; }
      .hist-toggle {
        position: absolute; top: 10px; right: 10px; z-index: 10;
        background: #FF6600; color: #fff; border: none; border-radius: 6px;
        padding: 6px 10px; cursor: pointer; font-size: 13px; font-weight: 700;
      }
      .hist-toggle:hover { background: #FF8800; }
      .forecast-panel {
        position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%);
        z-index: 12; min-width: 360px; max-width: calc(100% - 32px);
        background: rgba(22, 24, 27, 0.94); color: #f7f3ea;
        border-radius: 8px; padding: 14px 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .watch-shell .forecast-panel {
        left: 20px; right: auto; bottom: 18px; transform: none;
        width: min(370px, calc(100% - 440px)); min-width: 0; max-width: 370px;
        background: rgba(21, 15, 12, 0.92);
        border-radius: 14px;
        padding: 14px 16px 12px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.34);
        border: 1px solid rgba(255, 140, 92, 0.16);
      }
      .forecast-header {
        display: flex; align-items: center; justify-content: space-between; gap: 12px;
        margin-bottom: 10px;
      }
      .forecast-title {
        font-size: 13px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        color: #ffb347;
      }
      .forecast-context {
        font-size: 11px; color: rgba(247,243,234,0.64); margin-top: 4px;
      }
      .watch-shell .forecast-context {
        color: rgba(248, 231, 214, 0.58);
      }
      .forecast-label {
        font-size: 20px; font-weight: 700; margin: 2px 0 0;
      }
      .watch-shell .forecast-label {
        font-size: 17px;
      }
      .forecast-meta {
        font-size: 12px; color: rgba(247,243,234,0.72); margin-bottom: 12px;
      }
      .watch-shell .forecast-meta {
        margin-bottom: 9px;
      }
      .forecast-ramp {
        height: 8px; border-radius: 999px; margin: 8px 0 6px;
        background: linear-gradient(90deg, transparent 0%, #401051 18%, #bf2a1d 42%, #ff7a1a 66%, #ffe66d 100%);
        border: 1px solid rgba(255,255,255,0.16);
      }
      .forecast-metrics {
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px;
        margin-bottom: 10px;
      }
      .watch-shell .forecast-metrics {
        gap: 5px;
        margin-bottom: 8px;
      }
      .forecast-metric {
        min-width: 0; padding: 7px 8px; border-radius: 8px;
        background: rgba(255,255,255,0.06); font-size: 11px;
      }
      .watch-shell .forecast-metric {
        padding: 7px 7px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
      }
      .forecast-metric strong {
        display: block; margin-bottom: 2px; color: #fff; font-size: 12px;
      }
      .forecast-status {
        font-size: 11px; color: rgba(247,243,234,0.72); margin-bottom: 10px;
      }
      .watch-shell .forecast-status {
        margin-bottom: 8px;
      }
      .forecast-status.warn { color: #ffcf70; }
      .forecast-status.unavailable { color: #ff6b6b; }
      .forecast-advisory {
        font-size: 12px; line-height: 1.35; color: rgba(247,243,234,0.78);
        margin: 4px 0 10px;
      }
      .watch-shell .forecast-advisory {
        margin: 3px 0 8px;
        font-size: 11.5px;
        color: rgba(248, 231, 214, 0.76);
      }
      .forecast-badges {
        display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px;
      }
      .watch-shell .forecast-badges,
      .watch-shell .forecast-layer-controls {
        gap: 5px;
        margin-bottom: 8px;
      }
      .forecast-badge {
        font-size: 11px; font-weight: 700; border-radius: 8px;
        padding: 4px 8px; background: rgba(255,255,255,0.08); color: #f7f3ea;
        border: 1px solid rgba(255,255,255,0.12);
      }
      .forecast-badge.warn { color: #111; background: #ffcf70; border-color: transparent; }
      .forecast-badge.good { color: #111; background: #91f0b7; border-color: transparent; }
      .forecast-layer-controls {
        display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 10px;
      }
      .forecast-layer-btn {
        border: 1px solid rgba(255,255,255,0.14); border-radius: 8px;
        background: transparent; color: #f7f3ea; padding: 6px 9px;
        font-size: 11px; font-weight: 700; cursor: pointer;
      }
      .forecast-layer-btn.active { background: #ff7b2f; color: #111; border-color: #ff7b2f; }
      .forecast-layer-btn:disabled { opacity: 0.35; cursor: not-allowed; }
      .forecast-view-controls {
        display: flex; gap: 6px; margin: 0 0 8px;
      }
      .forecast-band-key {
        list-style: none; display: flex; flex-wrap: wrap; gap: 4px 10px;
        margin: 0 0 10px; padding: 0;
      }
      .forecast-band-key li {
        display: flex; align-items: center; gap: 5px;
        font-size: 11px; font-weight: 600; color: rgba(247,243,234,0.6);
      }
      .forecast-band-key li.current { color: #f7f3ea; }
      .forecast-band-key .swatch {
        width: 13px; height: 13px; border-radius: 3px;
        border: 1px solid rgba(0,0,0,0.35);
      }
      .forecast-band-key li.current .swatch {
        box-shadow: 0 0 0 2px rgba(247,243,234,0.85);
      }
      .forecast-metric em {
        display: block; font-style: normal; font-size: 10px;
        color: rgba(247,243,234,0.45); margin-top: 2px;
      }
      .forecast-slider {
        width: 100%;
        accent-color: #ff7b2f;
      }
      .forecast-controls {
        display: flex; align-items: center; justify-content: space-between; gap: 8px;
        margin-top: 12px;
      }
      .watch-shell .forecast-controls {
        margin-top: 10px;
      }
      .forecast-btn {
        border: none; border-radius: 8px; cursor: pointer; font-weight: 700;
        padding: 8px 12px; background: #2e343b; color: #fff;
      }
      .forecast-btn:hover { background: #454c54; }
      .forecast-btn.primary { background: #ff7b2f; color: #111; }
      .forecast-btn.primary:hover { background: #ff944f; }
      .forecast-btn.ghost { background: transparent; border: 1px solid rgba(255,255,255,0.16); }
      .forecast-loading-bar {
        height: 4px; border-radius: 2px; margin-top: 14px; overflow: hidden;
        background: #2e343b;
      }
      .forecast-loading-bar::after {
        content: ''; display: block; height: 100%;
        background: linear-gradient(90deg, transparent 0%, #ff7b2f 40%, #ffb347 60%, transparent 100%);
        background-size: 200% 100%;
        animation: forecast-shimmer 1.4s ease-in-out infinite;
      }
      @keyframes forecast-shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      .forecast-error-msg { color: #ff6b6b; font-size: 12px; margin-top: 8px; }
      @media (max-width: 1240px) {
        .watch-shell .forecast-panel {
          width: min(340px, calc(100% - 404px));
          max-width: 340px;
        }
      }
      @media (max-width: 520px) {
        .forecast-panel { min-width: 0; width: calc(100% - 24px); bottom: 12px; padding: 12px; }
        .forecast-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .watch-shell .forecast-panel {
          left: 12px; right: 12px; bottom: 12px; width: auto; max-width: none;
        }
      }
    `;
    document.head.appendChild(style);
    return () => { document.head.removeChild(style); };
  }, []);

  const activeForecastFrame = forecastFrames[activeForecastIndex] || null;
  const activeForecastMeta = activeForecastFrame?.meta || {};
  const activeInputSummary = activeForecastMeta.input_summary || {};
  const missingStaticChannels = Array.isArray(activeInputSummary.missing_or_placeholder_static)
    ? activeInputSummary.missing_or_placeholder_static
    : [];
  const activeThreshold = activeForecastMeta.threshold ?? activeForecastFrame?.threshold;
  const activeDisplayFloor = activeForecastMeta.display_floor ?? activeForecastFrame?.display_floor;
  const activeDisplayArea = activeForecastFrame?.display_area_fraction ?? activeForecastMeta.display_area_fraction;
  const activeStepHours = activeForecastMeta.step_hours ?? activeForecastFrame?.step_hours;
  const activeScaleMode = activeForecastMeta.probability_scale?.mode || 'absolute';
  const activeModelStatus = activeForecastMeta.model_meta?.config_status || activeForecastMeta.model_meta?.metadata_source || 'model metadata unavailable';
  const activeQuality = activeForecastMeta.quality || {};
  const activeQualityStatus = activeQuality.status || 'unknown';
  const activeQualityReasons = Array.isArray(activeQuality.reasons) ? activeQuality.reasons : [];
  const activeTargetMode = activeForecastMeta.model_meta?.target_mode || 'unknown';
  const activeNewBurn = activeForecastMeta.p_new_burn || {};
  const activeDisplayScore = activeForecastMeta.display_score || {};
  // The rendered field is the calibrated display score, not the raw sigmoid.
  // Reporting prob_max next to calibrated pixels overstated confidence by ~2x.
  const activeCalibratedPeak = activeDisplayScore.max ?? null;
  const activeRawPeak = activeNewBurn?.prob_max ?? activeForecastFrame?.prob_max ?? null;
  const sceneBandCount = forecastScene?.forecast?.features?.length ?? 0;
  const bandsAvailable = sceneBandCount > 0;
  const showingBands = spreadView === 'bands' && bandsAvailable;
  const activeNextFire = activeForecastMeta.p_next_fire || {};
  const activeObservedFire = activeForecastMeta.observed_fire || {};
  const activeDisplayMask = activeForecastMeta.display_mask || {};
  const activeAboveThreshold = activeNewBurn?.area_fraction ?? activeForecastFrame?.area_fraction;
  const activeThresholdOverride = Boolean(activeForecastMeta.threshold_override || activeForecastFrame?.threshold_override);
  const activeQualityLabel = activeQualityStatus === 'ok'
    ? 'NOAA gridded'
    : activeQualityStatus === 'unavailable'
      ? 'Unavailable'
      : (activeQualityReasons.includes('archive') ? 'Archive weather' : 'Fallback weather');
  const activeQualityText = [
    activeQualityStatus,
    ...activeQualityReasons,
  ].filter(Boolean).join(' / ') || 'unknown';
  const qualityBadgeClass = activeQualityStatus === 'ok' ? 'good' : (activeQualityStatus === 'unavailable' ? 'warn' : 'warn');

  return (
    <div className={watchShell ? 'watch-shell' : ''} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div
        ref={mapContainerRef}
        style={{ width: '100%', height: '100%' }}
        title="Press 'N' to toggle NDVI overlay"
      />
      {!watchShell && (
        <button className="hist-toggle" onClick={() => setShowHistPanel(v => !v)}>
          {showHistPanel ? 'X' : 'History'}
        </button>
      )}
      {showHistPanel && (
        <div className="hist-panel">
          <h4>Historical Fire Testing</h4>
          <p style={{fontSize:'11px',margin:'0 0 8px',color:'#aaa'}}>
            Run the model on known historical fires using archived weather.
          </p>
          {HISTORICAL_FIRES.map((fire, i) => (
            <button
              key={i}
              className="hist-btn"
              disabled={histRunning}
              onClick={() => runHistoricalPrediction(fire)}
            >
              {fire.name} ({fire.displayDate || fire.date})
            </button>
          ))}
          <hr style={{border:'none',borderTop:'1px solid #555',margin:'10px 0'}} />
          <p style={{fontSize:'11px',margin:'0 0 4px',color:'#aaa'}}>
            Or pick a custom date (runs prediction at current map center):
          </p>
          <input
            type="date"
            value={histDate}
            onChange={e => setHistDate(e.target.value)}
            style={{width:'100%',padding:'4px',borderRadius:'4px',border:'1px solid #555',background:'#333',color:'#eee',fontSize:'12px'}}
          />
          <button
            className="hist-btn"
            style={{marginTop:'6px',background:'#FF6600'}}
            disabled={histRunning || !histDate}
            onClick={runCustomHistorical}
          >
            {histRunning ? 'Running...' : 'Run Custom Date'}
          </button>
        </div>
      )}
      {isForecastLoading && (
        <div className="forecast-panel" data-testid="forecast-loading">
          <div className="forecast-header">
            <div className="forecast-title">Loading Forecast…</div>
          </div>
          <div className="forecast-meta">Preparing fire spread timeline</div>
          <div className="forecast-loading-bar" />
        </div>
      )}
      {forecastError && !forecastVisible && !isForecastLoading && (
        <div className="forecast-panel" data-testid="forecast-error">
          <div className="forecast-header">
            <div className="forecast-title">Forecast Unavailable</div>
            <button
              className="forecast-btn ghost"
              onClick={() => setForecastError(null)}
              aria-label="Dismiss forecast error"
            >
              Dismiss
            </button>
          </div>
          <div className="forecast-error-msg">{forecastError}</div>
        </div>
      )}
      {forecastVisible && activeForecastFrame && (
        <div className="forecast-panel" data-testid="forecast-panel">
          <div className="forecast-header">
            <div>
              <div className="forecast-title">Fire-spread risk — advisory, not a perimeter</div>
              {forecastTitle && <div className="forecast-context">{forecastTitle}</div>}
              <div className="forecast-label">{activeForecastFrame.label}</div>
            </div>
            <button
              className="forecast-btn ghost"
              onClick={clearForecastTimeline}
              aria-label="Clear forecast timeline"
            >
              Clear
            </button>
          </div>
          <div className="forecast-advisory">
            {showingBands
              ? 'Each band is where the model expects fire to have reached by that lead time. It is not an observed or predicted official perimeter.'
              : 'This heatmap shows modeled relative risk of new fire spread. It is not an observed or predicted official perimeter.'}
          </div>
          <div className="forecast-badges">
            <span className={`forecast-badge ${qualityBadgeClass}`}>
              {activeQualityLabel}
            </span>
            <span className="forecast-badge">Model: {activeTargetMode}</span>
            {activeThresholdOverride && (
              <span className="forecast-badge warn">Threshold override active</span>
            )}
            {activeQualityReasons.slice(0, 2).map(reason => (
              <span key={reason} className="forecast-badge warn">{reason.replace(/_/g, ' ')}</span>
            ))}
          </div>
          <div className="forecast-view-controls" role="group" aria-label="Spread view">
            <button
              className={`forecast-layer-btn ${showingBands ? 'active' : ''}`}
              onClick={() => setSpreadView('bands')}
              disabled={!bandsAvailable}
              title={bandsAvailable ? 'Cumulative arrival bands' : 'No arrival bands in this forecast'}
            >
              Arrival bands
            </button>
            <button
              className={`forecast-layer-btn ${showingBands ? '' : 'active'}`}
              onClick={() => setSpreadView('heat')}
              title="Per-cell probability heatmap"
            >
              Probability heat
            </button>
          </div>
          {showingBands && (
            <ol className="forecast-band-key" aria-label="Arrival band key">
              {[...forecastScene.forecast.features]
                .map(feature => feature.properties || {})
                .filter((props, idx, all) => all.findIndex(o => o.day === props.day) === idx)
                .sort((a, b) => (a.day ?? 0) - (b.day ?? 0))
                .map(props => (
                  <li
                    key={props.day}
                    className={props.lead_hours === activeForecastFrame?.lead_hours ? 'current' : ''}
                  >
                    <span className="swatch" style={{ background: props.color }} />
                    {props.label}
                  </li>
                ))}
            </ol>
          )}
          <div className="forecast-layer-controls" aria-label="Forecast layer controls">
            <button
              className={`forecast-layer-btn ${observedLayersVisible ? 'active' : ''}`}
              onClick={() => setObservedLayersVisible(!observedLayersVisible)}
            >
              Observed
            </button>
            <button
              className={`forecast-layer-btn ${forecastLayerMode === 'new_burn' ? 'active' : ''}`}
              onClick={() => setForecastLayerMode('new_burn')}
            >
              New Burn Risk
            </button>
            <button
              className={`forecast-layer-btn ${forecastLayerMode === 'next_fire' ? 'active' : ''}`}
              onClick={() => setForecastLayerMode('next_fire')}
            >
              Next Fire
            </button>
          </div>
          <div className="forecast-meta">
            Frame {activeForecastIndex + 1} of {forecastFrames.length}
            {activeCalibratedPeak != null
              ? ` · Peak calibrated confidence ${(activeCalibratedPeak * 100).toFixed(1)}%`
              : activeRawPeak != null ? ` · Peak raw score ${(activeRawPeak * 100).toFixed(1)}%` : ''}
            {activeAboveThreshold != null ? ` · Above threshold ${(activeAboveThreshold * 100).toFixed(1)}%` : ''}
            {activeStepHours ? ` · ${activeStepHours}h cadence` : ''}
          </div>
          <div className="forecast-ramp" aria-hidden="true" />
          <div className="forecast-metrics">
            <div className="forecast-metric">
              <strong>{activeThreshold != null ? `${(activeThreshold * 100).toFixed(0)}%` : '--'}</strong>
              Decision threshold
            </div>
            <div className="forecast-metric">
              <strong>{activeCalibratedPeak != null ? `${(activeCalibratedPeak * 100).toFixed(1)}%` : '--'}</strong>
              Peak calibrated confidence
              {activeRawPeak != null && (
                <em>raw {(activeRawPeak * 100).toFixed(0)}%</em>
              )}
            </div>
            <div className="forecast-metric">
              <strong>{activeAboveThreshold != null ? `${(activeAboveThreshold * 100).toFixed(1)}%` : '--'}</strong>
              Above decision threshold
            </div>
            <div className="forecast-metric">
              <strong>{activeNextFire?.area_fraction != null ? `${(activeNextFire.area_fraction * 100).toFixed(1)}%` : activeForecastFrame?.area_fraction != null ? `${(activeForecastFrame.area_fraction * 100).toFixed(1)}%` : '--'}</strong>
              Next-fire area
            </div>
          </div>
          <div className={`forecast-status ${missingStaticChannels.length ? 'warn' : ''}`}>
            Scale: {activeScaleMode} 0-1.
            {activeDisplayFloor != null ? ` Display floor ${(activeDisplayFloor * 100).toFixed(0)}%.` : ''}
            {activeDisplayArea != null ? ` Visible cells ${(activeDisplayArea * 100).toFixed(1)}%.` : ''}
            {activeObservedFire?.area_fraction != null ? ` Observed ${(activeObservedFire.area_fraction * 100).toFixed(1)}%.` : ''}
            {activeDisplayMask?.masked_fraction != null ? ` Masked nonburnable ${(activeDisplayMask.masked_fraction * 100).toFixed(1)}%.` : ''}
            {' '}
            View: {showingBands ? `arrival bands (${sceneBandCount} polygons)` : 'probability heat'}.
            {' '}
            Layer: {forecastLayerMode === 'next_fire' ? 'reconstructed next-fire context' : 'new-burn risk'}.
            {' '}
            {missingStaticChannels.length
              ? `Static placeholders: ${missingStaticChannels.slice(0, 5).join(', ')}${missingStaticChannels.length > 5 ? '...' : ''}`
              : `Input quality: ${activeQualityText}. ${activeModelStatus}`}
          </div>
          <input
            className="forecast-slider"
            type="range"
            min="0"
            max={Math.max(0, forecastFrames.length - 1)}
            step="1"
            value={activeForecastIndex}
            aria-label="Forecast timeline slider"
            onChange={e => {
              setIsForecastPlaying(false);
              setActiveForecastIndex(Number(e.target.value));
            }}
          />
          <div className="forecast-controls">
            <button
              className="forecast-btn"
              onClick={() => {
                setIsForecastPlaying(false);
                setActiveForecastIndex(idx => (idx - 1 + forecastFrames.length) % forecastFrames.length);
              }}
            >
              Prev
            </button>
            <button
              className="forecast-btn primary"
              onClick={() => setIsForecastPlaying(v => !v)}
            >
              {isForecastPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              className="forecast-btn"
              onClick={() => {
                setIsForecastPlaying(false);
                setActiveForecastIndex(idx => (idx + 1) % forecastFrames.length);
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

export default MapComponent;
