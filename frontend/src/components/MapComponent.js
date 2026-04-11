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
import { getWildfireData, predictFireSpread } from '../api';
import { addPredictionOverlay } from '../utils/addPredictionOverlay';

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

const getDirectionName = deg => toCompass(deg);

// ---- Historical fire presets for model testing ----
const HISTORICAL_FIRES = [
  { name: 'Camp/Paradise Fire',  date: '2018-11-08', lat: 39.80, lon: -121.44, zoom: 10 },
  { name: 'Eaton Fire',          date: '2025-01-07', lat: 34.19, lon: -118.06, zoom: 11 },
  { name: 'Palisades Fire',     date: '2025-01-07', lat: 34.05, lon: -118.55, zoom: 11 },
  { name: 'Dixie Fire',         date: '2021-07-14', lat: 40.05, lon: -121.38, zoom: 10 },
  { name: 'Caldor Fire',        date: '2021-08-14', lat: 38.75, lon: -120.30, zoom: 10 },
];

// ---- Component --------------------------------------------------------------
const MapComponent = forwardRef(({
  brightnessFilter,
  confidenceFilter,
  onFiresUpdated,
  setIsFetching,
  mapStyle,
  userLocation,
  range,
  onNearbyFiresUpdate
}, ref) => {
  const mapContainerRef = useRef();
  const mapRef          = useRef();
  const clickHandlerRef = useRef();

  const [wildfires, setWildfires]       = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [activePopup, setActivePopup]   = useState(null);

  // Historical fire testing state
  const [showHistPanel, setShowHistPanel] = useState(false);
  const [histDate, setHistDate]           = useState('');
  const [histRunning, setHistRunning]     = useState(false);

  // NDVI overlay state
  const [ndviOn, setNdviOn] = useState(false);
  const ndviTemplateRef     = useRef(null);
  const NDVI_SOURCE_ID      = 'ndvi-tiles';
  const NDVI_LAYER_ID       = 'ndvi-layer';

  // ---------- Data: FIRMS ----------
  const fetchWildfires = useCallback(async () => {
    setIsFetching?.(true);
    try {
      // You can pass { predictableOnly:true } if you want to hide weak signals by default.
      const { data } = await getWildfireData();
      const arr = data?.data || [];
      setWildfires(arr);
      onFiresUpdated?.(arr.length);
      updateWildfireSource(arr);
    } finally {
      setIsFetching?.(false);
    }
  }, [onFiresUpdated, setIsFetching]);

  // ---------- Update sources ----------
  function updateWildfireSource(dataArray = wildfires) {
    const map = mapRef.current;
    if (!map) return;
    const src = map.getSource('wildfires-source');
    if (!src) return;

    const filtered = dataArray.filter(f => {
      const b = getBrightnessCategory(f.brightness);
      const pct = confidenceToPercent(f.confidence);
      const c = bracketConfidence(pct);
      return (!brightnessFilter || brightnessFilter === b)
          && (!confidenceFilter || confidenceFilter === c);
    });
    onFiresUpdated?.(filtered.length);

    src.setData({
      type: 'FeatureCollection',
      features: filtered.map(f => {
        const pct = confidenceToPercent(f.confidence);
        const brightnessCat = getBrightnessCategory(f.brightness);
        // Predictability heuristic (also provided by backend in newer version)
        const predictable = (f.predictable === true) || (f.brightness >= 325 && pct >= 50);
        return {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [f.longitude, f.latitude] },
          properties: {
            brightnessCat,
            confidencePct: pct,           // store as 0..100 for UI
            predictable,
            timestamp: f.timestamp,
            brightness: f.brightness,
            latitude: f.latitude,
            longitude: f.longitude
          }
        };
      })
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

  // ---------- Expose actions to parent ----------
  useImperativeHandle(ref, () => ({
    refreshWildfires: fetchWildfires,
    toggleNdvi
  }), [fetchWildfires, toggleNdvi]);

  // ---------- Prediction ----------
  const handlePredictFireSpread = async (fireProps) => {
    if (isPredicting) return;
    setIsPredicting(true);
    try {
      if (activePopup) activePopup.remove();

      // Retry logic for cold-start timeouts (Render spins down idle services)
      let prediction = null;
      let lastErr = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          prediction = await predictFireSpread({
            lat: fireProps.latitude,
            lng: fireProps.longitude,
            thr: 0.01,
            Tseq: 1,
          });
          break;
        } catch (err) {
          lastErr = err;
          console.warn(`Prediction attempt ${attempt + 1} failed, retrying...`, err.message);
        }
      }
      if (!prediction) throw lastErr;

      // Show raster heatmap overlay (with same retry tolerance via fetch)
      const result = await addPredictionOverlay(mapRef.current, {
        apiBase: API_BASE,
        lat: fireProps.latitude,
        lon: fireProps.longitude,
        mode: 'raster',
        thr: 0.01,
        floor: 0.01,
        Tseq: 1,
        gamma: 0.7,
        opacity: 0.75,
        smooth: true,
        alphaThreshold: 0.01,
      });

      // Fit to overlay bounds (makes it look “accurate” instead of offset)
      if (result?.bounds) {
        const [w, s, e, n] = result.bounds;
        mapRef.current.fitBounds([[w, s], [e, n]], { padding: 40, duration: 800 });
      }

      // 2) Keep popup + stats
      showPredictionPopup(fireProps, prediction);

      // OPTIONAL: If you still want vector polygons sometimes, keep this behind a toggle.
      // if (prediction?.geojson) displayFirePrediction({ geojson: prediction.geojson });
    } catch (error) {
      console.error('Failed to predict fire spread:', error);
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
      setIsPredicting(false);
    }
  };

  const displayFirePrediction = ({ geojson }) => {
    const map = mapRef.current;
    if (!map || !geojson) return;

    if (map.getSource('fire-spread-prediction')) {
      ['fire-spread-area','fire-spread-direction','fire-spread-points'].forEach(id => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      map.removeSource('fire-spread-prediction');
    }

    map.addSource('fire-spread-prediction', { type: 'geojson', data: geojson });

    map.addLayer({
      id: 'fire-spread-area',
      type: 'fill',
      source: 'fire-spread-prediction',
      paint: {
        'fill-color': 'rgba(255,0,0,0.35)',
        'fill-outline-color': '#ff0000'
      }
    });
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
    const willSpread = maxPct >= 15 ? 'Yes' : maxPct >= 5 ? 'Possibly' : 'Unlikely';
    // Spread distance based on prob_max rather than area_fraction
    const spreadKm = prediction?.spread_distance_km ?? (64 * Math.sqrt(Math.max(0, probMax)));

    const popup = new mapboxgl.Popup()
      .setLngLat([fireProps.longitude, fireProps.latitude])
      .setHTML(`
        <div class="wildfire-popup">
          <h4>${getSeverityIcon(fireProps.brightnessCat)} ${fireProps.brightnessCat} Fire</h4>
          <div class="prediction-results">
            <h5>Ignis Prediction</h5>
            <p><strong>Will spread:</strong> ${willSpread}</p>
            <p><strong>Peak probability:</strong> ${pctStr(maxPct)}</p>
            <p><strong>Mean probability:</strong> ${meanPct}%</p>
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

      const thr = 0.01;

      const result = await addPredictionOverlay(map, {
        apiBase: API_BASE,
        lat: latitude,
        lon: longitude,
        mode,
        thr,
        Tseq: 1,
        gamma: 0.7,
        floor: 0.01,
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
    if (!map || histRunning) return;
    setHistRunning(true);
    try {
      const { lat, lon, date, zoom, name } = preset;

      map.flyTo({ center: [lon, lat], zoom: zoom || 10, duration: 1200 });

      const result = await addPredictionOverlay(map, {
        apiBase: API_BASE,
        lat,
        lon,
        mode: 'raster',
        thr: 0.01,
        Tseq: 1,
        date,
        gamma: 0.7,
        floor: 0.01,
        opacity: 0.75,
        smooth: true,
        alphaThreshold: 0.01,
      });

      if (result?.bounds) {
        const [w, s, e, n] = result.bounds;
        map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 800 });
      }

      // Show info popup
      const probMax = result?.meta?.prob_max;
      const probMean = result?.meta?.prob_mean;
      const popup = new mapboxgl.Popup()
        .setLngLat([lon, lat])
        .setHTML(`
          <div class="wildfire-popup">
            <h4>${name}</h4>
            <div class="prediction-results">
              <h5>Historical Prediction (${date})</h5>
              <p><strong>Max probability:</strong> ${probMax != null ? (probMax * 100).toFixed(1) + '%' : '--'}</p>
              <p><strong>Mean probability:</strong> ${probMean != null ? (probMean * 100).toFixed(2) + '%' : '--'}</p>
              <p style="margin-top:8px;font-size:0.85em;color:#888;">
                Uses archived weather data. FIRMS fire detections only available for recent dates;
                older fires use ignition-point mode.
              </p>
            </div>
          </div>
        `)
        .addTo(map);
      setActivePopup(popup);
    } catch (e) {
      console.error('Historical prediction error:', e);
    } finally {
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

  // ---------- Layers setup ----------
  function setupLayers() {
    const map = mapRef.current;
    if (!map) return;

    // Wildfire dots
    if (map.getLayer('wildfires-layer'))  map.removeLayer('wildfires-layer');
    if (map.getSource('wildfires-source')) map.removeSource('wildfires-source');
    map.addSource('wildfires-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
      id: 'wildfires-layer',
      type: 'circle',
      source: 'wildfires-source',
      paint: {
        'circle-radius': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 12,
          'Severe', 8,
          'Moderate', 6,
          4
        ],
        'circle-color': [
          'match', ['get', 'brightnessCat'],
          'Extreme', '#CC0000',
          'Severe',  '#FF2200',
          'Moderate', '#FF6600',
          '#FFA500'
        ],
        'circle-opacity': [
          'interpolate', ['linear'], ['get', 'confidencePct'],
          0, 0.4,
          50, 0.7,
          100, 1.0
        ],
        'circle-stroke-width': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 2,
          'Severe', 1.5,
          1
        ],
        'circle-stroke-color': '#fff',
        'circle-blur': [
          'match', ['get', 'brightnessCat'],
          'Extreme', 0.4,
          0
        ]
      }
    });

    // Click handler (re-bind safely)
    if (clickHandlerRef.current) {
      map.off('click', 'wildfires-layer', clickHandlerRef.current);
    }
    clickHandlerRef.current = async e => {
      const f = e.features?.[0];
      if (!f) return;

      const p = f.properties || {};
      const coords = f.geometry?.coordinates || [];
      const timeStr = p.timestamp ? new Date(p.timestamp).toLocaleString() : 'Unknown';
      const addr   = await reverseGeocode(coords[1], coords[0], mapboxgl.accessToken);
      const icon   = getSeverityIcon(p.brightnessCat);
      const confPct = Number(p.confidencePct ?? 0);
      const canPredict = String(p.predictable) === 'true' || p.predictable === true;

      if (activePopup) activePopup.remove();

      const fireProps = {
        brightnessCat: p.brightnessCat,
        brightness: Number(p.brightness),
        latitude: Number(coords[1]),
        longitude: Number(coords[0])
      };

      const popup = new mapboxgl.Popup()
        .setLngLat(coords)
        .setHTML(`
          <div class="wildfire-popup">
            <h4>${icon} ${p.brightnessCat} Fire</h4>
            <p><strong>Address:</strong> ${addr}</p>
            <p><strong>Confidence:</strong> ${pctStr(confPct)}</p>
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
    map.on('click', 'wildfires-layer', clickHandlerRef.current);

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
      updateUserSource();
    });
  }, [mapStyle]);

  useEffect(() => { updateWildfireSource(); }, [wildfires, brightnessFilter, confidenceFilter]);
  useEffect(() => { updateUserSource(); }, [userLocation]);

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
        background-color: #4CAF50;
        color: white;
        padding: 8px 12px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
        width: 100%;
        font-weight: bold;
      }
      .predict-spread-btn:hover { background-color: #45a049; }
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
    `;
    document.head.appendChild(style);
    return () => { document.head.removeChild(style); };
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div
        ref={mapContainerRef}
        style={{ width: '100%', height: '100%' }}
        title="Press 'N' to toggle NDVI overlay"
      />
      <button className="hist-toggle" onClick={() => setShowHistPanel(v => !v)}>
        {showHistPanel ? 'X' : 'History'}
      </button>
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
              {fire.name} ({fire.date})
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
    </div>
  );
});

export default MapComponent;
