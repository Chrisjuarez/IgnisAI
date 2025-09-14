// src/components/MapComponent.js
import React, {
  useEffect,
  useRef,
  useState,
  useImperativeHandle,
  forwardRef
} from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { getWildfireData, predictFireSpread } from '../api';

mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;
const API_BASE = process.env.REACT_APP_API || 'http://localhost:5000/api';
const R_EARTH = 3958.8;

// ─────────────── Helpers ───────────────
function getBrightnessCategory(b) {
  if (b >= 375) return 'Extreme';
  if (b >= 350) return 'Severe';
  if (b >= 325) return 'Moderate';
  return 'Small';
}
function confidenceToPercent(raw) {
  const n = parseFloat(raw);
  if (isNaN(n)) return 0;
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
  if (d == null) return '—';
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                'S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(((d % 360) / 22.5)) % 16];
};
const msToMph = v => (v == null ? null : v * 2.23694);

// ─────────────── Component ───────────────
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

  // NDVI state/refs
  const [ndviOn, setNdviOn]           = useState(false);
  const ndviTemplateRef               = useRef(null);
  const NDVI_SOURCE_ID = 'ndvi-tiles';
  const NDVI_LAYER_ID  = 'ndvi-layer';

  // ── Exposed actions (for parent controls if needed)
  useImperativeHandle(ref, () => ({
    refreshWildfires: fetchWildfires,
    toggleNdvi
  }), [wildfires, ndviOn]);

  // ─────────────── Data: FIRMS ───────────────
  const fetchWildfires = async () => {
    setIsFetching?.(true);
    try {
      const { data } = await getWildfireData();
      const arr = data.data || [];
      setWildfires(arr);
      onFiresUpdated?.(arr.length);
      updateWildfireSource(arr);
    } finally {
      setIsFetching?.(false);
    }
  };

  const updateWildfireSource = (dataArray = wildfires) => {
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
      features: filtered.map(f => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [f.longitude, f.latitude] },
        properties: {
          brightnessCat:  getBrightnessCategory(f.brightness),
          confidenceRaw:  f.confidence,
          timestamp:      f.timestamp,
          brightness:     f.brightness,
          latitude:       f.latitude,
          longitude:      f.longitude
        }
      }))
    });
  };

  // ─────────────── User marker ───────────────
  const updateUserSource = () => {
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
  };

  // ─────────────── Prediction ───────────────
  const getDirectionName = (degrees) => {
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(((degrees % 360) / 45)) % 8;
    return directions[index];
  };

  const handlePredictFireSpread = async (fireProps) => {
    if (isPredicting) return;
    setIsPredicting(true);
    try {
      if (activePopup) activePopup.remove();

      const fireData = {
        lat: fireProps.latitude,
        lng: fireProps.longitude,
        brightness: fireProps.brightness
      };

      const prediction = await predictFireSpread(fireData);

      // show spread viz if 10%+
      const spreadProbability = prediction.spread_probability * 100;
      if (spreadProbability >= 10) displayFirePrediction(prediction);

      showPredictionPopup(fireProps, prediction);
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

  const displayFirePrediction = (prediction) => {
    const map = mapRef.current;
    if (!map) return;

    if (map.getSource('fire-spread-prediction')) {
      ['fire-spread-direction','fire-spread-area','fire-spread-points'].forEach(id => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      map.removeSource('fire-spread-prediction');
    }

    map.addSource('fire-spread-prediction', { type: 'geojson', data: prediction.geojson });

    map.addLayer({
      id: 'fire-spread-area',
      type: 'fill',
      source: 'fire-spread-prediction',
      filter: ['==', ['get', 'type'], 'spread'],
      paint: {
        'fill-color': [
          'interpolate', ['linear'], ['get', 'probability'],
          0.3, 'rgba(255, 255, 0, 0.2)',
          0.5, 'rgba(255, 165, 0, 0.3)',
          0.7, 'rgba(255, 0, 0, 0.4)'
        ],
        'fill-outline-color': '#ff0000'
      }
    });
    map.addLayer({
      id: 'fire-spread-direction',
      type: 'line',
      source: 'fire-spread-prediction',
      filter: ['==', ['get', 'type'], 'direction'],
      paint: { 'line-color': '#ff0000', 'line-width': 2, 'line-dasharray': [2, 1] }
    });
    map.addLayer({
      id: 'fire-spread-points',
      type: 'circle',
      source: 'fire-spread-prediction',
      filter: ['==', ['get', 'type'], 'spread_point'],
      paint: {
        'circle-radius': 4,
        'circle-color': [
          'interpolate', ['linear'], ['get', 'probability'],
          0.3, '#ffff00', 0.5, '#ffa500', 0.7, '#ff0000'
        ],
        'circle-opacity': 0.7
      }
    });
  };

  const showPredictionPopup = (fireProps, prediction) => {
    const map = mapRef.current;
    if (!map) return;

    const env = prediction.environmental_data;
    const spreadProbability = Math.round(prediction.spread_probability * 100);

    let popupContent = '';
    if (spreadProbability < 9) {
      popupContent = `
        <div class="wildfire-popup">
          <h4>${getSeverityIcon(fireProps.brightnessCat)} ${fireProps.brightnessCat} Fire</h4>
          <div class="prediction-results">
            <h5>Fire Spread Prediction</h5>
            <p>This fire is unlikely to spread significantly.</p>
            <p><strong>Spread probability:</strong> ${spreadProbability}%</p>
            <div class="env-data">
              <h6>Environmental Factors:</h6>
              <p>Wind: ${env.wind_speed.toFixed(1)} km/h ${getDirectionName(env.wind_direction)}</p>
              <p>Temp: ${env.temperature.toFixed(1)}°C, Humidity: ${env.humidity.toFixed(0)}%</p>
              <p>Data source: ${env.data_source === "weather_api" ? "Real-time weather" : "Estimated"}</p>
            </div>
          </div>
        </div>
      `;
    } else {
      const spreadStatus = spreadProbability >= 20 ? 'Yes' : 'Possibly';
      popupContent = `
        <div class="wildfire-popup">
          <h4>${getSeverityIcon(fireProps.brightnessCat)} ${fireProps.brightnessCat} Fire</h4>
          <div class="prediction-results">
            <h5>Fire Spread Prediction</h5>
            <p><strong>Will spread:</strong> ${spreadStatus}</p>
            <p><strong>Spread probability:</strong> ${spreadProbability}%</p>
            <p><strong>Spread distance:</strong> ${prediction.spread_distance_km.toFixed(2)} km</p>
            <p><strong>Main direction:</strong> ${getDirectionName(prediction.spread_direction)}</p>
            <div class="env-data">
              <h6>Environmental Factors:</h6>
              <p>Wind: ${env.wind_speed.toFixed(1)} km/h ${getDirectionName(env.wind_direction)}</p>
              <p>Temp: ${env.temperature.toFixed(1)}°C, Humidity: ${env.humidity.toFixed(0)}%</p>
              <p>Data source: ${env.data_source === "weather_api" ? "Real-time weather" : "Estimated"}</p>
            </div>
          </div>
        </div>
      `;
    }

    const popup = new mapboxgl.Popup()
      .setLngLat([fireProps.longitude, fireProps.latitude])
      .setHTML(popupContent)
      .addTo(map);

    setActivePopup(popup);

    // After popup shows, fetch REAL-TIME Topography + Weather and append
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

  // ─────────────── Layers & sources ───────────────
  const setupLayers = () => {
    const map = mapRef.current;
    if (!map) return;

    // Wildfires
    if (map.getLayer('wildfires-layer'))  map.removeLayer('wildfires-layer');
    if (map.getSource('wildfires-source')) map.removeSource('wildfires-source');
    map.addSource('wildfires-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
      id: 'wildfires-layer',
      type: 'circle',
      source: 'wildfires-source',
      paint: {
        'circle-radius':       4,
        'circle-color':        '#FF4500',
        'circle-stroke-width': 1,
        'circle-stroke-color': '#fff'
      }
    });

    // click handler
    if (clickHandlerRef.current) {
      map.off('click', 'wildfires-layer', clickHandlerRef.current);
    }
    clickHandlerRef.current = async e => {
      const f = e.features?.[0];
      if (!f) return;

      const { brightnessCat, confidenceRaw, timestamp, brightness } = f.properties;
      const coords = f.geometry.coordinates;
      const timeStr = timestamp ? new Date(timestamp).toLocaleString() : 'Unknown';
      const pct    = confidenceToPercent(confidenceRaw);
      const addr   = await reverseGeocode(coords[1], coords[0], mapboxgl.accessToken);
      const icon   = getSeverityIcon(brightnessCat);

      if (activePopup) activePopup.remove();

      const fireProps = { brightnessCat, brightness, latitude: coords[1], longitude: coords[0] };

      const popup = new mapboxgl.Popup()
        .setLngLat(coords)
        .setHTML(`
          <div class="wildfire-popup">
            <h4>${icon} ${brightnessCat} Fire</h4>
            <p><strong>Address:</strong> ${addr}</p>
            <p><strong>Confidence:</strong> ${pct}%</p>
            <p><strong>Captured at:</strong> ${timeStr}</p>
            <button id="predict-spread-btn" class="predict-spread-btn">
              ${isPredicting ? 'Predicting...' : 'Predict Fire Spread'}
            </button>
          </div>
        `)
        .addTo(map);

      setActivePopup(popup);

      const btn = document.getElementById('predict-spread-btn');
      if (btn) btn.addEventListener('click', () => handlePredictFireSpread(fireProps));
    };
    map.on('click', 'wildfires-layer', clickHandlerRef.current);

    // User location
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

    // If NDVI was on before a style change, re-add it
    if (ndviOn) ensureNdviLayer();
  };

  // ─────────────── NDVI overlay ───────────────
  async function ensureNdviLayer() {
    const map = mapRef.current;
    if (!map) return;

    // Get template once
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

    // Add source/layer if missing
    if (!map.getSource(NDVI_SOURCE_ID)) {
      map.addSource(NDVI_SOURCE_ID, {
        type: 'raster',
        tiles: [ndviTemplateRef.current],
        tileSize: 256,
        attribution: 'NASA GIBS MODIS NDVI (16-day)'
      });
    }
    if (!map.getLayer(NDVI_LAYER_ID)) {
      // Put NDVI under the wildfire points so markers stay visible
      const before = map.getStyle().layers.find(l => l.id === 'wildfires-layer') ? 'wildfires-layer' : undefined;
      map.addLayer({
        id: NDVI_LAYER_ID,
        type: 'raster',
        source: NDVI_SOURCE_ID,
        paint: { 'raster-opacity': 0.6 }
      }, before);
    }
  }

  async function toggleNdvi() {
    const map = mapRef.current;
    if (!map) return;

    if (!ndviOn) {
      await ensureNdviLayer();
      setNdviOn(true);
      // make sure visible
      if (map.getLayer(NDVI_LAYER_ID)) {
        map.setLayoutProperty(NDVI_LAYER_ID, 'visibility', 'visible');
      }
    } else {
      // just hide (faster than remove/re-add)
      if (map.getLayer(NDVI_LAYER_ID)) {
        map.setLayoutProperty(NDVI_LAYER_ID, 'visibility', 'none');
      }
      setNdviOn(false);
    }
  }

  // Keyboard shortcut: press "n" to toggle NDVI
  useEffect(() => {
    const onKey = e => {
      if (e.key.toLowerCase() === 'n') toggleNdvi();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [ndviOn]);

  // ─────────────── Lifecycle ───────────────
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

  // Reapply on style change (rebuild layers, reattach NDVI if on)
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

  // Refresh layer when filters/data change
  useEffect(() => {
    updateWildfireSource();
  }, [wildfires, brightnessFilter, confidenceFilter]);

  // Update user marker
  useEffect(() => {
    updateUserSource();
  }, [userLocation]);

  // Compute & send all in-range fires (for side panel)
  useEffect(() => {
    if (!onNearbyFiresUpdate) return;
    const t = setTimeout(async () => {
      if (!userLocation || wildfires.length === 0 || range <= 0) {
        onNearbyFiresUpdate([]);
        return;
      }
      const inRange = wildfires
        .map(f => {
          const d = haversineDistance(
            userLocation.lat, userLocation.lng,
            f.latitude, f.longitude
          );
          return {
            ...f,
            distance:      d,
            brightnessCat: getBrightnessCategory(f.brightness),
            confidenceRaw: f.confidence
          };
        })
        .filter(f => f.distance <= range);

      const enriched = await Promise.all(
        inRange.map(async f => {
          const pct = confidenceToPercent(f.confidenceRaw);
          const confCat = bracketConfidence(pct);
          const cityName = await reverseGeocode(f.latitude, f.longitude, mapboxgl.accessToken);
          return { cityName, distance: f.distance, brightnessCat: f.brightnessCat, confidenceCat: confCat };
        })
      );
      onNearbyFiresUpdate(enriched);
    }, 500);
    return () => clearTimeout(t);
  }, [userLocation, range, wildfires, onNearbyFiresUpdate]);

  // Popup button styles
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
    `;
    document.head.appendChild(style);
    return () => { document.head.removeChild(style); };
  }, []);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: '100%', height: '100%', position: 'relative' }}
      title="Press 'N' to toggle NDVI overlay"
    />
  );
});

export default MapComponent;