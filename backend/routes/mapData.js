const express = require('express');
const axios = require('axios');
const { _private: fireDataHelpers } = require('./fireData');
const { LruTtlCache } = require('../utils/lruCache');

const router = express.Router();

const WESTERN_CONUS_BBOX = '-125.1,31.0,-101.8,49.5';
const WESTERN_STATES = ['CA', 'OR', 'WA', 'NV', 'AZ', 'UT', 'ID', 'MT', 'WY', 'CO', 'NM'];
const CACHE_TTL_MS = Number(process.env.MAP_BOOTSTRAP_CACHE_MS || 180000);
// Entry count is a weak bound here: one bootstrap ranges from ~0.4 MB for a
// city bbox to tens of MB for western CONUS, and the key is the bbox, so
// panning the map mints a new entry each time. 200 of the large ones is
// gigabytes on a 512 MB instance, which is what took the backend down.
// The byte budget is the real bound; the count is a backstop.
const CACHE_MAX_ENTRIES = Number(process.env.MAP_BOOTSTRAP_CACHE_MAX || 12);
const CACHE_MAX_BYTES = Number(process.env.MAP_BOOTSTRAP_CACHE_BYTES || 64 * 1024 * 1024);
const USER_AGENT = process.env.IGNIS_USER_AGENT || 'IgnisAI wildfire map';

const WFIGS_INCIDENTS_URL =
  process.env.WFIGS_INCIDENTS_URL ||
  'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query';

const WFIGS_PERIMETERS_URL =
  process.env.WFIGS_CURRENT_PERIMETERS_URL ||
  'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query';

const FIRIS_PUBLIC_URL =
  process.env.FIRIS_PERIMETERS_URL ||
  'https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/CA_Perimeters_NIFC_FIRIS_public_view/FeatureServer/0/query';

// In-memory cache. Bounded LRU + TTL to prevent unbounded growth on
// long-running instances; older entries are evicted lazily.
const cache = new LruTtlCache({ max: CACHE_MAX_ENTRIES, ttl: CACHE_TTL_MS, maxBytes: CACHE_MAX_BYTES });

function nowIso() {
  return new Date().toISOString();
}

function statusOk(source, count, extra = {}) {
  return {
    source,
    ok: true,
    partial: false,
    stale: false,
    count,
    lastFetchedAt: nowIso(),
    ...extra,
  };
}

function statusError(source, error, extra = {}) {
  return {
    source,
    ok: false,
    partial: true,
    stale: false,
    count: 0,
    lastFetchedAt: nowIso(),
    error: error?.message || String(error || 'unavailable'),
    ...extra,
  };
}

function getCached(key) {
  return cache.get(key);
}

function setCached(key, value) {
  return cache.set(key, value);
}

function parseBbox(raw = WESTERN_CONUS_BBOX) {
  const parts = String(raw || '')
    .split(',')
    .map((value) => Number(value.trim()));
  if (parts.length !== 4 || parts.some((value) => !Number.isFinite(value))) {
    return parseBbox(WESTERN_CONUS_BBOX);
  }
  const [w, s, e, n] = parts;
  if (w >= e || s >= n || w < -180 || e > 180 || s < -90 || n > 90) {
    return parseBbox(WESTERN_CONUS_BBOX);
  }
  return { w, s, e, n, raw: `${w},${s},${e},${n}` };
}

function bboxIntersectsPoint(bbox, lon, lat) {
  return Number.isFinite(lon) && Number.isFinite(lat) &&
    lon >= bbox.w && lon <= bbox.e && lat >= bbox.s && lat <= bbox.n;
}

function arcgisParams({ bbox, recordCount = 1000 }) {
  const params = {
    f: 'geojson',
    where: '1=1',
    outFields: '*',
    returnGeometry: true,
    outSR: 4326,
    resultRecordCount: recordCount,
  };
  if (bbox) {
    params.geometry = `${bbox.w},${bbox.s},${bbox.e},${bbox.n}`;
    params.geometryType = 'esriGeometryEnvelope';
    params.inSR = 4326;
    params.spatialRel = 'esriSpatialRelIntersects';
  }
  return params;
}

async function fetchArcgisGeoJson(url, params) {
  const { data } = await axios.get(url, {
    params,
    timeout: 25000,
    headers: { 'User-Agent': USER_AGENT },
  });
  if (data?.type === 'FeatureCollection') return data;
  return { type: 'FeatureCollection', features: [] };
}

function toIso(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number') {
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? null : d.toISOString();
  }
  const ts = Date.parse(value);
  return Number.isFinite(ts) ? new Date(ts).toISOString() : null;
}

function toNumber(value, fallback = null) {
  if (value == null || value === '') return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function firstValue(props, names) {
  for (const name of names) {
    const value = props?.[name];
    if (value != null && value !== '') return value;
  }
  return null;
}

function normalizeName(value) {
  return String(value || '')
    .replace(/\s+fire$/i, '')
    .replace(/[^a-z0-9]+/gi, ' ')
    .trim()
    .toLowerCase();
}

function resolvePredictionEligibility(incident, { hasPerimeter = false, hasHotspots = false } = {}) {
  const reasons = [];
  if (incident?.type !== 'wildfire') reasons.push('record is not an active wildfire');
  if (incident?.status !== 'active') reasons.push('incident is not active');
  if (!hasPerimeter && !hasHotspots) reasons.push('prediction requires a recent hotspot cluster or matching perimeter');
  return {
    eligible: reasons.length === 0,
    reasons,
  };
}

function normalizeIncident(feature) {
  const props = feature?.properties || {};
  const coords = Array.isArray(feature?.geometry?.coordinates)
    ? feature.geometry.coordinates
    : [];
  const lon = toNumber(firstValue(props, ['InitialLongitude', 'POOLongitude', 'Longitude', 'lon']), toNumber(coords[0]));
  const lat = toNumber(firstValue(props, ['InitialLatitude', 'POOLatitude', 'Latitude', 'lat']), toNumber(coords[1]));
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const rawName = firstValue(props, ['IncidentName', 'attr_IncidentName', 'poly_IncidentName', 'FireName', 'FIRE_NAME']) || 'Unnamed Incident';
  const irwinId = firstValue(props, ['IrwinID', 'irwinID', 'IRWINID']);
  const uniqueId = firstValue(props, ['UniqueFireIdentifier', 'UniqueFireId', 'FireCode']);
  const objectId = firstValue(props, ['OBJECTID', 'objectid', 'FID']);
  const category = String(firstValue(props, ['IncidentTypeCategory', 'IncidentTypeKind']) || 'WF').toUpperCase();
  const fireOut = toIso(firstValue(props, ['FireOutDateTime', 'FireOutDateTime_dt']));
  const isActiveCandidate = String(firstValue(props, ['ActiveFireCandidate']) ?? '').toLowerCase();
  const status = fireOut || isActiveCandidate === '0' || isActiveCandidate === 'false' ? 'inactive' : 'active';

  return {
    id: `wfigs:${irwinId || uniqueId || objectId || `${lat.toFixed(4)}:${lon.toFixed(4)}`}`,
    sourceId: irwinId || uniqueId || objectId || null,
    name: String(rawName).replace(/\s+/g, ' ').trim(),
    type: category.includes('RX') || category.includes('PRESCRIB') ? 'prescribed' : 'wildfire',
    status,
    lat,
    lon,
    county: firstValue(props, ['POOCounty', 'County', 'county']) || null,
    state: firstValue(props, ['POOState', 'State', 'state']) || null,
    acres: toNumber(firstValue(props, ['IncidentSize', 'DailyAcres', 'GISAcres', 'Acres'])),
    containmentPct: toNumber(firstValue(props, ['PercentContained', 'Containment', 'ContainmentPercent'])),
    updatedAt: toIso(firstValue(props, ['ModifiedOnDateTime_dt', 'ModifiedOnDateTime', 'CreateDate', 'DateCurrent'])),
    createdAt: toIso(firstValue(props, ['FireDiscoveryDateTime', 'FireDiscoveryDateTime_dt', 'CreatedOnDateTime_dt'])),
    sourceNames: ['WFIGS', 'NIFC'],
    hasPerimeter: false,
    hasHotspots: false,
    hasPrediction: true,
    quality: { status: 'ok', reasons: [] },
    geometry: { type: 'Point', coordinates: [lon, lat] },
    rawProperties: props,
  };
}

function perimeterIncidentName(feature) {
  const props = feature?.properties || {};
  return firstValue(props, [
    'poly_IncidentName',
    'attr_IncidentName',
    'IncidentName',
    'FIRE_NAME',
    'FireName',
    'incidentname',
  ]);
}

function annotateIncidents(incidents, perimeters, hotspots) {
  const perimeterNames = new Set(
    (perimeters?.features || [])
      .map(perimeterIncidentName)
      .map(normalizeName)
      .filter(Boolean)
  );

  return incidents.map((incident) => {
    const nameKey = normalizeName(incident.name);
    const hasPerimeter = perimeterNames.has(nameKey) ||
      Array.from(perimeterNames).some((candidate) => candidate && nameKey && (candidate.includes(nameKey) || nameKey.includes(candidate)));
    const hasHotspots = hotspots.some((fire) => {
      const dLat = Math.abs(Number(fire.latitude) - incident.lat);
      const dLon = Math.abs(Number(fire.longitude) - incident.lon);
      return dLat <= 0.35 && dLon <= 0.35;
    });
    const predictionEligibility = resolvePredictionEligibility(incident, { hasPerimeter, hasHotspots });
    return {
      ...incident,
      hasPerimeter,
      hasHotspots,
      sourceNames: [
        ...incident.sourceNames,
        ...(hasPerimeter ? ['Perimeters'] : []),
        ...(hasHotspots ? ['FIRMS'] : []),
      ],
      hasPrediction: predictionEligibility.eligible,
      predictionEligibility,
    };
  });
}

async function fetchIncidents(bbox) {
  const geojson = await fetchArcgisGeoJson(WFIGS_INCIDENTS_URL, arcgisParams({ bbox, recordCount: 1500 }));
  return (geojson.features || [])
    .map(normalizeIncident)
    .filter(Boolean)
    .filter((incident) => bboxIntersectsPoint(bbox, incident.lon, incident.lat));
}

async function fetchPerimeters(bbox) {
  const [wfigs, firis] = await Promise.allSettled([
    fetchArcgisGeoJson(WFIGS_PERIMETERS_URL, arcgisParams({ bbox, recordCount: 1000 })),
    fetchArcgisGeoJson(FIRIS_PUBLIC_URL, arcgisParams({ bbox, recordCount: 1000 })),
  ]);
  const features = [];
  for (const result of [wfigs, firis]) {
    if (result.status === 'fulfilled') {
      features.push(...(result.value.features || []));
    }
  }
  return {
    geojson: { type: 'FeatureCollection', features },
    partial: [wfigs, firis].some((result) => result.status === 'rejected'),
  };
}

async function fetchHotspots(bbox) {
  const result = await fireDataHelpers.fetchCurrentFires({
    bbox: bbox.raw,
    days: 2,
    excludeFlares: true,
  });
  const fires = Array.isArray(result.fires) ? result.fires : [];
  // Footprint polygons are deliberately not built here. They are one polygon
  // per detection - thousands at full extent - and /wildfires/footprints
  // builds its own copy for the client that actually renders them.
  return {
    fires,
    stale: result.stale === true,
    message: result.message,
  };
}

function eventMatchesFireWeather(event) {
  return /red flag warning|fire weather watch|fire weather warning/i.test(String(event || ''));
}

async function fetchAlerts() {
  const responses = await Promise.allSettled(
    WESTERN_STATES.map((state) => axios.get('https://api.weather.gov/alerts/active', {
      params: { area: state },
      timeout: 20000,
      headers: {
        Accept: 'application/geo+json',
        'User-Agent': USER_AGENT,
      },
    }))
  );
  const seen = new Set();
  const features = [];
  for (const response of responses) {
    if (response.status !== 'fulfilled') continue;
    const collection = response.value?.data;
    for (const feature of collection?.features || []) {
      const props = feature.properties || {};
      if (!eventMatchesFireWeather(props.event)) continue;
      const id = feature.id || props.id || `${props.event}:${props.effective}:${props.areaDesc}`;
      if (seen.has(id)) continue;
      seen.add(id);
      features.push({
        ...feature,
        properties: {
          ...props,
          id,
          source: 'NWS',
          status: props.status || null,
          event: props.event || 'Fire Weather Alert',
          headline: props.headline || props.event || 'Fire Weather Alert',
          updatedAt: props.sent || props.effective || null,
        },
      });
    }
  }
  return {
    geojson: { type: 'FeatureCollection', features },
    partial: responses.some((response) => response.status === 'rejected'),
  };
}

function normalizeAlert(feature) {
  const props = feature?.properties || {};
  return {
    id: props.id || feature.id || `${props.event}:${props.effective}`,
    type: 'nws_alert',
    event: props.event || 'Fire Weather Alert',
    headline: props.headline || props.event || 'Fire Weather Alert',
    areaDesc: props.areaDesc || null,
    severity: props.severity || null,
    certainty: props.certainty || null,
    urgency: props.urgency || null,
    status: props.status || null,
    effective: props.effective || null,
    expires: props.expires || null,
    ends: props.ends || null,
    sent: props.sent || null,
    updatedAt: props.sent || props.effective || null,
    description: props.description || '',
    instruction: props.instruction || '',
    geometry: feature.geometry || null,
    sourceNames: ['NWS'],
  };
}

function buildUpdates(incident, alerts = []) {
  const updates = [];
  if (incident?.updatedAt) {
    updates.push({
      id: `${incident.id}:updated`,
      source: 'WFIGS',
      title: `${incident.name} source record updated`,
      body: 'Official incident metadata changed in the WFIGS/NIFC feed.',
      createdAt: incident.updatedAt,
    });
  }
  if (incident?.createdAt) {
    updates.push({
      id: `${incident.id}:created`,
      source: 'WFIGS',
      title: `${incident.name} discovered`,
      body: 'Initial official incident discovery timestamp from the source feed.',
      createdAt: incident.createdAt,
    });
  }
  for (const alert of alerts.slice(0, 5)) {
    updates.push({
      id: `${incident?.id || 'area'}:${alert.id}`,
      source: 'NWS',
      title: alert.headline,
      body: alert.areaDesc || alert.description || 'Fire weather alert affecting the broader region.',
      createdAt: alert.updatedAt || alert.effective || nowIso(),
    });
  }
  return updates.sort((a, b) => Date.parse(b.createdAt || 0) - Date.parse(a.createdAt || 0));
}

async function buildBootstrap(query = {}) {
  const bbox = parseBbox(query.bbox || WESTERN_CONUS_BBOX);
  const cacheKey = `bootstrap:${bbox.raw}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const [incidentResult, perimeterResult, hotspotResult, alertResult] = await Promise.allSettled([
    fetchIncidents(bbox),
    fetchPerimeters(bbox),
    fetchHotspots(bbox),
    fetchAlerts(),
  ]);

  const rawIncidents = incidentResult.status === 'fulfilled' ? incidentResult.value : [];
  const perimeters = perimeterResult.status === 'fulfilled'
    ? perimeterResult.value
    : { geojson: { type: 'FeatureCollection', features: [] }, partial: true };
  const hotspots = hotspotResult.status === 'fulfilled'
    ? hotspotResult.value
    : { fires: [], footprints: { type: 'FeatureCollection', features: [] }, stale: false };
  const alerts = alertResult.status === 'fulfilled'
    ? alertResult.value
    : { geojson: { type: 'FeatureCollection', features: [] }, partial: true };

  const incidents = annotateIncidents(rawIncidents, perimeters.geojson, hotspots.fires)
    .sort((a, b) => {
      if (a.status !== b.status) return a.status === 'active' ? -1 : 1;
      return Number(b.acres || 0) - Number(a.acres || 0);
    });
  const normalizedAlerts = (alerts.geojson.features || []).map(normalizeAlert);

  return setCached(cacheKey, {
    updatedAt: nowIso(),
    bbox: bbox.raw,
    incidents,
    perimeters: perimeters.geojson,
    // Detections stay on the shared payload because /incidents/:id reads them
    // to find hotspots near one fire. They are withheld from the bootstrap
    // RESPONSE instead - see bootstrapResponse below. `hotspotFootprints` is
    // gone entirely: nothing ever read it.
    hotspots: hotspots.fires,
    alerts: normalizedAlerts,
    alertsGeojson: alerts.geojson,
    evacuations: [],
    layerStatus: {
      incidents: incidentResult.status === 'fulfilled'
        ? statusOk('WFIGS Incident Locations Current', incidents.length)
        : statusError('WFIGS Incident Locations Current', incidentResult.reason),
      perimeters: perimeterResult.status === 'fulfilled'
        ? statusOk('WFIGS/FIRIS perimeters', perimeters.geojson.features.length, { partial: perimeters.partial })
        : statusError('WFIGS/FIRIS perimeters', perimeterResult.reason),
      hotspots: hotspotResult.status === 'fulfilled'
        ? statusOk('NASA FIRMS VIIRS/MODIS', hotspots.fires.length, { stale: hotspots.stale, message: hotspots.message })
        : statusError('NASA FIRMS VIIRS/MODIS', hotspotResult.reason),
      alerts: alertResult.status === 'fulfilled'
        ? statusOk('NWS Alerts API', normalizedAlerts.length, { partial: alerts.partial })
        : statusError('NWS Alerts API', alertResult.reason),
      evacuations: {
        source: 'Official county/Genasys/CAL FIRE providers',
        ok: false,
        partial: true,
        stale: false,
        count: 0,
        lastFetchedAt: nowIso(),
        error: 'provider registry not configured for this county',
      },
    },
  });
}

function filterIncidents(incidents, query = {}) {
  const status = String(query.status || '').toLowerCase();
  const type = String(query.type || '').toLowerCase();
  const q = String(query.q || '').trim().toLowerCase();
  return incidents.filter((incident) => {
    if (status && status !== 'all' && incident.status !== status) return false;
    if (type && type !== 'all' && incident.type !== type) return false;
    if (q) {
      const haystack = [incident.name, incident.county, incident.state].filter(Boolean).join(' ').toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    return true;
  });
}

/**
 * What the map actually needs from a bootstrap.
 *
 * The full payload carries every FIRMS detection in the bbox so that
 * /incidents/:id can find the ones near a given fire. At western-CONUS extent
 * that is thousands of records, and serialising them dominated a response
 * large enough to exhaust a 512 MB instance on a single cold request. The map
 * never read them - it fetches /wildfires directly - so they stay server-side.
 */
function bootstrapResponse(payload) {
  const { hotspots, ...response } = payload;
  return { ...response, hotspotCount: Array.isArray(hotspots) ? hotspots.length : 0 };
}

router.get('/map/bootstrap', async (req, res) => {
  try {
    const payload = await buildBootstrap(req.query);
    res.json(bootstrapResponse(payload));
  } catch (err) {
    console.error('map bootstrap error:', err.message);
    res.status(200).json({
      updatedAt: nowIso(),
      incidents: [],
      perimeters: { type: 'FeatureCollection', features: [] },
      alerts: [],
      alertsGeojson: { type: 'FeatureCollection', features: [] },
      evacuations: [],
      layerStatus: {
        partial: true,
        error: err.message,
      },
    });
  }
});

router.get('/incidents', async (req, res) => {
  const payload = await buildBootstrap(req.query);
  const incidents = filterIncidents(payload.incidents, req.query);
  res.json({ updatedAt: payload.updatedAt, count: incidents.length, incidents, layerStatus: payload.layerStatus });
});

router.get('/incidents/:id', async (req, res) => {
  const payload = await buildBootstrap(req.query);
  const decodedId = decodeURIComponent(req.params.id);
  const incident = payload.incidents.find((item) => item.id === decodedId);
  if (!incident) {
    return res.status(404).json({ error: 'incident not found' });
  }
  const alerts = payload.alerts.slice(0, 8);
  return res.json({
    incident,
    perimeters: payload.perimeters,
    recentHotspots: payload.hotspots.filter((fire) => {
      const dLat = Math.abs(Number(fire.latitude) - incident.lat);
      const dLon = Math.abs(Number(fire.longitude) - incident.lon);
      return dLat <= 0.35 && dLon <= 0.35;
    }).slice(0, 100),
    alerts,
    evacuations: [],
    predictionEligibility: {
      eligible: incident.predictionEligibility?.eligible ?? incident.hasPrediction,
      reasons: incident.predictionEligibility?.reasons || [],
    },
    updates: buildUpdates(incident, alerts),
  });
});

router.get('/incidents/:id/updates', async (req, res) => {
  const payload = await buildBootstrap(req.query);
  const decodedId = decodeURIComponent(req.params.id);
  const incident = payload.incidents.find((item) => item.id === decodedId);
  if (!incident) {
    return res.status(404).json({ error: 'incident not found' });
  }
  res.json({ incidentId: incident.id, updates: buildUpdates(incident, payload.alerts) });
});

router.get('/alerts', async (req, res) => {
  const payload = await buildBootstrap(req.query);
  res.json({
    updatedAt: payload.updatedAt,
    count: payload.alerts.length,
    alerts: payload.alerts,
    geojson: payload.alertsGeojson,
    layerStatus: payload.layerStatus.alerts,
  });
});

router.get('/layers', async (_req, res) => {
  res.json({
    updatedAt: nowIso(),
    layers: [
      { id: 'incidents', label: 'Incidents', source: 'WFIGS Incident Locations Current', defaultVisible: true },
      { id: 'perimeters', label: 'Official perimeters', source: 'WFIGS/FIRIS', defaultVisible: true },
      { id: 'hotspots', label: 'FIRMS hotspots', source: 'NASA FIRMS VIIRS/MODIS', defaultVisible: true },
      { id: 'warnings', label: 'Fire weather warnings', source: 'NWS Alerts API', defaultVisible: true },
      { id: 'evacuations', label: 'Evacuations', source: 'Official county/provider registry', defaultVisible: false },
      { id: 'prediction', label: 'Ignis prediction', source: 'IgnisAI tilesvc', defaultVisible: true },
      { id: 'ndvi', label: 'NDVI', source: 'NASA/GIBS NDVI tile service', defaultVisible: false },
    ],
  });
});

module.exports = router;
module.exports._private = {
  parseBbox,
  normalizeIncident,
  normalizeAlert,
  buildUpdates,
  eventMatchesFireWeather,
  bootstrapResponse,
};
