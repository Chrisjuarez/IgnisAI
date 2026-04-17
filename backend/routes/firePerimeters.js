// backend/routes/firePerimeters.js
const express = require('express');
const axios = require('axios');

const router = express.Router();

const WFIGS_CURRENT_URL =
  process.env.WFIGS_CURRENT_PERIMETERS_URL ||
  'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query';

const FIRIS_PUBLIC_URL =
  process.env.FIRIS_PERIMETERS_URL ||
  'https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/CA_Perimeters_NIFC_FIRIS_public_view/FeatureServer/0/query';

function emptyFeatureCollection() {
  return { type: 'FeatureCollection', features: [] };
}

function parseBbox(raw) {
  if (!raw) return null;
  const parts = String(raw).split(',').map(v => Number(v.trim()));
  if (parts.length !== 4 || parts.some(v => !Number.isFinite(v))) return null;
  const [w, s, e, n] = parts;
  if (w >= e || s >= n || w < -180 || e > 180 || s < -90 || n > 90) return null;
  return { w, s, e, n };
}

function escapeSqlLike(value) {
  return String(value || '').replace(/'/g, "''");
}

function buildWhere(incidentName) {
  const name = String(incidentName || '').trim();
  if (!name) return '1=1';
  const escaped = escapeSqlLike(name).toUpperCase();
  return [
    `UPPER(poly_IncidentName) LIKE '%${escaped}%'`,
    `UPPER(attr_IncidentName) LIKE '%${escaped}%'`,
    `UPPER(IncidentName) LIKE '%${escaped}%'`,
    `UPPER(FIRE_NAME) LIKE '%${escaped}%'`,
    `UPPER(FireName) LIKE '%${escaped}%'`,
  ].join(' OR ');
}

function arcgisParams({ bbox }) {
  const params = {
    f: 'geojson',
    where: '1=1',
    outFields: '*',
    returnGeometry: true,
    outSR: 4326,
    resultRecordCount: 500,
  };

  if (bbox) {
    params.geometry = `${bbox.w},${bbox.s},${bbox.e},${bbox.n}`;
    params.geometryType = 'esriGeometryEnvelope';
    params.inSR = 4326;
    params.spatialRel = 'esriSpatialRelIntersects';
  }

  return params;
}

function asTimestamp(value) {
  if (value == null) return null;
  if (typeof value === 'number') return value;
  const ts = Date.parse(value);
  return Number.isFinite(ts) ? ts : null;
}

function featureDateCandidates(props = {}) {
  return [
    props.poly_DateCurrent,
    props.poly_CreateDate,
    props.poly_ModifiedOnDateTime_dt,
    props.attr_ModifiedOnDateTime_dt,
    props.attr_FireDiscoveryDateTime_dt,
    props.DateCurrent,
    props.CreateDate,
    props.EditDate,
    props.perimeterdatetime,
    props.missiontimestamp,
  ].map(asTimestamp).filter(v => v != null);
}

function filterByTime(features, from, to) {
  const fromTs = asTimestamp(from);
  const toTs = asTimestamp(to);
  if (fromTs == null && toTs == null) return features;

  return features.filter(feature => {
    const dates = featureDateCandidates(feature.properties || {});
    if (!dates.length) return true;
    return dates.some(ts => (
      (fromTs == null || ts >= fromTs) &&
      (toTs == null || ts <= toTs)
    ));
  });
}

function filterByIncidentName(features, incidentName) {
  const name = String(incidentName || '').trim().toUpperCase();
  if (!name) return features;
  return features.filter(feature => {
    const props = feature.properties || {};
    return Object.values(props).some(value => (
      typeof value === 'string' && value.toUpperCase().includes(name)
    ));
  });
}

async function fetchPerimetersFrom(url, params) {
  const { data } = await axios.get(url, { params, timeout: 20000 });
  if (data?.type === 'FeatureCollection') return data;
  return emptyFeatureCollection();
}

function mergeFeatureCollections(collections) {
  const seen = new Set();
  const features = [];
  for (const collection of collections) {
    for (const feature of collection.features || []) {
      const props = feature.properties || {};
      const id = props.OBJECTID || props.objectid || props.poly_SourceOID || props.irwinID || JSON.stringify(feature.geometry);
      const key = `${id}:${feature.geometry?.type || ''}`;
      if (seen.has(key)) continue;
      seen.add(key);
      features.push(feature);
    }
  }
  return { type: 'FeatureCollection', features };
}

router.get('/', async (req, res) => {
  const bbox = parseBbox(req.query.bbox);
  const params = arcgisParams({ bbox });

  try {
    const results = await Promise.allSettled([
      fetchPerimetersFrom(WFIGS_CURRENT_URL, params),
      fetchPerimetersFrom(FIRIS_PUBLIC_URL, params),
    ]);

    const collections = results
      .filter(result => result.status === 'fulfilled')
      .map(result => result.value);

    const merged = mergeFeatureCollections(collections);
    merged.features = filterByIncidentName(merged.features, req.query.incidentName);
    merged.features = filterByTime(merged.features, req.query.from, req.query.to);

    return res.json({
      geojson: merged,
      count: merged.features.length,
      sources: ['WFIGS', 'FIRIS'],
      caveat: 'Perimeters are authoritative when available, but may lag or be missing for active incidents.',
      partial: results.some(result => result.status === 'rejected'),
    });
  } catch (err) {
    console.error('fire-perimeters error:', err.message);
    return res.json({
      geojson: emptyFeatureCollection(),
      count: 0,
      sources: ['WFIGS', 'FIRIS'],
      caveat: 'Perimeter data is unavailable; continue showing FIRMS footprints.',
      partial: true,
    });
  }
});

module.exports = router;
module.exports._private = { parseBbox, buildWhere, filterByIncidentName, filterByTime, mergeFeatureCollections };
