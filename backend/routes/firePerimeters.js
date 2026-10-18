// backend/routes/firePerimeters.js
const express = require('express');
const { fetchArcgisGeoJson } = require('../utils/arcgis');
const { bboxOf } = require('../utils/geo');

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

// Perimeters arrive digitised far finer than the map can draw. At the
// western-US view the map opens on, full precision is 50 MB of GeoJSON, and
// parsing and re-serialising it lifts process memory by about 450 MB. ArcGIS
// simplifies server side at about 9 m to 7.3 MB, within a pixel at zoom 14.
// That bound matters because the map draws these and does not refetch on zoom.
const PERIMETER_GEOMETRY_TOLERANCE_DEG = 0.0001;

function arcgisParams({ bbox }) {
  const params = {
    f: 'geojson',
    where: '1=1',
    outFields: '*',
    returnGeometry: true,
    outSR: 4326,
    resultRecordCount: 500,
    maxAllowableOffset: PERIMETER_GEOMETRY_TOLERANCE_DEG,
    geometryPrecision: 6,
  };

  if (bbox) {
    params.geometry = `${bbox.w},${bbox.s},${bbox.e},${bbox.n}`;
    params.geometryType = 'esriGeometryEnvelope';
    params.inSR = 4326;
    params.spatialRel = 'esriSpatialRelIntersects';
  }

  return params;
}

// FIRIS keeps a fire's perimeters once it is over and marks them Inactive,
// about when WFIGS Current drops the fire. On 2026-10-18, 238 of the 315
// perimeters drawn at the western-US view were Inactive FIRIS ones, Yosemite
// fires out since August among them. Left out server side, they also stop
// counting toward resultRecordCount: FIRIS held 488 rows, 500 are asked for.
// The status is today's, so a past window still gets every perimeter.
const FIRIS_ACTIVE_WHERE = "displayStatus IS NULL OR displayStatus <> 'Inactive'";

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

// When a perimeter's shape was mapped. On WFIGS, poly_DateCurrent is when the
// record was last edited - for the Dome Fire, three days after its shape was
// mapped - and poly_PolygonDateTime is the mapping. FIRIS dates each flight in
// poly_DateCurrent, and a few of its perimeters only in CreationDate.
const MAPPED_AT_FIELDS = ['poly_PolygonDateTime', 'poly_DateCurrent', 'CreationDate', 'poly_CreateDate'];
const NAME_FIELDS = ['poly_IncidentName', 'attr_IncidentName', 'IncidentName', 'incident_name', 'FIRE_NAME', 'FireName'];

function firstValue(props, names) {
  for (const name of names) {
    const value = props[name];
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

// FIRIS mission codes read state-unit-name-aircraft, as CA-ANF-BOQUET-N50X. A
// FIRIS perimeter with no incident name carries its fire's name only there.
function missionName(mission) {
  const [state, unit, name] = String(mission || '').split('-');
  return /^[A-Z]{2}$/.test(state) && /^[A-Z]+$/.test(unit || '') ? name || null : null;
}

function perimeterIdentity(feature) {
  const props = feature.properties || {};
  const irwinId = firstValue(props, ['attr_IrwinID', 'poly_IRWINID']);
  return {
    feature,
    mappedAt: asTimestamp(firstValue(props, MAPPED_AT_FIELDS)),
    irwinId: irwinId ? String(irwinId).replace(/[{}]/g, '').trim().toUpperCase() : null,
    name: normalizeName(firstValue(props, NAME_FIELDS) || missionName(props.mission)),
    bbox: bboxOf(feature.geometry),
  };
}

function bboxesOverlap(a, b) {
  return Boolean(a && b) && a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
}

// Whether two perimeters map the same fire. An IRWIN ID names one incident, so
// when both carry one it decides; otherwise a fire is its name where it burns.
function sameFire(a, b) {
  if (a.irwinId && b.irwinId) return a.irwinId === b.irwinId;
  return Boolean(a.name) && a.name === b.name && bboxesOverlap(a.bbox, b.bbox);
}

// Whether `other` makes `entry` history: the same fire, mapped later - or at
// the same moment by WFIGS, which FIRIS republishes as NIFC rows without the
// IRWIN ID. The pieces one flight mapped of a fire share a time and all stay.
function supersedes(other, entry) {
  if (!sameFire(other, entry)) return false;
  if (other.mappedAt > entry.mappedAt) return true;
  return other.mappedAt === entry.mappedAt && Boolean(other.irwinId) && !entry.irwinId;
}

// Each fire as it was last mapped. FIRIS keeps every flight's perimeter, and a
// month of them drawn together hid the Dome Fire's shape under their own. A
// perimeter with no date to order it by stays.
function latestPerimeters(features) {
  const entries = features.map(perimeterIdentity);
  return entries
    .filter((entry) => entry.mappedAt == null || !entries.some((other) => supersedes(other, entry)))
    .map((entry) => entry.feature);
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
  const pastWindow = asTimestamp(req.query.from) != null || asTimestamp(req.query.to) != null;

  try {
    const results = await Promise.allSettled([
      fetchArcgisGeoJson(WFIGS_CURRENT_URL, params),
      fetchArcgisGeoJson(FIRIS_PUBLIC_URL, pastWindow ? params : { ...params, where: FIRIS_ACTIVE_WHERE }),
    ]);

    const collections = results
      .filter(result => result.status === 'fulfilled')
      .map(result => result.value);

    const merged = mergeFeatureCollections(collections);
    merged.features = filterByIncidentName(merged.features, req.query.incidentName);
    merged.features = filterByTime(merged.features, req.query.from, req.query.to);
    merged.features = latestPerimeters(merged.features);

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
