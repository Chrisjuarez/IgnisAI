// backend/utils/arcgis.js
//
// GeoJSON from an ArcGIS feature service. ArcGIS reports a failed query as an
// HTTP 200 whose body is { error: { code, message } }; read as data, that was
// an empty map reported as healthy. Here it is a failure, and so is any other
// reply that is not a FeatureCollection. A FeatureCollection with no features
// is a legitimate empty result.

const axios = require('axios');

const ARCGIS_TIMEOUT_MS = 25000;
const USER_AGENT = process.env.IGNIS_USER_AGENT || 'IgnisAI wildfire map';

async function fetchArcgisGeoJson(url, params) {
  const { data } = await axios.get(url, {
    params,
    timeout: ARCGIS_TIMEOUT_MS,
    headers: { 'User-Agent': USER_AGENT },
  });
  if (data?.type === 'FeatureCollection') return data;
  if (data?.error) {
    throw new Error(`ArcGIS query failed (${data.error.code ?? 'no code'}): ${data.error.message || 'no message'}`);
  }
  throw new Error('ArcGIS reply was not GeoJSON');
}

module.exports = { fetchArcgisGeoJson };
