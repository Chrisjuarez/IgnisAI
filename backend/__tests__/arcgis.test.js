jest.mock('axios', () => ({ get: jest.fn() }));

const axios = require('axios');
const { fetchArcgisGeoJson } = require('../utils/arcgis');

const URL = 'https://services.arcgis.example/FeatureServer/0/query';

describe('fetchArcgisGeoJson', () => {
  it('returns a FeatureCollection as it came, empty ones included', async () => {
    const empty = { type: 'FeatureCollection', features: [] };
    axios.get.mockResolvedValue({ data: empty });

    await expect(fetchArcgisGeoJson(URL, {})).resolves.toBe(empty);
  });

  it('fails on the error envelope ArcGIS sends with HTTP 200', async () => {
    axios.get.mockResolvedValue({ data: { error: { code: 400, message: 'Invalid query parameters' } } });

    await expect(fetchArcgisGeoJson(URL, {})).rejects.toThrow('ArcGIS query failed (400): Invalid query parameters');
  });

  it.each([
    ['an HTML error page', '<html>Service Unavailable</html>'],
    ['JSON that is not GeoJSON', { features: [] }],
    ['an empty body', ''],
  ])('fails on %s', async (_label, data) => {
    axios.get.mockResolvedValue({ data });

    await expect(fetchArcgisGeoJson(URL, {})).rejects.toThrow('ArcGIS reply was not GeoJSON');
  });
});
