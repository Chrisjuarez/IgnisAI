const request = require('supertest');

jest.mock('axios', () => ({ get: jest.fn() }));

const axios = require('axios');
const app = require('../app');

describe('GET /api/fire-perimeters', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns an empty FeatureCollection when authoritative sources are unavailable', async () => {
    axios.get.mockRejectedValue(new Error('ArcGIS unavailable'));

    const res = await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-119,33,-117,35', incidentName: 'Palisades' })
      .expect(200);

    expect(res.body).toMatchObject({
      geojson: { type: 'FeatureCollection', features: [] },
      count: 0,
      partial: true,
    });
  });

  it('merges available perimeter FeatureCollections', async () => {
    axios.get
      .mockResolvedValueOnce({
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: { OBJECTID: 1, poly_IncidentName: 'Palisades' },
              geometry: { type: 'Polygon', coordinates: [[[-118.6, 34], [-118.5, 34], [-118.5, 34.1], [-118.6, 34.1], [-118.6, 34]]] },
            },
          ],
        },
      })
      .mockRejectedValueOnce(new Error('secondary unavailable'));

    const res = await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-119,33,-117,35', incidentName: 'Palisades' })
      .expect(200);

    expect(res.body.count).toBe(1);
    expect(res.body.partial).toBe(true);
    expect(res.body.geojson.features[0].properties.poly_IncidentName).toBe('Palisades');
  });

  it('asks every source for simplified geometry so a wide viewport stays small', async () => {
    axios.get.mockResolvedValue({ data: { type: 'FeatureCollection', features: [] } });

    await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-125.1,31,-101.8,49.5' })
      .expect(200);

    expect(axios.get).toHaveBeenCalledTimes(2);
    for (const [, config] of axios.get.mock.calls) {
      expect(config.params).toMatchObject({ maxAllowableOffset: 0.0001, geometryPrecision: 6 });
    }
  });
});
