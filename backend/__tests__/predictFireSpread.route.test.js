// __tests__/predictFireSpread.route.test.js
jest.mock('axios', () => ({ get: jest.fn() }));

const request = require('supertest');
const axios   = require('axios');
const app     = require('../app');

describe('POST /api/predict-fire-spread → tilesvc proxy', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns overlay artifacts when tilesvc succeeds', async () => {
    // Mock tilesvc endpoints
    axios.get.mockImplementation((url, { params } = {}) => {
      if (url.includes('/predict_geojson')) {
        return Promise.resolve({
          data: {
            type: 'FeatureCollection',
            features: [
              {
                type: 'Feature',
                properties: { type: 'spread', probability: 0.4, wind_dir: 180 },
                geometry: {
                  type: 'Polygon',
                  coordinates: [[[0,0],[1,0],[1,1],[0,1],[0,0]]]
                }
              }
            ]
          }
        });
      }
      if (url.includes('/predict') && params?.png === false) {
        return Promise.resolve({ data: { area_fraction: 0.42, threshold: 0.5 } });
      }
      return Promise.reject(new Error('axios mock: unexpected URL ' + url));
    });

    const res = await request(app)
      .post('/api/predict-fire-spread')
      .send({ lat: 34.05, lon: -118.25, Tseq: 3 })   // send lat + lon explicitly
      .timeout(5000);

    expect([200, 201]).toContain(res.status);
    expect(res.body).toMatchObject({
      ok: true,
      threshold: expect.any(Number),
      spread_probability: expect.any(Number),
      spread_distance_km: expect.any(Number),
    });
    expect(res.body.geojson?.type).toBe('FeatureCollection');
  });

  it('propagates tilesvc errors as 5xx JSON', async () => {
    // First call to /predict_geojson fails
    axios.get.mockRejectedValueOnce(new Error('tilesvc down'));

    const res = await request(app)
      .post('/api/predict-fire-spread')
      .send({ lat: 34.05, lon: -118.25, Tseq: 3 })
      .timeout(5000);

    expect([500, 502, 503]).toContain(res.status);
    expect(res.body).toHaveProperty('error');
  });
});