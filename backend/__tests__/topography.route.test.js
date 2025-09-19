// backend/__tests__/topography.route.test.js

// 1) Register mocks FIRST (before requiring app)
jest.mock('axios', () => ({
  get: jest.fn(),
}));

jest.mock('../models/Topography', () => ({
  // Return the doc the route tries to "save", so response has data
  create: jest.fn(async (doc) => ({ _id: 'test-id', ...doc })),
}));

const request = require('supertest');
const axios = require('axios');
// 2) Now require the app (it will see the mocks)
const app = require('../app');

describe('GET /api/topography/point', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('returns elevation + flat slope when EPQS responds', async () => {
    // EPQS center + 4 neighbors; same elevation -> slope 0, aspect 0
    axios.get.mockResolvedValue({ data: { elevation: 100 } });

    const res = await request(app).get('/api/topography/point?lat=34&lon=-118');

    // Helpful debug if it ever fails again:
    // console.log(JSON.stringify(res.body, null, 2));

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('ok', true);
    expect(res.body).toHaveProperty('data');
    expect(res.body.data).toMatchObject({
      latitude: 34,
      longitude: -118,
      elevation: 100,
      slope_deg: 0,
      aspect_deg: 0,
    });
  });

  test('400 when lat/lon missing', async () => {
    const res = await request(app).get('/api/topography/point?lat=34');
    expect(res.status).toBe(400);
  });
});