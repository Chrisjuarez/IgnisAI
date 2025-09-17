const express = require('express');
const request = require('supertest');

// Mock external modules used by the route
jest.mock('canvas', () => {
  const getContext = () => ({
    drawImage: jest.fn(),
    // Ignore x/y and just return constant pixel data so elevation is deterministic
    getImageData: () => ({ data: [0, 0, 0, 255] }), // elevation => -10000, slope => 0
  });
  return {
    createCanvas: () => ({ getContext }),
    loadImage: () => Promise.resolve({ width: 5, height: 5 }),
  };
});

jest.mock('axios', () => ({
  post: jest.fn().mockResolvedValue({
    data: { elements: [{ tags: { landuse: 'forest' } }] },
  }),
}));

jest.mock('../models/Topography', () => ({
  findOneAndUpdate: jest.fn().mockImplementation((filter, update) => {
    // Echo back a doc resembling what Mongo would return
    return Promise.resolve({
      location: update.$set.location,
      elevation: update.$set.elevation,
      slope: update.$set.slope,
      vegetationType: update.$set.vegetationType,
    });
  }),
}));

describe('GET /api/topography', () => {
  let app;

  beforeAll(() => {
    // Ensure token is set to avoid route complaining about env var
    process.env.MAPBOX_ACCESS_TOKEN = 'test-token';

    const router = require('../routes/topography');
    app = express();
    app.use('/api/topography', router);
  });

  it('responds with upserted topography data', async () => {
    const lat = 12.34;
    const lon = 56.78;

    const res = await request(app)
      .get('/api/topography')
      .query({ lat, lon });

    expect(res.status).toBe(200);
    expect(res.body.message).toBe('ok');

    // From mocks: elevation = -10000 (all zeros RGB), slope = 0, landuse = 'forest'
    expect(res.body.data).toMatchObject({
      location: { type: 'Point', coordinates: [lon, lat] },
      elevation: -10000,
      slope: 0,
      vegetationType: 'forest',
    });
  });
});

