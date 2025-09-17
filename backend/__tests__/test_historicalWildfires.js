const express = require('express');
const request = require('supertest');

// Mock external dependencies used by the route
jest.mock('axios', () => ({
  get: jest.fn().mockResolvedValue({ data: '' }), // empty CSV to skip parsing/inserts
}));

// Route requires csv-parser at module load; provide a harmless mock
jest.mock('csv-parser', () => () => {
  const { Transform } = require('stream');
  // No-op transform; not used in this test because CSV is empty
  return new Transform();
});

// Mock the Mongoose model to avoid requiring mongoose
jest.mock('../models/HistoricalWildfire', () => ({
  insertMany: jest.fn().mockResolvedValue(undefined),
}));

describe('GET /historical-wildfires', () => {
  let app;
  const axios = require('axios');
  const Model = require('../models/HistoricalWildfire');

  beforeAll(() => {
    process.env.NASA_API_KEY = 'test-key';
    const router = require('../routes/historicalWildfires');
    app = express();
    app.use('/', router);
  });

  it('calls NASA FIRMS per chunk and returns zero totals on empty CSV', async () => {
    const res = await request(app).get('/historical-wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('message');
    expect(res.body).toHaveProperty('results');

    // Two historical fires, each chunked into 3 ranges (10,10,remaining)
    expect(axios.get).toHaveBeenCalledTimes(6);

    // No records inserted because CSV was empty for all chunks
    expect(Model.insertMany).not.toHaveBeenCalled();

    // Validate aggregated results are zero for each fire
    const results = res.body.results;
    expect(Array.isArray(results)).toBe(true);
    expect(results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ fireName: 'Eaton Fire', totalRecords: 0 }),
        expect.objectContaining({ fireName: 'Palisades', totalRecords: 0 }),
      ])
    );
  });
});

