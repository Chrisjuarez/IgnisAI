// backend/__tests__/fireData.route.test.js
const request = require('supertest');

// --- Mocks (declare BEFORE requiring app) ---
jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ insertMany: jest.fn() }));

const axios = require('axios');
const Wildfire = require('../models/Wildfire');
const app = require('../app');

describe('GET /api/wildfires', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('fetches FIRMS CSV, parses, stores, and returns count', async () => {
    // Minimal valid CSV: header + 2 rows (14 columns)
    const csv = [
      'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
      '34.0522,-118.2437,330.5,0.4,0.4,2024-01-15,1430,N21,VIIRS,high,2.0,328.5,12.3,D',
      '33.7749,-118.1234,345.2,0.5,0.5,2024-01-15,1435,N21,VIIRS,nominal,2.0,343.1,18.7,D'
    ].join('\n');

    axios.get.mockResolvedValue({ data: csv });
    Wildfire.insertMany.mockResolvedValue([
      { _id: 'a1' }, { _id: 'a2' }
    ]);

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('message', 'Wildfire data fetched & stored');
    expect(res.body).toHaveProperty('count', 2);
    // we don’t assert on the exact shape of each doc to avoid tight-coupling
    expect(Wildfire.insertMany).toHaveBeenCalledTimes(1);
    const toInsert = Wildfire.insertMany.mock.calls[0][0];
    expect(Array.isArray(toInsert)).toBe(true);
    expect(toInsert).toHaveLength(2);
  });

  it('handles empty CSV (header only)', async () => {
    const csv = 'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight';
    axios.get.mockResolvedValue({ data: csv });

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ message: 'No valid fire data', count: 0, data: [] });
    expect(Wildfire.insertMany).not.toHaveBeenCalled();
  });

  it('propagates provider errors (500)', async () => {
    axios.get.mockRejectedValue(new Error('NASA API unavailable'));

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(500);
    expect(res.body).toHaveProperty('error', 'NASA API unavailable');
    expect(Wildfire.insertMany).not.toHaveBeenCalled();
  });
});