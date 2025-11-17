// backend/__tests__/wildfires.filters.test.js
const request = require('supertest');

// Mock BEFORE requiring app
jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ insertMany: jest.fn() }));

const axios = require('axios');
const Wildfire = require('../models/Wildfire');
const app = require('../app');

describe('GET /api/wildfires filtering flags', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // CSV rows:
  // r1 = likely flare (night, low conf, low FRP, low brightness)
  // r2 = predictable (bright & confident)
  // r3 = not predictable (too dim / low confidence)
  const csv = [
    'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
    '34.111,-118.111,329.0,0.5,0.5,2024-01-15,0430,N21,VIIRS,low,2.0,320.0,10.0,N', // flare candidate
    '34.222,-118.222,345.0,0.5,0.5,2024-01-15,1430,N21,VIIRS,high,2.0,343.0,30.0,D', // predictable
    '34.333,-118.333,320.0,0.5,0.5,2024-01-15,1500,N21,VIIRS,low,2.0,318.0,15.0,D'   // not predictable
  ].join('\n');

  test('default behavior excludes likely flares (count = 2)', async () => {
    axios.get.mockResolvedValue({ data: csv });
    Wildfire.insertMany.mockResolvedValue([{_id:'x1'},{_id:'x2'}]); // 2 inserts

    const res = await request(app).get('/api/wildfires').expect(200);

    expect(res.body).toHaveProperty('message');
    expect(res.body).toHaveProperty('count', 2);
    expect(Wildfire.insertMany).toHaveBeenCalledTimes(1);
    const docs = Wildfire.insertMany.mock.calls[0][0];
    expect(Array.isArray(docs)).toBe(true);
    expect(docs).toHaveLength(2); // flare removed
  });

  test('excludeFlares=false includes flare (count = 3)', async () => {
    axios.get.mockResolvedValue({ data: csv });
    Wildfire.insertMany.mockResolvedValue([{_id:'x1'},{_id:'x2'},{_id:'x3'}]);

    const res = await request(app)
      .get('/api/wildfires')
      .query({ excludeFlares: 'false' })
      .expect(200);

    expect(res.body.count).toBe(3);
    const docs = Wildfire.insertMany.mock.calls[0][0];
    expect(docs).toHaveLength(3);
  });

  test('predictableOnly=true returns only predictable rows (count = 1)', async () => {
    axios.get.mockResolvedValue({ data: csv });
    Wildfire.insertMany.mockResolvedValue([{_id:'p1'}]);

    const res = await request(app)
      .get('/api/wildfires')
      .query({ predictableOnly: 'true' })
      .expect(200);

    expect(res.body.count).toBe(1);
    const docs = Wildfire.insertMany.mock.calls[0][0];
    expect(docs).toHaveLength(1);
    // sanity check on derived fields we rely on in UI:
    expect(docs[0]).toHaveProperty('confidence');
    expect(typeof docs[0].confidence).toBe('number'); // mapped 0..100
    expect(docs[0]).toHaveProperty('brightnessCat');  // category present
    expect(docs[0]).toHaveProperty('predictable', true);
  });
});