// backend/__tests__/fireData.route.test.js
const request = require('supertest');

// --- Mocks (declare BEFORE requiring app) ---
jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ bulkWrite: jest.fn(), find: jest.fn() }));

const axios = require('axios');
const Wildfire = require('../models/Wildfire');
const app = require('../app');
const fireDataRoute = require('../routes/fireData');

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
    Wildfire.bulkWrite.mockResolvedValue({ upsertedCount: 2 });

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('message', 'Wildfire data fetched & stored');
    expect(res.body).toHaveProperty('count', 2);
    // we don’t assert on the exact shape of each doc to avoid tight-coupling
    expect(Wildfire.bulkWrite).toHaveBeenCalledTimes(1);
    const operations = Wildfire.bulkWrite.mock.calls[0][0];
    expect(Array.isArray(operations)).toBe(true);
    expect(operations).toHaveLength(2);
    expect(operations[0].updateOne.upsert).toBe(true);
    expect(operations[0].updateOne.filter).toMatchObject({
      latitude: 34.0522,
      longitude: -118.2437,
      satellite: 'N21',
      instrument: 'VIIRS',
    });
    expect(operations[0].updateOne.update.$set).toMatchObject({
      scan: 0.4,
      track: 0.4,
      frp: 12.3,
      instrument: 'VIIRS',
      satellite: 'N21',
      product: 'VIIRS_NOAA21_NRT',
      daynight: 'D',
    });
  });

  it('handles empty CSV (header only)', async () => {
    const csv = 'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight';
    axios.get.mockResolvedValue({ data: csv });

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ message: 'No valid fire data', count: 0, data: [] });
    expect(Wildfire.bulkWrite).not.toHaveBeenCalled();
  });

  it('propagates provider errors (500)', async () => {
    Wildfire.find.mockResolvedValue([]);
    axios.get.mockRejectedValue(new Error('NASA API unavailable'));

    const res = await request(app).get('/api/wildfires');

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({
      message: 'FIRMS fetch failed; no cached wildfire data',
      count: 0,
      data: [],
      stale: true,
    });
    expect(Wildfire.bulkWrite).not.toHaveBeenCalled();
  });

  it('returns FIRMS detection footprints as valid GeoJSON polygons', async () => {
    const csv = [
      'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
      '34.0522,-118.2437,330.5,0.4,0.5,2024-01-15,1430,N21,VIIRS,high,2.0,328.5,12.3,D'
    ].join('\n');
    axios.get.mockResolvedValue({ data: csv });

    const res = await request(app).get('/api/wildfires/footprints');

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({
      count: 1,
      geojson: { type: 'FeatureCollection' },
    });
    const feature = res.body.geojson.features[0];
    expect(feature.geometry.type).toBe('Polygon');
    expect(feature.geometry.coordinates[0]).toHaveLength(5);
    expect(feature.properties).toMatchObject({
      scan: 0.4,
      track: 0.5,
      frp: 12.3,
      footprint_source: 'firms_scan_track',
    });
  });

  it('falls back to nominal MODIS footprint dimensions when scan/track are absent', () => {
    const feature = fireDataRoute._private.fireToFootprintFeature({
      latitude: 34,
      longitude: -118,
      brightness: 340,
      confidence: 90,
      instrument: 'MODIS',
    });

    expect(feature.properties.footprint_source).toBe('MODIS_1KM');
    expect(feature.properties.scan).toBe(1);
    expect(feature.properties.track).toBe(1);
  });
});
