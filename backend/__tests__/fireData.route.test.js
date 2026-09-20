// backend/__tests__/fireData.route.test.js
const request = require('supertest');

jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ bulkWrite: jest.fn(), find: jest.fn() }));

const axios = require('axios');
const Wildfire = require('../models/Wildfire');
const app = require('../app');
const fireDataRoute = require('../routes/fireData');

/** Mongoose chain stub: find().sort().limit().lean() resolves to docs. */
function mockFind(docs) {
  const chain = {
    sort: () => chain,
    limit: () => chain,
    lean: () => chain,
    then: (resolve, reject) => Promise.resolve(docs).then(resolve, reject),
  };
  Wildfire.find.mockReturnValue(chain);
  return chain;
}

const storedFire = (over = {}) => ({
  latitude: 34.0522,
  longitude: -118.2437,
  brightness: 340,
  confidence: 90,
  satellite: 'N21',
  instrument: 'VIIRS',
  scan: 0.4,
  track: 0.5,
  frp: 12.3,
  timestamp: new Date(),
  ...over,
});

describe('GET /api/wildfires', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    fireDataRoute._private.ingest.lastRunAt = Date.now();
  });

  it('serves stored detections without calling FIRMS', async () => {
    mockFind([storedFire(), storedFire({ longitude: -118.1234 })]);

    const res = await request(app).get('/api/wildfires').expect(200);

    expect(res.body.count).toBe(2);
    // The whole point of the split: a map load costs one indexed query.
    expect(axios.get).not.toHaveBeenCalled();
    expect(Wildfire.bulkWrite).not.toHaveBeenCalled();
  });

  it('scopes the query to the requested viewport and time window', async () => {
    mockFind([]);

    await request(app)
      .get('/api/wildfires')
      .query({ bbox: '-119,33,-117,35', days: 1 })
      .expect(200);

    const filter = Wildfire.find.mock.calls[0][0];
    expect(filter.longitude).toMatchObject({ $gte: -119, $lte: -117 });
    expect(filter.latitude).toMatchObject({ $gte: 33, $lte: 35 });
    expect(filter.timestamp.$gte).toBeInstanceOf(Date);
  });

  it('falls back to the configured bbox when the viewport is nonsense', async () => {
    mockFind([]);

    await request(app).get('/api/wildfires').query({ bbox: 'not,a,bbox,at all' }).expect(200);

    const filter = Wildfire.find.mock.calls[0][0];
    expect(Number.isFinite(filter.longitude.$gte)).toBe(true);
    expect(filter.longitude.$gte).toBeLessThan(filter.longitude.$lte);
  });

  it('reports ingest freshness so a stale archive is visible', async () => {
    mockFind([storedFire()]);

    const res = await request(app).get('/api/wildfires').expect(200);

    expect(res.body.ingest).toHaveProperty('lastRunAt');
    expect(res.body.ingest).toHaveProperty('ageSeconds');
  });

  it('returns FIRMS detection footprints as valid GeoJSON polygons', async () => {
    mockFind([storedFire()]);

    const res = await request(app).get('/api/wildfires/footprints').expect(200);

    expect(res.body).toMatchObject({ count: 1, geojson: { type: 'FeatureCollection' } });
    const feature = res.body.geojson.features[0];
    expect(feature.geometry.type).toBe('Polygon');
    expect(feature.geometry.coordinates[0]).toHaveLength(5);
    expect(feature.properties).toMatchObject({
      scan: 0.4,
      track: 0.5,
      frp: 12.3,
      footprint_source: 'firms_scan_track',
    });
    expect(axios.get).not.toHaveBeenCalled();
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

describe('FIRMS ingest', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    fireDataRoute._private.ingest.lastRunAt = null;
    fireDataRoute._private.ingest.running = false;
  });

  const csv = [
    'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
    '34.0522,-118.2437,330.5,0.4,0.4,2024-01-15,1430,N21,VIIRS,high,2.0,328.5,12.3,D',
    '33.7749,-118.1234,345.2,0.5,0.5,2024-01-15,1435,N21,VIIRS,nominal,2.0,343.1,18.7,D',
  ].join('\n');

  it('parses FIRMS rows and upserts them by detection identity', async () => {
    axios.get.mockResolvedValue({ data: csv });
    Wildfire.bulkWrite.mockResolvedValue({ upsertedCount: 2 });

    const status = await fireDataRoute.refreshDetections({ reason: 'test' });

    expect(status.parsed).toBe(2);
    const operations = Wildfire.bulkWrite.mock.calls[0][0];
    expect(operations).toHaveLength(2);
    expect(operations[0].updateOne.upsert).toBe(true);
    expect(operations[0].updateOne.filter).toMatchObject({
      latitude: 34.0522,
      longitude: -118.2437,
      satellite: 'N21',
      instrument: 'VIIRS',
    });
  });

  it('writes nothing when FIRMS returns only a header', async () => {
    axios.get.mockResolvedValue({
      data: 'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
    });

    await fireDataRoute.refreshDetections({ reason: 'test' });

    expect(Wildfire.bulkWrite).not.toHaveBeenCalled();
  });

  it('records a provider outage instead of throwing into the request path', async () => {
    mockFind([]);
    axios.get.mockRejectedValue(new Error('NASA API unavailable'));

    const status = await fireDataRoute.refreshDetections({ reason: 'test' });

    expect(status.error).toMatch(/FIRMS fetch failed/i);
    expect(Wildfire.bulkWrite).not.toHaveBeenCalled();
  });

  it('does not run two ingests at once', async () => {
    fireDataRoute._private.ingest.running = true;

    await fireDataRoute.refreshDetections({ reason: 'test' });

    expect(axios.get).not.toHaveBeenCalled();
  });
});
