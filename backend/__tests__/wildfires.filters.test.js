// backend/__tests__/wildfires.filters.test.js
//
// Flare and predictability filtering happens during ingest, which is where the
// FIRMS rows are parsed. The read endpoints serve what ingest stored, so these
// assertions target the ingest directly rather than reaching it through a
// request - a map load no longer triggers a fetch.
jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ bulkWrite: jest.fn(), find: jest.fn() }));

const axios = require('axios');
const Wildfire = require('../models/Wildfire');
const fireData = require('../routes/fireData');

const { fetchCurrentFires } = fireData._private;

const storedDocuments = () =>
  Wildfire.bulkWrite.mock.calls[0][0].map((op) => op.updateOne.update.$set);

describe('FIRMS ingest filtering', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // r1 = likely flare (night, low conf, low FRP, low brightness)
  // r2 = predictable (bright & confident)
  // r3 = not predictable (too dim / low confidence)
  const csv = [
    'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
    '34.111,-118.111,329.0,0.5,0.5,2024-01-15,0430,N21,VIIRS,low,2.0,320.0,10.0,N',
    '34.222,-118.222,345.0,0.5,0.5,2024-01-15,1430,N21,VIIRS,high,2.0,343.0,30.0,D',
    '34.333,-118.333,320.0,0.5,0.5,2024-01-15,1500,N21,VIIRS,low,2.0,318.0,15.0,D',
  ].join('\n');

  test('default behavior excludes likely flares', async () => {
    axios.get.mockResolvedValue({ data: csv });

    const result = await fetchCurrentFires({});

    expect(result.fires).toHaveLength(2);
  });

  test('excludeFlares=false keeps the flare', async () => {
    axios.get.mockResolvedValue({ data: csv });

    const result = await fetchCurrentFires({ excludeFlares: false });

    expect(result.fires).toHaveLength(3);
  });

  test('predictableOnly keeps only rows bright and confident enough to model', async () => {
    axios.get.mockResolvedValue({ data: csv });

    const result = await fetchCurrentFires({ predictableOnly: true });

    expect(result.fires).toHaveLength(1);
    expect(result.fires[0]).toMatchObject({ predictable: true });
    expect(typeof result.fires[0].confidence).toBe('number'); // mapped 0..100
    expect(result.fires[0]).toHaveProperty('brightnessCat');
  });

  test('ingest stores the parsed rows as idempotent upserts', async () => {
    axios.get.mockResolvedValue({ data: csv });
    Wildfire.bulkWrite.mockResolvedValue({ upsertedCount: 2 });

    await fireData.refreshDetections({ reason: 'test' });

    expect(Wildfire.bulkWrite).toHaveBeenCalledTimes(1);
    expect(storedDocuments()).toHaveLength(2);
  });
});
