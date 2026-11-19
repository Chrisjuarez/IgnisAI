// backend/__tests__/wildfire.model.test.js
//
// Stored FIRMS detections expire through a TTL index on `timestamp`. The
// retention is read from WILDFIRE_RETENTION_DAYS when the model is defined, so
// each scenario loads a fresh copy via jest.isolateModules.

const { MongoMemoryServer } = require('mongodb-memory-server');

const DAY_S = 24 * 60 * 60;

function loadWildfire() {
  let Wildfire;
  jest.isolateModules(() => {
    Wildfire = require('../models/Wildfire');
  });
  return Wildfire;
}

function expireAfterSeconds(Wildfire) {
  const [, options] = Wildfire.schema.indexes()
    .find(([fields]) => Object.keys(fields).join() === 'timestamp');
  return options.expireAfterSeconds;
}

describe('Wildfire detection retention', () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    process.env = { ...ORIGINAL_ENV };
    delete process.env.WILDFIRE_RETENTION_DAYS;
  });

  afterAll(() => {
    process.env = ORIGINAL_ENV;
  });

  test('expires detections 30 days after they were observed by default', () => {
    expect(expireAfterSeconds(loadWildfire())).toBe(30 * DAY_S);
  });

  test('WILDFIRE_RETENTION_DAYS sets the retention', () => {
    process.env.WILDFIRE_RETENTION_DAYS = '90';
    expect(expireAfterSeconds(loadWildfire())).toBe(90 * DAY_S);
  });

  test.each(['0', '-30', '2'])('WILDFIRE_RETENTION_DAYS=%s keeps the 7-day minimum', (value) => {
    process.env.WILDFIRE_RETENTION_DAYS = value;
    expect(expireAfterSeconds(loadWildfire())).toBe(7 * DAY_S);
  });

  test('a WILDFIRE_RETENTION_DAYS that is not a number falls back to 30 days', () => {
    process.env.WILDFIRE_RETENTION_DAYS = 'thirty';
    expect(expireAfterSeconds(loadWildfire())).toBe(30 * DAY_S);
  });

  test('MongoDB builds timestamp_1 as a TTL index', async () => {
    const mongoServer = await MongoMemoryServer.create();
    const Wildfire = loadWildfire();
    try {
      await Wildfire.base.connect(mongoServer.getUri());
      await Wildfire.init();
      const indexes = await Wildfire.collection.indexes();
      expect(indexes.find((index) => index.name === 'timestamp_1')).toMatchObject({
        key: { timestamp: 1 },
        expireAfterSeconds: 30 * DAY_S,
      });
    } finally {
      await Wildfire.base.disconnect();
      await mongoServer.stop();
    }
  });
});
