// backend/__tests__/mongo.connection.test.js
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const connectDB = require('../db');
const Wildfire = require('../models/Wildfire');

const GeoSchema = new mongoose.Schema({
  name: { type: String, required: true },
  location: {
    type: {
      type: String,
      enum: ['Point'],
      required: true,
    },
    coordinates: {
      type: [Number],
      required: true,
    },
  },
});
GeoSchema.index({ location: '2dsphere' });
const TestLocation =
  mongoose.models.TestLocation || mongoose.model('TestLocation', GeoSchema);

describe('MongoDB Connection Integration', () => {
  let mongoServer;
  let originalEnv;

  beforeAll(async () => {
    originalEnv = { ...process.env };
    process.env.NODE_ENV = 'ci-mongo';
    mongoServer = await MongoMemoryServer.create();
    process.env.MONGODB_URI = mongoServer.getUri();
    await connectDB();
  });

  afterAll(async () => {
    await mongoose.connection.close();
    await mongoServer.stop();
    process.env = originalEnv;
  });

  beforeEach(async () => {
    await Wildfire.deleteMany({});
    await TestLocation.deleteMany({});
  });

  it('establishes a connection and responds to admin commands', async () => {
    expect(mongoose.connection.readyState).toBe(1); // connected

    const admin = mongoose.connection.db.admin();
    const ping = await admin.ping();
    expect(ping).toHaveProperty('ok', 1);

    const buildInfo = await admin.command({ buildInfo: 1 });
    expect(buildInfo).toHaveProperty('version');

    const poolSize =
      mongoose.connection.client?.options?.maxPoolSize ??
      mongoose.connection.client?.s?.options?.maxPoolSize ??
      0;
    expect(poolSize).toBeGreaterThan(0);
  });

  it('supports CRUD operations on the Wildfire collection', async () => {
    const payload = {
      latitude: 34.0522,
      longitude: -118.2437,
      brightness: 330.3,
      confidence: 80,
      satellite: 'VIIRS',
      instrument: 'Mock',
      timestamp: new Date(),
    };

    const created = await Wildfire.create(payload);
    expect(created._id).toBeDefined();

    const fetched = await Wildfire.findById(created._id);
    expect(fetched.latitude).toBe(payload.latitude);
    expect(fetched.longitude).toBe(payload.longitude);

    await Wildfire.updateOne(
      { _id: created._id },
      { $set: { confidence: 95 } }
    );
    const updated = await Wildfire.findById(created._id);
    expect(updated.confidence).toBe(95);

    await Wildfire.deleteOne({ _id: created._id });
    const count = await Wildfire.countDocuments();
    expect(count).toBe(0);
  });

  it('executes geospatial queries via 2dsphere index', async () => {
    await TestLocation.create([
      {
        name: 'Los Angeles',
        location: { type: 'Point', coordinates: [-118.2437, 34.0522] },
      },
      {
        name: 'San Francisco',
        location: { type: 'Point', coordinates: [-122.4194, 37.7749] },
      },
      {
        name: 'Seattle',
        location: { type: 'Point', coordinates: [-122.3321, 47.6062] },
      },
    ]);

    const results = await TestLocation.aggregate([
      {
        $geoNear: {
          near: { type: 'Point', coordinates: [-118.25, 34.05] },
          distanceField: 'distanceMeters',
          maxDistance: 600_000, // 600km
          spherical: true,
        },
      },
    ]);

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].name).toBe('Los Angeles');
    expect(results[0].distanceMeters).toBeLessThan(20_000); // ~20km radius
  });
});
