// backend/__tests__/geospatial-storage.test.js
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const Wildfire = require('../models/Wildfire');

let mongoServer;

describe('Geospatial Data Storage', () => {
  beforeAll(async () => {
    mongoServer = await MongoMemoryServer.create();
    await mongoose.connect(mongoServer.getUri());
  });

  afterAll(async () => {
    await mongoose.connection.close();
    await mongoServer.stop();
  });

  beforeEach(async () => {
    await Wildfire.deleteMany({});
  });

  it('should store fire location data with coordinates', async () => {
    const fireData = {
      latitude: 34.0522,
      longitude: -118.2437,
      brightness: 350.5,
      confidence: 85,
      acq_date: new Date('2025-11-14'),
      satellite: 'VIIRS',
      timestamp: new Date() // Added required field
    };

    const fire = await Wildfire.create(fireData);

    expect(fire.latitude).toBe(34.0522);
    expect(fire.longitude).toBe(-118.2437);
    expect(fire._id).toBeDefined();
  });

  it('should query data within geographic bounds', async () => {
    // Create multiple fire points
    await Wildfire.create([
      { 
        latitude: 34.0, 
        longitude: -118.0, 
        brightness: 300, 
        confidence: 80, 
        acq_date: new Date(), 
        satellite: 'VIIRS',
        timestamp: new Date() 
      },
      { 
        latitude: 34.5, 
        longitude: -118.5, 
        brightness: 320, 
        confidence: 85, 
        acq_date: new Date(), 
        satellite: 'VIIRS',
        timestamp: new Date() 
      },
      { 
        latitude: 35.0, 
        longitude: -119.0, 
        brightness: 310, 
        confidence: 75, 
        acq_date: new Date(), 
        satellite: 'MODIS',
        timestamp: new Date() 
      }
    ]);

    // Query within bounds
    const fires = await Wildfire.find({
      latitude: { $gte: 34.0, $lte: 34.6 },
      longitude: { $gte: -118.6, $lte: -117.5 }
    });

    expect(fires.length).toBeGreaterThanOrEqual(1);
    expect(fires.length).toBeLessThanOrEqual(2);
  });

  it('should validate data integrity', async () => {
    const fireData = {
      latitude: 34.0522,
      longitude: -118.2437,
      brightness: 350.5,
      confidence: 85,
      acq_date: new Date(),
      satellite: 'VIIRS',
      timestamp: new Date()
    };

    const fire = await Wildfire.create(fireData);
    const retrieved = await Wildfire.findById(fire._id);

    expect(retrieved.latitude).toBe(fire.latitude);
    expect(retrieved.longitude).toBe(fire.longitude);
    expect(retrieved.brightness).toBe(fire.brightness);
  });

  it('should handle spatial queries efficiently', async () => {
    // Create 100 test fires
    const fires = Array.from({ length: 100 }, (_, i) => ({
      latitude: 34 + (i * 0.01),
      longitude: -118 + (i * 0.01),
      brightness: 300 + i,
      confidence: 70 + (i % 30),
      acq_date: new Date(),
      satellite: i % 2 === 0 ? 'VIIRS' : 'MODIS',
      timestamp: new Date()
    }));

    await Wildfire.insertMany(fires);

    const startTime = Date.now();
    const nearbyFires = await Wildfire.find({
      latitude: { $gte: 34.0, $lte: 34.5 },
      longitude: { $gte: -118.5, $lte: -118.0 }
    });
    const queryTime = Date.now() - startTime;

    expect(nearbyFires.length).toBeGreaterThan(0);
    expect(queryTime).toBeLessThan(1000); // Should complete within 1 second
  });
});