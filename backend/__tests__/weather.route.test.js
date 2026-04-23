// backend/__tests__/weather.route.test.js

jest.mock('axios', () => ({
  get: jest.fn(),
}));

jest.mock('../models/Weather', () => ({
  create: jest.fn(),
}));

const request = require('supertest');
const axios = require('axios');
const Weather = require('../models/Weather');
const mongoose = require('mongoose');
const app = require('../app');

describe('GET /api/weather/current', () => {
  let originalReadyState;

  beforeAll(() => {
    // Save original readyState so we can restore it
    originalReadyState = mongoose.connection.readyState;
  });

  beforeEach(() => {
    jest.clearAllMocks();

    // 🔑 Force MongoDB to appear "connected"
    Object.defineProperty(mongoose.connection, 'readyState', {
      value: 1,
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    // Restore original readyState after each test
    Object.defineProperty(mongoose.connection, 'readyState', {
      value: originalReadyState,
      writable: true,
      configurable: true,
    });
  });

  test('returns weather snapshot when provider succeeds', async () => {
    const sampleCurrent = {
      temperature_2m: 21,
      relative_humidity_2m: 40,
      wind_speed_10m: 5,
      wind_gusts_10m: 12,
      precipitation: 0,
    };

    const buildSeries = (offset = 0) =>
      Array.from({ length: 30 }, (_, idx) => idx + offset);

    const hourlyData = {
      time: Array.from(
        { length: 30 },
        (_, idx) => `2024-01-01T${String(idx).padStart(2, '0')}:00:00Z`
      ),
      temperature_2m: buildSeries(10),
      relative_humidity_2m: buildSeries(50),
      wind_speed_10m: buildSeries(3),
      wind_gusts_10m: buildSeries(6),
      precipitation: buildSeries(0),
    };

    axios.get.mockResolvedValue({
      data: {
        current: sampleCurrent,
        hourly: hourlyData,
      },
    });

    Weather.create.mockImplementation(async (doc) => ({
      _id: 'weather-doc',
      ...doc,
    }));

    const res = await request(app).get(
      '/api/weather/current?lat=34&lon=-118'
    );

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('ok', true);

    const expectedHourly = {
      time: hourlyData.time.slice(0, 24),
      temperature_2m: hourlyData.temperature_2m.slice(0, 24),
      relative_humidity_2m: hourlyData.relative_humidity_2m.slice(0, 24),
      wind_speed_10m: hourlyData.wind_speed_10m.slice(0, 24),
      wind_gusts_10m: hourlyData.wind_gusts_10m.slice(0, 24),
      precipitation: hourlyData.precipitation.slice(0, 24),
    };

    expect(axios.get).toHaveBeenCalledWith(
      'https://api.open-meteo.com/v1/forecast',
      {
        params: expect.objectContaining({
          latitude: 34,
          longitude: -118,
          wind_speed_unit: 'ms',
        }),
      }
    );

    // ✅ Now this will pass again
    expect(Weather.create).toHaveBeenCalledWith(
      expect.objectContaining({
        latitude: 34,
        longitude: -118,
        fetchedAt: expect.any(Date),
        current: sampleCurrent,
        hourly: expectedHourly,
      })
    );

    expect(res.body.data).toMatchObject({
      _id: 'weather-doc',
      latitude: 34,
      longitude: -118,
      current: sampleCurrent,
      hourly: expectedHourly,
    });

    expect(new Date(res.body.data.fetchedAt).toString()).not.toBe(
      'Invalid Date'
    );
  });

  test('400 when lat or lon missing', async () => {
    const res = await request(app).get('/api/weather/current?lat=34');

    expect(res.status).toBe(400);
    expect(axios.get).not.toHaveBeenCalled();
    expect(Weather.create).not.toHaveBeenCalled();
  });
});
