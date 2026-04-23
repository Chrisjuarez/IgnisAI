jest.mock('axios', () => ({ get: jest.fn() }));

const request = require('supertest');
const axios = require('axios');
const app = require('../app');

describe('GET /api/predict-fire-spread routes', () => {
  const originalPredictionsEnabled = process.env.PREDICTIONS_ENABLED;
  const restorePredictionsFlag = () => {
    if (originalPredictionsEnabled === undefined) {
      delete process.env.PREDICTIONS_ENABLED;
    } else {
      process.env.PREDICTIONS_ENABLED = originalPredictionsEnabled;
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
    restorePredictionsFlag();
  });

  afterAll(() => {
    restorePredictionsFlag();
  });

  it('uses crop_frac=0.5 by default for raster requests', async () => {
    axios.get.mockResolvedValue({
      data: {
        bounds: [-118.6, 34.0, -118.1, 34.4],
        coordinates: [
          [-118.6, 34.4],
          [-118.1, 34.4],
          [-118.1, 34.0],
          [-118.6, 34.0],
        ],
        image_base64: 'abc123',
        threshold: 0.85,
        display_floor: 0.02,
        prob_min: 0.0,
        prob_mean: 0.04,
        prob_max: 0.21,
        area_fraction: 0.00,
        display_area_fraction: 0.13,
        probability_scale: { mode: 'absolute', min: 0, max: 1, display_floor: 0.02 },
        quality: { status: 'degraded', reasons: ['open_meteo_weather_fallback'] },
        data_sources: { weather: 'Open-Meteo fallback' },
        p_new_burn: { prob_max: 0.21, area_fraction: 0.0 },
        p_next_fire: { area_fraction: 0.02 },
        observed_fire: { area_fraction: 0.01 },
        display_score: { max: 0.3 },
        risk_class: { fractions: { extreme: 0.01 } },
        layer_images: { new_burn: 'abc123', next_fire: 'def456', observed_fire: 'ghi789' },
        model_meta: {
          Tseq: 6,
          step_hours: 24,
          dynamic_order: ['fire_t', 'u', 'v', 'gust', 'tempC', 'q', 'precip'],
        },
      }
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(200);

    expect(res.body).toMatchObject({
      bounds: [-118.6, 34.0, -118.1, 34.4],
      coordinates: [
        [-118.6, 34.4],
        [-118.1, 34.4],
        [-118.1, 34.0],
        [-118.6, 34.0],
      ],
      image_base64: 'abc123',
      threshold: 0.85,
      display_floor: 0.02,
      display_area_fraction: 0.13,
      probability_scale: { mode: 'absolute', min: 0, max: 1, display_floor: 0.02 },
      quality: { status: 'degraded', reasons: ['open_meteo_weather_fallback'] },
      data_sources: { weather: 'Open-Meteo fallback' },
      p_new_burn: { prob_max: 0.21, area_fraction: 0.0 },
      p_next_fire: { area_fraction: 0.02 },
      observed_fire: { area_fraction: 0.01 },
      display_score: { max: 0.3 },
      risk_class: { fractions: { extreme: 0.01 } },
      layer_images: { new_burn: 'abc123', next_fire: 'def456', observed_fire: 'ghi789' },
      model_meta: {
        Tseq: 6,
        step_hours: 24,
        dynamic_order: ['fire_t', 'u', 'v', 'gust', 'tempC', 'q', 'precip'],
      },
    });
    const params = axios.get.mock.calls[0][1].params;
    expect(axios.get.mock.calls[0][0]).toContain('/predict_raster_json');
    expect(params).toMatchObject({
      lat: '34.05',
      lon: '-118.25',
      crop_frac: 0.5,
    });
    expect(params).not.toHaveProperty('Tseq');
  });

  it('returns observed-only unavailable contract when predictions are disabled', async () => {
    process.env.PREDICTIONS_ENABLED = 'false';

    const res = await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(503);

    expect(res.body).toMatchObject({
      error: 'predictions_disabled',
      quality: {
        status: 'unavailable',
        reasons: ['predictions_disabled'],
      },
    });
    expect(axios.get).not.toHaveBeenCalled();
  });

  it('forwards multistep params and sets ignition for dated requests', async () => {
    axios.get.mockResolvedValue({
      data: {
        bounds: [-118.6, 34.0, -118.1, 34.4],
        coordinates: [
          [-118.6, 34.4],
          [-118.1, 34.4],
          [-118.1, 34.0],
          [-118.6, 34.0],
        ],
        threshold: 0.85,
        display_floor: 0.02,
        step_hours: 6,
        probability_scale: { mode: 'absolute', min: 0, max: 1, display_floor: 0.02 },
        model_meta: { Tseq: 2, step_hours: 6 },
        steps: [
          {
            index: 0,
            lead_hours: 6,
            label: '6 hours',
            image_base64: 'frame-1',
            prob_min: 0.0,
            prob_mean: 0.03,
            prob_max: 0.14,
            area_fraction: 0.00,
            display_area_fraction: 0.08,
            display_floor: 0.02,
          }
        ],
      }
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({
        lat: 34.05,
        lon: -118.25,
        steps: 6,
        step_hours: 6,
        Tseq: 2,
        thr: 0.05,
        display_floor: 0.02,
        crop_frac: 0.4,
        date: '2021-08-14',
      })
      .expect(200);

    expect(res.body).toMatchObject({
      bounds: [-118.6, 34.0, -118.1, 34.4],
      coordinates: [
        [-118.6, 34.4],
        [-118.1, 34.4],
        [-118.1, 34.0],
        [-118.6, 34.0],
      ],
      threshold: 0.85,
      display_floor: 0.02,
      step_hours: 6,
      probability_scale: { mode: 'absolute', min: 0, max: 1, display_floor: 0.02 },
      model_meta: { Tseq: 2, step_hours: 6 },
      steps: [
        expect.objectContaining({
          index: 0,
          lead_hours: 6,
          label: '6 hours',
        })
      ]
    });
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/predict_multistep'),
      expect.objectContaining({
        params: expect.objectContaining({
          lat: '34.05',
          lon: '-118.25',
          steps: '6',
          step_hours: '6',
          Tseq: '2',
          thr: '0.05',
          display_floor: '0.02',
          crop_frac: '0.4',
          date: '2021-08-14',
          ignition: true,
        }),
        // Multistep uses a longer ceiling (default 240s) because tilesvc
        // cold-start + 6-step rollout exceeds the 80s used by other routes.
        timeout: 240000,
      })
    );
  });

  it('always sends ignition=true on live multistep requests (no date)', async () => {
    axios.get.mockResolvedValue({
      data: {
        bounds: [-118.6, 34.0, -118.1, 34.4],
        threshold: 0.85,
        step_hours: 6,
        steps: [
          { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1' },
        ],
      }
    });

    await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(200);

    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/predict_multistep'),
      expect.objectContaining({
        params: expect.objectContaining({
          lat: '34.05',
          lon: '-118.25',
          ignition: true,
        }),
      })
    );
    // Dateless request should not forward a date param.
    expect(axios.get.mock.calls[0][1].params).not.toHaveProperty('date');
    expect(axios.get.mock.calls[0][1].params).not.toHaveProperty('thr');
  });

  it('honors explicit ignition=false opt-out for multistep requests', async () => {
    axios.get.mockResolvedValue({
      data: {
        bounds: [-118.6, 34.0, -118.1, 34.4],
        threshold: 0.85,
        step_hours: 6,
        steps: [
          { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1' },
        ],
      }
    });

    await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25, ignition: 'false' })
      .expect(200);

    expect(axios.get.mock.calls[0][1].params).toMatchObject({ ignition: false });
  });

  it('sends ignition=true on live raster requests (no date)', async () => {
    axios.get.mockResolvedValue({
      data: {
        bounds: [-118.6, 34.0, -118.1, 34.4],
        coordinates: [
          [-118.6, 34.4],
          [-118.1, 34.4],
          [-118.1, 34.0],
          [-118.6, 34.0],
        ],
        image_base64: 'abc123',
        threshold: 0.85,
      },
    });

    await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(200);

    expect(axios.get.mock.calls[0][1].params).toMatchObject({ ignition: true });
    expect(axios.get.mock.calls[0][1].params).not.toHaveProperty('date');
  });

  it('propagates multistep tilesvc failures as structured 5xx json', async () => {
    axios.get.mockRejectedValue(new Error('tilesvc down'));

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({
      error: 'tilesvc_multistep_failed',
      detail: expect.any(String),
    });
  });

  it('proxies input audit requests to tilesvc with model metadata', async () => {
    axios.get.mockResolvedValue({
      data: {
        ok: true,
        model_meta: {
          Tseq: 6,
          step_hours: 24,
          dynamic_order: ['fire_t', 'u', 'v', 'gust', 'tempC', 'q', 'precip'],
        },
        input_summary: {
          dyn_shape: [6, 7, 128, 128],
          stat_shape: [15, 128, 128],
          missing_or_placeholder_static: ['NDVI'],
        },
      },
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/input-audit')
      .query({ lat: 34.05, lon: -118.25, date: '2025-01-07T18:30:00Z', step_hours: 24 })
      .expect(200);

    expect(res.body).toMatchObject({
      ok: true,
      model_meta: { Tseq: 6, step_hours: 24 },
      input_summary: { dyn_shape: [6, 7, 128, 128] },
    });
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/input_audit'),
      expect.objectContaining({
        params: expect.objectContaining({
          lat: '34.05',
          lon: '-118.25',
          date: '2025-01-07T18:30:00Z',
          step_hours: '24',
          ignition: true,
        }),
      })
    );
  });

  // ── /vector ────────────────────────────────────────────────────────────────

  it('returns vector prediction with geojson + env data', async () => {
    // /vector makes 3 sequential tilesvc calls: predict_geojson, predict (meta), then weather
    axios.get
      .mockResolvedValueOnce({
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: { threshold: 0.01 },
              geometry: { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] },
            },
          ],
        },
      })
      .mockResolvedValueOnce({
        data: {
          bounds: [-118.6, 34.0, -118.1, 34.4],
          area_fraction: 0.15,
          threshold: 0.01,
          prob_min: 0.0,
          prob_mean: 0.05,
          prob_max: 0.22,
        },
      })
      // weather call — allow it to fail gracefully (the route swallows weather errors)
      .mockRejectedValueOnce(new Error('weather unavailable'));

    const res = await request(app)
      .get('/api/predict-fire-spread/vector')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(200);

    expect(res.body).toMatchObject({
      geojson: { type: 'FeatureCollection' },
      spread_probability: expect.any(Number),
      spread_distance_km: expect.any(Number),
      threshold: 0.01,
      bounds: [-118.6, 34.0, -118.1, 34.4],
      prob_min: 0.0,
      prob_max: 0.22,
      environmental_data: expect.objectContaining({ data_source: 'unknown' }),
    });
    expect(res.body.geojson.features).toHaveLength(1);
  });

  it('enriches vector features with wind direction and spread probability', async () => {
    const mockFeature = {
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] },
    };

    axios.get
      .mockResolvedValueOnce({
        data: { type: 'FeatureCollection', features: [mockFeature] },
      })
      .mockResolvedValueOnce({
        data: {
          bounds: [-118.6, 34.0, -118.1, 34.4],
          area_fraction: 0.10,
          threshold: 0.01,
        },
      })
      .mockResolvedValueOnce({
        data: {
          data: {
            current: {
              wind_speed_10m: 5.5,
              wind_direction_10m: 270,
              temperature_2m: 35,
              relative_humidity_2m: 20,
            },
          },
        },
      });

    const res = await request(app)
      .get('/api/predict-fire-spread/vector')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(200);

    const feature = res.body.geojson.features[0];
    expect(feature.properties.direction).toBe(270);
    expect(feature.properties.spread_probability).toBeCloseTo(0.10);
    expect(res.body.environmental_data).toMatchObject({
      wind_direction: 270,
      temperature: 35,
      humidity: 20,
      data_source: 'weather_api',
    });
  });

  it('returns 400 when lat/lon are missing from vector request', async () => {
    const res = await request(app)
      .get('/api/predict-fire-spread/vector')
      .query({ lat: 34.05 }) // missing lon
      .expect(400);

    expect(res.body).toMatchObject({ error: expect.any(String) });
    expect(axios.get).not.toHaveBeenCalled();
  });

  it('propagates vector tilesvc failures as structured 5xx json', async () => {
    axios.get.mockRejectedValue(new Error('tilesvc unreachable'));

    const res = await request(app)
      .get('/api/predict-fire-spread/vector')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({
      error: 'tilesvc_vector_failed',
      detail: expect.any(String),
    });
  });

  // ── Error paths ─────────────────────────────────────────────────────────────

  it('returns 400 when lat/lon are missing from raster request', async () => {
    const res = await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lon: -118.25 }) // missing lat
      .expect(400);

    expect(res.body).toMatchObject({ error: expect.any(String) });
    expect(axios.get).not.toHaveBeenCalled();
  });

  it('returns 400 when lat/lon are missing from multistep request', async () => {
    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({}) // missing both
      .expect(400);

    expect(res.body).toMatchObject({ error: expect.any(String) });
    expect(axios.get).not.toHaveBeenCalled();
  });

  it('propagates raster tilesvc failures as structured 5xx json', async () => {
    axios.get.mockRejectedValue(new Error('connection refused'));

    const res = await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({
      error: 'tilesvc_raster_failed',
      detail: expect.any(String),
    });
  });

  it('returns 502 when raster response is missing required fields', async () => {
    // tilesvc returns something unexpected (e.g. missing image_base64)
    axios.get.mockResolvedValue({
      data: { bounds: [-118.6, 34.0, -118.1, 34.4] }, // no image_base64
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/raster')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({ error: 'tilesvc_raster_failed' });
  });

  it('returns 502 when multistep response is missing steps array', async () => {
    axios.get.mockResolvedValue({
      data: { bounds: [-118.6, 34.0, -118.1, 34.4] }, // no steps array
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({ error: 'tilesvc_multistep_failed' });
  });
});
