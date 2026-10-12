jest.mock('axios', () => ({ get: jest.fn(), post: jest.fn() }));

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

  it('returns observed-only unavailable contract when predictions are disabled', async () => {
    process.env.PREDICTIONS_ENABLED = 'false';

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(503);

    expect(res.body).toMatchObject({
      error: 'predictions_disabled',
      quality: {
        status: 'unavailable',
        reasons: ['predictions_disabled'],
      },
    });
    expect(axios.post).not.toHaveBeenCalled();
  });

  it('forwards multistep params and sets ignition for dated requests', async () => {
    axios.post.mockResolvedValue({
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
    expect(axios.post).toHaveBeenCalledWith(
      expect.stringContaining('/predict_multistep'),
      { burned_area: null },
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

  it('forwards every field tilesvc returns on a multistep forecast', async () => {
    // The route used to rebuild this response from a whitelist, which dropped
    // `scene` - the arrival bands - on every request. The band view, its labels
    // and the hybrid layer shipped in tilesvc and never reached the browser.
    // Asserting on a field no list could have anticipated is the point: the
    // route must forward what it receives, not what it was told to expect.
    const forecast = {
      bounds: [-118.6, 34.0, -118.1, 34.4],
      steps: [{ index: 0, lead_hours: 24, label: '1 day', image_base64: 'frame-1' }],
      scene: {
        ignition: { type: 'Feature', geometry: { type: 'Point', coordinates: [-118.4, 34.56] }, properties: {} },
        observed: { type: 'FeatureCollection', features: [] },
        forecast: {
          type: 'FeatureCollection',
          features: [{ type: 'Feature', properties: { day: 1, color: '#bd0026' }, geometry: null }],
        },
      },
      hybrid: { available: true, engines: ['learned', 'rothermel'], units: 'percentile_rank' },
      calibration: { ok: true, method: 'isotonic' },
      model: 'learned',
      field_added_next_quarter: { anything: true },
    };
    axios.post.mockResolvedValue({ data: forecast });

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.56, lon: -118.4 })
      .expect(200);

    expect(res.body).toEqual(forecast);
  });

  it('still refuses a forecast without bounds or steps', async () => {
    axios.post.mockResolvedValue({ data: { scene: {}, steps: [] } });

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.56, lon: -118.4 })
      .expect(502);

    expect(res.body.error).toBe('forecast_service_unavailable');
  });

  it('always sends ignition=true on live multistep requests (no date)', async () => {
    axios.post.mockResolvedValue({
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

    expect(axios.post).toHaveBeenCalledWith(
      expect.stringContaining('/predict_multistep'),
      { burned_area: null },
      expect.objectContaining({
        params: expect.objectContaining({
          lat: '34.05',
          lon: '-118.25',
          ignition: true,
        }),
      })
    );
    // Dateless request should not forward a date param.
    expect(axios.post.mock.calls[0][2].params).not.toHaveProperty('date');
    expect(axios.post.mock.calls[0][2].params).not.toHaveProperty('thr');
  });

  it('honors explicit ignition=false opt-out for multistep requests', async () => {
    axios.post.mockResolvedValue({
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

    expect(axios.post.mock.calls[0][2].params).toMatchObject({ ignition: false });
  });

  it('propagates multistep tilesvc failures as structured 5xx json', async () => {
    axios.post.mockRejectedValue(new Error('tilesvc down'));

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({
      error: 'forecast_service_unavailable',
      message: expect.any(String),
    });
  });

  it('says why tilesvc could not gather a forecast\'s inputs', async () => {
    const failure = Object.assign(new Error('Request failed with status code 422'), {
      response: { status: 422, data: { error: 'input_unavailable', reason: 'static_channel_nodata_heavy', detail: 'fuel_model mostly nodata' } },
    });
    axios.post.mockRejectedValue(failure);

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(422);

    expect(res.body).toEqual({
      error: 'forecast_inputs_unavailable',
      reason: 'static_channel_nodata_heavy',
      message: 'Terrain and fuel data for this area could not be loaded.',
    });
  });

  it('answers 504 when the forecast times out', async () => {
    axios.post.mockRejectedValue(Object.assign(new Error('timeout of 240000ms exceeded'), { code: 'ECONNABORTED' }));

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(504);

    expect(res.body.error).toBe('forecast_timeout');
  });

  it('gives site exposure the same failure messages', async () => {
    axios.get.mockRejectedValue(Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }));

    const res = await request(app)
      .get('/api/predict-fire-spread/site-exposure')
      .query({ site_lat: 34.05, site_lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({ error: 'forecast_service_unavailable', message: expect.any(String) });
  });

  describe('incident burned area', () => {
    const mapData = require('../routes/mapData');
    const perimeter = {
      type: 'FeatureCollection',
      features: [{ type: 'Feature', properties: {}, geometry: {
        type: 'Polygon', coordinates: [[[-118.45, 34.55], [-118.4, 34.55], [-118.4, 34.58], [-118.45, 34.55]]],
      } }],
    };
    const forecast = { bounds: [-118.6, 34.0, -118.1, 34.4], steps: [], burned_area: { applied: true } };

    afterEach(() => jest.restoreAllMocks());

    it('sends the incident\'s mapped perimeter to tilesvc as its burned area', async () => {
      const lookup = jest.spyOn(mapData, 'burnedAreaForIncident').mockResolvedValue(perimeter);
      axios.post.mockResolvedValue({ data: forecast });

      await request(app)
        .get('/api/predict-fire-spread/multistep')
        .query({ lat: 34.56, lon: -118.4, incident_id: 'IRWIN-1' })
        .expect(200);

      expect(lookup).toHaveBeenCalledWith('IRWIN-1');
      expect(axios.post).toHaveBeenCalledWith(
        expect.stringContaining('/predict_multistep'),
        { burned_area: perimeter },
        expect.objectContaining({ params: expect.not.objectContaining({ incident_id: expect.anything() }) }),
      );
    });

    it('still forecasts when the perimeter lookup fails', async () => {
      jest.spyOn(mapData, 'burnedAreaForIncident').mockRejectedValue(new Error('ArcGIS down'));
      jest.spyOn(console, 'warn').mockImplementation(() => {});
      axios.post.mockResolvedValue({ data: forecast });

      await request(app)
        .get('/api/predict-fire-spread/multistep')
        .query({ lat: 34.56, lon: -118.4, incident_id: 'IRWIN-1' })
        .expect(200);

      expect(axios.post.mock.calls[0][1]).toEqual({ burned_area: null });
    });

    it('cannot forge log lines through the incident id', async () => {
      jest.spyOn(mapData, 'burnedAreaForIncident').mockRejectedValue(new Error('ArcGIS down'));
      const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
      axios.post.mockResolvedValue({ data: forecast });

      await request(app)
        .get('/api/predict-fire-spread/multistep')
        .query({ lat: 34.56, lon: -118.4, incident_id: 'IRWIN-1\r\n[admin] login ok' })
        .expect(200);

      const [line] = warn.mock.calls.find(([message]) => message.startsWith('burned area lookup failed'));
      expect(line).not.toMatch(/[\r\n]/);
      expect(line).toContain('IRWIN-1[admin] login ok');
    });

    it('does not look anything up for a forecast that is not tied to an incident', async () => {
      const lookup = jest.spyOn(mapData, 'burnedAreaForIncident');
      axios.post.mockResolvedValue({ data: forecast });

      await request(app)
        .get('/api/predict-fire-spread/multistep')
        .query({ lat: 34.56, lon: -118.4 })
        .expect(200);

      expect(lookup).not.toHaveBeenCalled();
    });
  });

  // ── Error paths ─────────────────────────────────────────────────────────────

  it('returns 400 when lat/lon are missing from multistep request', async () => {
    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({}) // missing both
      .expect(400);

    expect(res.body).toMatchObject({ error: expect.any(String) });
    expect(axios.get).not.toHaveBeenCalled();
  });

  it('returns 502 when multistep response is missing steps array', async () => {
    axios.post.mockResolvedValue({
      data: { bounds: [-118.6, 34.0, -118.1, 34.4] }, // no steps array
    });

    const res = await request(app)
      .get('/api/predict-fire-spread/multistep')
      .query({ lat: 34.05, lon: -118.25 })
      .expect(502);

    expect(res.body).toMatchObject({ error: 'forecast_service_unavailable' });
  });
});
