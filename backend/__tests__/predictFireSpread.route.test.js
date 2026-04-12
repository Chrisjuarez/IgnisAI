jest.mock('axios', () => ({ get: jest.fn() }));

const request = require('supertest');
const axios = require('axios');
const app = require('../app');

describe('GET /api/predict-fire-spread routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
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
        threshold: 0.01,
        prob_min: 0.0,
        prob_mean: 0.04,
        prob_max: 0.21,
        area_fraction: 0.13,
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
      threshold: 0.01,
    });
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/predict_raster_json'),
      expect.objectContaining({
        params: expect.objectContaining({
          lat: '34.05',
          lon: '-118.25',
          Tseq: 1,
          crop_frac: 0.5,
        }),
        timeout: 80000,
      })
    );
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
        threshold: 0.01,
        step_hours: 6,
        steps: [
          {
            index: 0,
            lead_hours: 6,
            label: '6 hours',
            image_base64: 'frame-1',
            prob_min: 0.0,
            prob_mean: 0.03,
            prob_max: 0.14,
            area_fraction: 0.08,
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
      threshold: 0.01,
      step_hours: 6,
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
          crop_frac: '0.4',
          date: '2021-08-14',
          ignition: true,
        }),
        timeout: 80000,
      })
    );
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
});
