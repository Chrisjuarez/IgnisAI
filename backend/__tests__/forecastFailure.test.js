const { forecastFailure } = require('../utils/forecastFailure');

const upstream = (status, data) => Object.assign(new Error(`Request failed with status code ${status}`), { response: { status, data } });

describe('forecastFailure', () => {
  it.each(['ECONNABORTED', 'ETIMEDOUT'])('reads %s as a timeout', (code) => {
    const { status, body } = forecastFailure(Object.assign(new Error('timeout'), { code }));
    expect(status).toBe(504);
    expect(body).toMatchObject({ error: 'forecast_timeout', message: expect.stringMatching(/too long/) });
  });

  it.each([
    ['static_channel_nodata_heavy', /Terrain and fuel data/],
    ['firms_snapshot_missing', /Satellite fire detections/],
    ['calibration_missing', /model is not available/],
    ['predictions_disabled', /turned off/],
    ['something_new', /could not gather the inputs/],
  ])('explains tilesvc reason %s', (reason, message) => {
    const { status, body } = forecastFailure(upstream(422, { error: 'input_unavailable', reason, detail: 'internal detail' }));
    expect(status).toBe(422);
    expect(body).toEqual({ error: 'forecast_inputs_unavailable', reason, message: expect.stringMatching(message) });
  });

  it('reads any other 4xx as a request tilesvc rejected', () => {
    const { status, body } = forecastFailure(upstream(422, { detail: [{ loc: ['query', 'lat'] }] }));
    expect(status).toBe(400);
    expect(body.error).toBe('forecast_request_invalid');
  });

  it.each([
    ['a tilesvc 5xx', upstream(500, 'Internal Server Error')],
    ['a refused connection', Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' })],
    ['a malformed forecast', new Error('tilesvc multistep missing bounds/steps')],
  ])('reads %s as the service being unavailable', (_label, err) => {
    const { status, body } = forecastFailure(err);
    expect(status).toBe(502);
    expect(body.error).toBe('forecast_service_unavailable');
    expect(body.message).not.toMatch(/ECONNREFUSED|bounds/);
  });
});
