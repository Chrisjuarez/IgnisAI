const { respondNotFound, respondWithError } = require('../middleware/errorResponses');

function fakeResponse() {
  const res = {};
  res.status = jest.fn(() => res);
  res.json = jest.fn(() => res);
  return res;
}

describe('respondNotFound', () => {
  it('answers 404 in JSON', () => {
    const res = fakeResponse();
    respondNotFound({}, res);
    expect(res.status).toHaveBeenCalledWith(404);
    expect(res.json).toHaveBeenCalledWith({ error: 'not_found' });
  });
});

describe('respondWithError', () => {
  let errorLog;
  beforeEach(() => { errorLog = jest.spyOn(console, 'error').mockImplementation(() => {}); });
  afterEach(() => errorLog.mockRestore());

  it('hides an unexpected error behind a generic 500 and logs it', () => {
    const res = fakeResponse();
    const err = new Error('Mongo password rejected');

    respondWithError(err, { method: 'GET', originalUrl: '/api/incidents/1' }, res);

    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalledWith({ error: 'internal_error' });
    expect(errorLog).toHaveBeenCalledWith('Request failed:', 'GET', '/api/incidents/1', err);
  });

  it('cannot be made to forge log lines through the request URL', () => {
    respondWithError(new Error('boom'), { method: 'GET', originalUrl: '/api/x\nFAKE LOG LINE' }, fakeResponse());

    const [format, , url] = errorLog.mock.calls[0];
    expect(format).toBe('Request failed:');
    expect(url).not.toMatch(/[\r\n]/);
  });

  it('passes a 4xx message through only when the error marks it safe to show', () => {
    const shown = fakeResponse();
    const hidden = fakeResponse();

    respondWithError(Object.assign(new Error('Origin not allowed'), { status: 403, expose: true }), {}, shown);
    respondWithError(Object.assign(new Error('internal detail'), { status: 400 }), {}, hidden);

    expect(shown.json).toHaveBeenCalledWith({ error: 'request_rejected', message: 'Origin not allowed' });
    expect(hidden.json).toHaveBeenCalledWith({ error: 'request_rejected', message: undefined });
    expect(errorLog).not.toHaveBeenCalled();
  });
});
