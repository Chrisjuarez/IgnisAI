const request = require('supertest');

function loadAppWithCors(corsOrigin) {
  jest.resetModules();
  if (corsOrigin == null) {
    delete process.env.CORS_ORIGIN;
  } else {
    process.env.CORS_ORIGIN = corsOrigin;
  }
  return require('../app');
}

describe('CORS configuration', () => {
  const originalCorsOrigin = process.env.CORS_ORIGIN;

  afterEach(() => {
    jest.resetModules();
    if (originalCorsOrigin == null) {
      delete process.env.CORS_ORIGIN;
    } else {
      process.env.CORS_ORIGIN = originalCorsOrigin;
    }
  });

  it('allows the Render frontend origin by default', async () => {
    const app = loadAppWithCors(null);

    const response = await request(app)
      .options('/api/auth/login')
      .set('Origin', 'https://ignisai-frontend.onrender.com')
      .set('Access-Control-Request-Method', 'POST')
      .expect(204);

    expect(response.headers['access-control-allow-origin']).toBe('https://ignisai-frontend.onrender.com');
  });

  it('supports comma-separated CORS_ORIGIN values', async () => {
    const app = loadAppWithCors('https://one.example, https://two.example');

    const response = await request(app)
      .options('/api/auth/login')
      .set('Origin', 'https://two.example')
      .set('Access-Control-Request-Method', 'POST')
      .expect(204);

    expect(response.headers['access-control-allow-origin']).toBe('https://two.example');
  });

  it('refuses an unlisted origin with a JSON 403 and logs which origin it was', async () => {
    const app = loadAppWithCors(null);
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const response = await request(app)
      .get('/api/map/bootstrap')
      .set('Origin', 'https://elsewhere.example')
      .expect(403);

    expect(response.body).toEqual({ error: 'request_rejected', message: 'Origin not allowed' });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('CORS_ORIGIN'), 'https://elsewhere.example');
    warn.mockRestore();
  });
});

describe('unmatched routes', () => {
  it('answer with a JSON 404', async () => {
    jest.resetModules();
    const app = require('../app');

    const response = await request(app).get('/api/no-such-route').expect(404);

    expect(response.body).toEqual({ error: 'not_found' });
  });
});
