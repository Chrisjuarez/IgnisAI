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
});
