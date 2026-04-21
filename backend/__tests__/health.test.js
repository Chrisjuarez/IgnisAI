// backend/__tests__/health.test.js
const request = require('supertest');
const app = require('../app');

describe('GET /health', () => {
  it('returns health status with system metrics', async () => {
    const res = await request(app).get('/health');

    // Render uses /health as a liveness check; dependency failures are reported
    // in the JSON body without failing deployment.
    expect(res.statusCode).toBe(200);
    
    // Verify response structure
    expect(res.body).toHaveProperty('status');
    expect(res.body).toHaveProperty('uptime');
    expect(res.body).toHaveProperty('timestamp');
    expect(res.body).toHaveProperty('checks');
    
    // Verify checks object
    expect(res.body.checks).toHaveProperty('database');
    expect(res.body.checks).toHaveProperty('tilesvc');
    expect(res.body.checks).toHaveProperty('predictions');
    expect(res.body.checks).toHaveProperty('memory');
    expect(res.body.checks).toHaveProperty('cpu');
    
    // Status should be OK or DEGRADED
    expect(['OK', 'DEGRADED']).toContain(res.body.status);
    
    // If DB disconnected, status should be DEGRADED
    if (res.body.checks.database === 'disconnected') {
      expect(res.body.status).toBe('DEGRADED');
    }
    
    // If DB and tilesvc are connected, status should be OK unless one of the
    // production model checks reports degraded.
    if (
      res.body.checks.database === 'connected' &&
      res.body.checks.tilesvc?.status === 'connected' &&
      res.body.checks.tilesvc?.staticCatalog?.ok !== false &&
      res.body.checks.tilesvc?.calibration?.ok !== false
    ) {
      expect(res.body.status).toBe('OK');
      expect(res.statusCode).toBe(200);
    }
  });

  it('can return non-200 for strict readiness checks', async () => {
    const res = await request(app).get('/health?strict=1');

    expect([200, 503]).toContain(res.statusCode);
    if (res.body.status === 'DEGRADED') {
      expect(res.statusCode).toBe(503);
    }
  });
});
