// backend/__tests__/health.test.js
const request = require('supertest');
const app = require('../app');

describe('GET /health', () => {
  it('returns health status with system metrics', async () => {
    const res = await request(app).get('/health');
    
    // Accept either 200 (DB connected) or 503 (DB disconnected)
    expect([200, 503]).toContain(res.statusCode);
    
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
      expect(res.statusCode).toBe(503);
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
});
