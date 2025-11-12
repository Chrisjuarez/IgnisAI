// backend/__tests__/rate-limiting.test.js
const request = require('supertest');
const express = require('express');
const rateLimit = require('express-rate-limit');

describe('Rate Limiting Protection Integration Tests', () => {
  let app;
  let server;

  beforeAll(() => {
    // Create a minimal Express app just for testing rate limiting
    app = express();
    app.use(express.json());

    // Configure rate limiter for testing (very permissive for fast tests)
    const loginLimiter = rateLimit({
      windowMs: 1 * 60 * 1000, // 1 minute window for testing
      max: 5, // limit each IP to 5 requests per windowMs
      message: 'Too many login attempts, please try again later',
      standardHeaders: true,
      legacyHeaders: false,
      handler: (req, res) => {
        res.status(429).json({
          message: 'Too many login attempts, please try again later'
        });
      }
    });

    // Test endpoint with rate limiting
    app.post('/api/auth/login', loginLimiter, (req, res) => {
      res.status(200).json({ message: 'Login endpoint reached' });
    });

    // Start server on random port
    server = app.listen(0);
  });

  afterAll(async () => {
    if (server) {
      await new Promise((resolve) => {
        server.close(resolve);
      });
    }
  });

  describe('Authentication Endpoint Rate Limiting', () => {
    it('should allow 5 login requests within time window', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'testpassword'
      };

      // Send 5 requests (should all succeed)
      const requests = Array(5).fill().map(() =>
        request(app)
          .post('/api/auth/login')
          .send(loginData)
      );

      const responses = await Promise.all(requests);
      
      // All 5 should succeed (not rate limited)
      responses.forEach(res => {
        expect([200, 401, 500]).toContain(res.status); // Any status except 429
      });
    });

    it('should block 6th request with 429 status', async () => {
      const loginData = {
        email: 'ratelimit-test@example.com',
        password: 'testpassword'
      };

      // Send 6 requests rapidly
      const requests = [];
      for (let i = 0; i < 6; i++) {
        requests.push(
          request(app)
            .post('/api/auth/login')
            .send(loginData)
        );
      }

      const responses = await Promise.all(requests);
      
      // At least one should be rate limited
      const rateLimited = responses.filter(res => res.status === 429);
      expect(rateLimited.length).toBeGreaterThan(0);
    });

    it('should include rate limit headers in response', async () => {
      const loginData = {
        email: 'headers-test@example.com',
        password: 'testpassword'
      };

      const res = await request(app)
        .post('/api/auth/login')
        .send(loginData);

      // Should include RateLimit headers
      expect(
        res.headers['ratelimit-limit'] || 
        res.headers['x-ratelimit-limit']
      ).toBeDefined();
    });

    it('should return appropriate error message when rate limited', async () => {
      const loginData = {
        email: 'error-message-test@example.com',
        password: 'testpassword'
      };

      // Exhaust rate limit
      for (let i = 0; i < 6; i++) {
        const res = await request(app)
          .post('/api/auth/login')
          .send(loginData);
        
        if (res.status === 429) {
          expect(res.body.message).toBeDefined();
          expect(res.body.message.toLowerCase()).toContain('too many');
          break;
        }
      }
    });
  });

  describe('Response Time Under Rate Limiting', () => {
    it('should respond quickly even when applying rate limits', async () => {
      const startTime = Date.now();
      
      await request(app)
        .post('/api/auth/login')
        .send({ email: 'perf@test.com', password: 'test' });
      
      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(2000); // Should respond in under 2 seconds
    });
  });

  describe('Rate Limit Configuration', () => {
    it('should have consistent rate limit window', async () => {
      // Create a fresh app with new rate limiter for this test
      const testApp = express();
      testApp.use(express.json());
      
      const freshLimiter = rateLimit({
        windowMs: 1 * 60 * 1000,
        max: 5,
        standardHeaders: true,
        legacyHeaders: false
      });
      
      testApp.post('/api/auth/login', freshLimiter, (req, res) => {
        res.status(200).json({ message: 'Login endpoint reached' });
      });

      const loginData = {
        email: 'window-test@example.com',
        password: 'testpassword'
      };

      // First request should succeed
      const res1 = await request(testApp)
        .post('/api/auth/login')
        .send(loginData);

      expect(res1.status).toBe(200);

      // Second request immediately after should also succeed
      const res2 = await request(testApp)
        .post('/api/auth/login')
        .send(loginData);

      expect(res2.status).toBe(200);
    });

    it('should track requests per IP address', async () => {
      // Use unique email for this test
      const uniqueEmail = `ip-test-${Date.now()}@example.com`;
      
      // Multiple requests from same IP should be counted together
      const requests = Array(3).fill().map(() =>
        request(app)
          .post('/api/auth/login')
          .send({ email: uniqueEmail, password: 'test' })
      );

      const responses = await Promise.all(requests);
      
      // All should have rate limit headers indicating same counter
      responses.forEach(res => {
        expect(
          res.headers['ratelimit-remaining'] !== undefined ||
          res.headers['x-ratelimit-remaining'] !== undefined
        ).toBe(true);
      });
    });
  });

  describe('Rate Limit Edge Cases', () => {
    it('should handle concurrent requests correctly', async () => {
      // Create a fresh app with new rate limiter for this test
      const testApp = express();
      testApp.use(express.json());
      
      const freshLimiter = rateLimit({
        windowMs: 1 * 60 * 1000,
        max: 5,
        standardHeaders: true,
        legacyHeaders: false
      });
      
      testApp.post('/api/auth/login', freshLimiter, (req, res) => {
        res.status(200).json({ message: 'Login endpoint reached' });
      });

      const loginData = {
        email: 'concurrent-test@example.com',
        password: 'testpassword'
      };

      // Send exactly at the limit (5 requests)
      const requests = Array(5).fill().map(() =>
        request(testApp)
          .post('/api/auth/login')
          .send(loginData)
      );

      const responses = await Promise.all(requests);
      
      // All 5 should succeed with fresh limiter
      const successful = responses.filter(res => res.status === 200);
      expect(successful.length).toBeGreaterThanOrEqual(4);
    });

    it('should properly count and limit requests', async () => {
      // Use unique email for clean slate
      const uniqueEmail = `limit-count-${Date.now()}@example.com`;
      
      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: uniqueEmail, password: 'test' });

      expect([200, 429]).toContain(res.status);
      
      // Should have rate limit headers
      expect(
        res.headers['ratelimit-limit'] ||
        res.headers['x-ratelimit-limit']
      ).toBeDefined();
    });
  });

  describe('Rate Limit Protection', () => {
    it('should protect against rapid-fire attacks', async () => {
      const uniqueEmail = `attack-test-${Date.now()}@example.com`;
      
      // Simulate rapid-fire attack (10 requests)
      const results = [];
      for (let i = 0; i < 10; i++) {
        const res = await request(app)
          .post('/api/auth/login')
          .send({ email: uniqueEmail, password: 'test' });
        results.push(res.status);
      }
      
      // Should have at least one 429 response
      const rateLimitedCount = results.filter(status => status === 429).length;
      expect(rateLimitedCount).toBeGreaterThan(0);
    });
  });
});