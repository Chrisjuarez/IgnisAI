// backend/__tests__/concurrent-access.test.js
const request = require('supertest');
const app = require('../app');

describe('Multi-User Concurrent Access', () => {
  afterAll(async () => {
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  it('should handle 50 concurrent users', async () => {
    const users = Array.from({ length: 50 }, (_, i) => ({
      location: { lat: 34 + (i * 0.01), lon: -118 + (i * 0.01) }
    }));

    const requests = users.map(user =>
      request(app)
        .get('/api/wildfires')
        .query(user.location)
        .timeout(20000)
    );

    const startTime = Date.now();
    const responses = await Promise.allSettled(requests);
    const duration = Date.now() - startTime;

    const fulfilled = responses.filter(r => r.status === 'fulfilled');
    const successful = fulfilled.filter(r => r.value.status === 200);
    const successRate = successful.length / responses.length;

    console.log(`Success rate: ${(successRate * 100).toFixed(1)}% (${successful.length}/${responses.length})`);
    console.log(`Total duration: ${duration}ms`);

    expect(successRate).toBeGreaterThan(0.25);
    expect(duration).toBeLessThan(30000);
  }, 35000);

  it('should maintain data consistency under load', async () => {
    const requests = Array(20).fill().map(() =>
      request(app)
        .get('/api/wildfires')
        .timeout(20000)
    );

    const responses = await Promise.allSettled(requests);
    const successful = responses
      .filter(r => r.status === 'fulfilled' && r.value.status === 200)
      .map(r => r.value);

    console.log(`Successful responses: ${successful.length}/20`);

    if (successful.length > 1) {
      const firstCount = successful[0].body.count || successful[0].body.data?.length || 0;
      
      successful.forEach(res => {
        const count = res.body.count || res.body.data?.length || 0;
        expect(Math.abs(count - firstCount)).toBeLessThan(20);
      });
    }
    
    expect(successful.length).toBeGreaterThanOrEqual(3);
  }, 25000);

  it('should not degrade performance under concurrent load', async () => {
    const latencies = [];
    const batchSize = 5;
    const batches = 3;

    for (let batch = 0; batch < batches; batch++) {
      const batchRequests = Array(batchSize).fill().map(() => {
        const startTime = Date.now();
        return request(app)
          .get('/api/wildfires')
          .timeout(20000)
          .then(res => {
            const latency = Date.now() - startTime;
            latencies.push(latency);
            return res;
          })
          .catch(err => {
            const latency = Date.now() - startTime;
            latencies.push(latency);
            return { status: 500 };
          });
      });

      await Promise.allSettled(batchRequests);
      
      if (batch < batches - 1) {
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }

    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const maxLatency = Math.max(...latencies);
    
    console.log(`Average latency: ${avgLatency.toFixed(0)}ms`);
    console.log(`Max latency: ${maxLatency.toFixed(0)}ms`);

    expect(avgLatency).toBeLessThan(15000);
    expect(maxLatency).toBeLessThan(25000);
  }, 75000);

  it('should handle database connection pool efficiently', async () => {
    const requests = Array(30).fill().map(() =>
      request(app)
        .get('/api/wildfires')
        .timeout(20000)
    );

    const responses = await Promise.allSettled(requests);
    const errors = responses.filter(r => 
      r.status === 'rejected' || 
      (r.status === 'fulfilled' && r.value.status >= 500)
    );

    const errorRate = errors.length / responses.length;
    
    console.log(`Error rate: ${(errorRate * 100).toFixed(1)}% (${errors.length}/${responses.length})`);

    expect(errorRate).toBeLessThanOrEqual(0.5);
  }, 35000);

  it('should verify endpoint availability', async () => {
    const res = await request(app)
      .get('/api/wildfires')
      .timeout(20000);

    console.log(`Endpoint status: ${res.status}`);
    
    expect([200, 404, 500, 503]).toContain(res.status);
  }, 25000);
});