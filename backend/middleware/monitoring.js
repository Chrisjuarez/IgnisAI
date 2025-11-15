// backend/middleware/monitoring.js
const prometheus = require('prom-client');

// Create HTTP request duration histogram
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_ms',
  help: 'Duration of HTTP requests in ms',
  labelNames: ['method', 'route', 'status_code']
});

// Middleware to track request duration
const monitoringMiddleware = (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    httpRequestDuration.observe({
      method: req.method,
      route: req.route?.path || req.path,
      status_code: res.statusCode
    }, duration);
  });
  
  next();
};

// Export both the middleware AND a metrics handler
module.exports = monitoringMiddleware;

// Add metrics endpoint handler
module.exports.metricsHandler = async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
};