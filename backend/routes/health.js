// backend/routes/health.js
const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');

router.get('/', async (req, res) => {
  const health = {
    uptime: process.uptime(),
    timestamp: Date.now(),
    status: 'OK',
    checks: {
      database: 'unknown',
      memory: process.memoryUsage(),
      cpu: process.cpuUsage()
    }
  };

  try {
    // Check if mongoose connection exists and is ready
    if (mongoose.connection && mongoose.connection.readyState === 1) {
      // Only ping if connection is established
      await mongoose.connection.db.admin().ping();
      health.checks.database = 'connected';
    } else {
      health.checks.database = 'disconnected';
      health.status = 'DEGRADED';
    }
  } catch (error) {
    // If ping fails, mark as disconnected
    health.checks.database = 'disconnected';
    health.status = 'DEGRADED';
  }

  const statusCode = health.status === 'OK' ? 200 : 503;
  res.status(statusCode).json(health);
});

module.exports = router;