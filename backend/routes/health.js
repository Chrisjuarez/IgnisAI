// backend/routes/health.js
const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');
const axios = require('axios');

const TILE_SVC = process.env.TILE_SVC || process.env.TILE_SVC_BASE || 'http://localhost:8008';

router.get('/', async (req, res) => {
  const health = {
    uptime: process.uptime(),
    timestamp: Date.now(),
    status: 'OK',
    checks: {
      database: 'unknown',
      tilesvc: 'unknown',
      predictions: process.env.PREDICTIONS_ENABLED === 'false' ? 'disabled' : 'enabled',
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

  try {
    const { data } = await axios.get(`${TILE_SVC}/healthz`, { timeout: 2500 });
    health.checks.tilesvc = {
      status: 'connected',
      predictionsEnabled: data?.predictionsEnabled,
      modelExists: data?.modelExists,
      modelSha256: data?.modelSha256,
      targetMode: data?.target_mode,
      runtimeArchVersion: data?.runtime_arch_version,
      staticCatalog: data?.staticCatalog,
      calibration: data?.calibration,
      firmsSnapshot: data?.firmsSnapshot,
      noaaCycle: data?.noaaCycle,
      weatherQuality: data?.weatherQuality,
      mlSource: data?.mlSource,
    };
    if (data?.staticCatalog && data.staticCatalog.ok === false) {
      health.status = 'DEGRADED';
    }
    if (data?.calibration && data.calibration.ok === false) {
      health.status = 'DEGRADED';
    }
    if (data?.predictionsEnabled !== false && data?.modelExists === false) {
      health.status = 'DEGRADED';
    }
    if (data?.predictionsEnabled !== false && data?.firmsSnapshot?.configured && data.firmsSnapshot.ok === false) {
      health.status = 'DEGRADED';
    }
    if (data?.predictionsEnabled !== false && data?.noaaCycle?.configured && data.noaaCycle.ok === false) {
      health.status = 'DEGRADED';
    }
  } catch (error) {
    health.checks.tilesvc = 'disconnected';
    health.status = 'DEGRADED';
  }

  const statusCode = health.status === 'OK' ? 200 : 503;
  res.status(statusCode).json(health);
});

module.exports = router;
