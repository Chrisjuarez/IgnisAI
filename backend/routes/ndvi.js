// ignis-ai-backend/routes/ndvi.js
const express = require('express');
const router  = express.Router();
const NDVI    = require('../models/NDVI');

router.get('/', async (req, res) => {
  const lat = parseFloat(req.query.lat);
  const lon = parseFloat(req.query.lon);
  if (isNaN(lat) || isNaN(lon)) {
    return res.status(400).json({ data: [] });
  }

  // ±0.02° ≈ 2 km
  const TOL = 0.02;
  const rec = await NDVI.findOne({
    latitude:  { $gte: lat - TOL, $lte: lat + TOL },
    longitude: { $gte: lon - TOL, $lte: lon + TOL }
  }).lean();

  // always return 200
  return res.json({ data: rec ? [rec] : [] });
});

module.exports = router;