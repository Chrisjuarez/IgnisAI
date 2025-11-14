// routes/ ndvi.js
const express = require('express');
const router = express.Router();

function buildTemplate(dateParam) {
  const isoDay = dateParam
    ? new Date(dateParam).toISOString().slice(0, 10)
    : new Date().toISOString().slice(0, 10);

  return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_16Day/default/${isoDay}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png`;
}

function respondWithTemplate(req, res) {
  const template = buildTemplate(req.query.date);
  res.json({ ok: true, template, attribution: 'NASA GIBS MODIS NDVI (16-day)' });
}

router.get('/', respondWithTemplate);
router.get('/tile', respondWithTemplate);

module.exports = router;
