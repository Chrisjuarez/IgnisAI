// routes/ ndvi.js                 const express = require('express');
const router = express.Router();

router.get('/tile', (req, res) => {
  const { date } = req.query;
  const yyyy_mm_dd = (date && /^\d{4}-\d{2}-\d{2}$/.test(date))
    ? date
    : new Date().toISOString().slice(0,10);

  const template = https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_16Day/default/${yyyy_mm_dd}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png;

  res.json({ ok: true, template, attribution: 'NASA GIBS MODIS NDVI (16-day)' });
});

module.exports = router;
