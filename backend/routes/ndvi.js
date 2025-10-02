// routes/ ndvi.js
const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  const date = req.query.date 
    ? new Date(req.query.date).toISOString().slice(0,10) 
    : new Date().toISOString().slice(0,10);

  const yyyy_mm_dd = date;

  const template = `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_16Day/default/${yyyy_mm_dd}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png`;

  res.json({ ok: true, template, attribution: 'NASA GIBS MODIS NDVI (16-day)' });
});

module.exports = router;
