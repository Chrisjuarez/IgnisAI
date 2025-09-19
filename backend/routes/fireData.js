// ignis-ai-backend/routes/fireData.js
const express = require('express');
const router = express.Router();
const axios  = require('axios');
const Wildfire = require('../models/Wildfire');
require('dotenv').config();

// Parse the 14-col “area” CSV from FIRMS into objects
function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  const dataLines = lines.slice(1);            // drop header

  return dataLines
    .map(line => {
      const cols = line.split(',');
      if (cols.length < 14) return null;       // malformed row

      // Column ordering: latitude,longitude,bright_ti4,scan,track,
      //                 acq_date,acq_time,satellite,instrument,
      //                 confidence,version,bright_ti5,frp,daynight
      const [
        lat, lon, brightTi4, scan, track,
        acq_date, acq_time,
        satellite, instrument,
        confidence, version,
        brightTi5, frp, daynight
      ] = cols;

      // build ISO timestamp: “YYYY-MM-DDTHH:MM:00Z”
      const hhmm = acq_time.padStart(4, '0');
      const iso = `${acq_date}T${hhmm.slice(0,2)}:${hhmm.slice(2)}:00Z`;

      return {
        latitude:   parseFloat(lat),
        longitude:  parseFloat(lon),
        brightness: parseFloat(brightTi4),
        confidence: cols[3],                  // stored as string (e.g. "0.78")
        satellite,
        timestamp:  new Date(iso)
      };
    })
    .filter(f =>
      f &&
      !isNaN(f.latitude) &&
      !isNaN(f.longitude) &&
      !isNaN(f.brightness) &&
      f.timestamp.toString() !== 'Invalid Date'
    );
}

router.get('/wildfires', async (req, res) => {
  try {
    const url = [
      'https://firms.modaps.eosdis.nasa.gov/api/area/csv',
      process.env.NASA_API_KEY,
      'VIIRS_NOAA21_NRT',
      '-125.0,24.0,-66.0,49.0',
      '2'
    ].join('/');

    console.log(`Fetching FIRMS area data from:\n  ${url}`);
    const { data: csvText } = await axios.get(url, {
      headers: { 'User-Agent': 'ignis-ai (chrisjuarez1596@gmail.com)' },
      responseType: 'text'
    });

    const fires = parseCSV(csvText);
    if (fires.length === 0) {
      console.log('⚠️  No valid wildfire rows parsed.');
      return res.json({ message: 'No valid fire data', count: 0, data: [] });
    }

    // insertMany will bulk‐insert all your clean objects
    const inserted = await Wildfire.insertMany(fires);
    console.log(`🔥  Inserted ${inserted.length} wildfires`);

    res.json({
      message: 'Wildfire data fetched & stored',
      count: inserted.length,
      data: inserted
    });

  } catch (err) {
    console.error('❌ Error fetching wildfire data:', err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;