//ignis-ai-backend/routes/historicalFireProcessor.js
const express = require('express');
const axios = require('axios');
const csv = require('csv-parser');
const stream = require('stream');
const router = express.Router();

const MAP_KEY = process.env.NASA_API_KEY; // Your NASA FIRMS MAP_KEY

// Fetch historical FIRMS data from the API.
// Adjust the dataset identifier and bounding box as needed; here we use MODIS_SP.
async function fetchHistoricalFIRMSData() {
  // Dates can be further parameterized. Here are defaults.
  const startDate = '2020-01-01';
  const endDate = '2022-12-31';
  // The bounding box here is for the contiguous US; adjust accordingly.
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${MAP_KEY}/MODIS_SP/-125.0,24.0,-66.0,49.0/0?startDate=${startDate}&endDate=${endDate}`;
  console.log("Fetching FIRMS CSV from:", url);
  const response = await axios.get(url, { responseType: 'text' });
  return response.data;
}

// Define the route to process historical fire data.
router.get('/process-historical-fires', async (req, res) => {
  try {
    const csvData = await fetchHistoricalFIRMSData();
    
    // Create a readable stream from the CSV data.
    const csvStream = new stream.Readable();
    csvStream.push(csvData);
    csvStream.push(null);

    // Parse CSV rows into an array of records.
    const records = [];
    await new Promise((resolve, reject) => {
      csvStream
        .pipe(csv())
        .on('data', (data) => {
          // Map FIRMS attributes (adjust field names based on the actual CSV).
          const timestamp = data.acq_date ? new Date(data.acq_date) : null;
          records.push({
            latitude: parseFloat(data.latitude),
            longitude: parseFloat(data.longitude),
            brightness: data.bright_ti4 ? parseFloat(data.bright_ti4) : null,
            acq_date: data.acq_date ? data.acq_date.trim() : null,
            acq_time: data.acq_time ? data.acq_time.trim() : null,
            timestamp,
            satellite: data.satellite ? data.satellite.trim() : null,
            instrument: data.instrument ? data.instrument.trim() : null,
            confidence: data.confidence ? parseFloat(data.confidence) : null
          });
        })
        .on('end', resolve)
        .on('error', reject);
    });

    console.log(`Fetched ${records.length} FIRMS records.`);
    
    res.json({
      message: "Historical fire data fetched successfully.",
      recordCount: records.length,
      sampleRecords: records.slice(0, 5) // Return first 5 records as sample
    });

  } catch (err) {
    console.error("Error processing historical fire data:", err.message);
    res.status(500).json({ error: "Processing error", details: err.message });
  }
});

module.exports = router;