//ignis-ai-backend/routes/historicalFireProcessor.js
const express = require('express');
const axios = require('axios');
const csv = require('csv-parser');
const stream = require('stream');
const clustering = require('density-clustering');
const router = express.Router();

const MAP_KEY = process.env.NASA_API_KEY; // Your NASA FIRMS MAP_KEY

// Helper: Compute Haversine distance (in meters)
function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371e3; // Earth's radius in meters
  const toRadians = (deg) => (deg * Math.PI) / 180;
  const φ1 = toRadians(lat1);
  const φ2 = toRadians(lat2);
  const Δφ = toRadians(lat2 - lat1);
  const Δλ = toRadians(lon2 - lon1);
  const a = Math.sin(Δφ / 2) ** 2 +
            Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Helper: Compute bearing between two points (in degrees)
function computeBearing(lat1, lon1, lat2, lon2) {
  const toRadians = (deg) => (deg * Math.PI) / 180;
  const toDegrees = (rad) => (rad * 180) / Math.PI;
  const φ1 = toRadians(lat1);
  const φ2 = toRadians(lat2);
  const Δλ = toRadians(lon2 - lon1);
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) -
            Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  const θ = Math.atan2(y, x);
  return (toDegrees(θ) + 360) % 360;
}

async function fetchHistoricalFIRMSData() {
  const startDate = '2020-01-01';
  const endDate = '2022-12-31';
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${MAP_KEY}/MODIS_SP/-125.0,24.0,-66.0,49.0/0?startDate=${startDate}&endDate=${endDate}`;
  console.log("Fetching FIRMS CSV from:", url);
  const response = await axios.get(url, { responseType: 'text' });
  return response.data;
}

router.get('/process-historical-fires', async (req, res) => {
  try {
    const csvData = await fetchHistoricalFIRMSData();
    
    const csvStream = new stream.Readable();
    csvStream.push(csvData);
    csvStream.push(null);

    const records = [];
    await new Promise((resolve, reject) => {
      csvStream
        .pipe(csv())
        .on('data', (data) => {
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
    
    // Use DBSCAN to cluster records spatially (by latitude and longitude).
    const dbscan = new clustering.DBSCAN();
    const dataset = records.map(rec => [rec.latitude, rec.longitude]);
    
    // epsilon: clustering radius (in degrees); minPts: minimum points to form a cluster.
    const epsilon = req.query.epsilon ? parseFloat(req.query.epsilon) : 0.1;
    const minPts = req.query.minPts ? parseInt(req.query.minPts, 10) : 3;
    const clusters = dbscan.run(dataset, epsilon, minPts);
    
    console.log(`Found ${clusters.length} clusters (potential fire events).`);
    
    // Basic cluster information for now
    const clusterSummary = clusters.map((cluster, index) => ({
      clusterId: index,
      detectionCount: cluster.length,
      firstDetection: records[cluster[0]],
      lastDetection: records[cluster[cluster.length - 1]]
    }));

    res.json({
      message: "Fire events clustered using DBSCAN algorithm.",
      recordCount: records.length,
      clusterCount: clusters.length,
      clusteringParams: { epsilon, minPts },
      clusters: clusterSummary
    });

  } catch (err) {
    console.error("Error processing historical fire data:", err.message);
    res.status(500).json({ error: "Processing error", details: err.message });
  }
});

module.exports = router;