// backend/routes/historicalWildfires
// Fetches fire acticity for the area around LA and Malibu from 2025-01-07 to 2025-01-31
const express = require('express');
const axios = require('axios');
const csv = require('csv-parser');
const stream = require('stream');
const historicalWildfire = require('../models/HistoricalWildfire');
const router = express.Router();

const MAP_KEY = process.env.NASA_API_KEY; // NASA FIRMS API key

// Helper: Break a large date range into chunks of maxDays (default 10)
function chunkDateRange(start, end, maxDays = 10) {
    const chunks = [];
    let currentStart = new Date(start);
    const finalEnd = new Date(end);

    while (currentStart <= finalEnd) {
        let currentEnd = new Date(currentStart);
        currentEnd.setDate(currentEnd.getDate() + maxDays - 1);
        if (currentEnd > finalEnd) {
            currentEnd = finalEnd;
        }
        const chunkStart = currentStart.toISOString().split('T')[0];
        const chunkEnd = currentEnd.toISOString().split('T')[0];
        chunks.push({ startDate: chunkStart, endDate: chunkEnd });
        // Move currentStart to the day after currentEnd
        currentStart.setDate(currentEnd.getDate() + 1);
    }
    return chunks;
}

// Helper: Parse CSV data from a string into an array of JSON records.
function parseCSVData(csvData) {
    return new Promise((resolve, reject) => {
        const results = [];
        const readStream = new stream.Readable();
        readStream.push(csvData);
        readStream.push(null);

        readStream
            .pipe(csv())
            .on('data', (data) => {
                const dateStr = data.acq_date ? data.acq_date.trim() : null;
                const timeStr = data.acq_time ? data.acq_time.trim() : '1200'; // fallback to noon if missing
                let mergedTimestamp = null;
                if (dateStr) {
                    const padded = timeStr.padStart(4, '0');
                    const hours = padded.slice(0, 2);
                    const mins = padded.slice(2);
                    mergedTimestamp = new Date(`${dateStr}T${hours}:${mins}`);
                }
                results.push({
                    latitude: parseFloat(data.latitude),
                    longitude: parseFloat(data.longitude),
                    brightness: data.bright_ti4 ? parseFloat(data.bright_ti4) : null,
                    confidence: data.confidence && !isNaN(Number(data.confidence)) ? Number(data.confidence) : null,
                    satellite: data.satellite ? data.satellite.trim() : null,
                    instrument: data.instrument ? data.instrument.trim() : null,
                    acq_date: dateStr,
                    acq_time: timeStr,
                    year: dateStr ? new Date(dateStr).getFullYear() : null,
                    timestamp: mergedTimestamp
                });
            })
            .on('end', () => resolve(results))
            .on('error', reject);
        });
}

// Define your historical fires – note the date range covers January 7 to 31, 2025.
const historicalFires = [
    {
        name: "Eaton Fire",
        boundingBox: "-118.176917,34.114582,-118.004912,34.273324",
        startDate: "2025-01-07",
        endDate: "2025-01-31"
    },
    {
        name: "Palisades",
        boundingBox: "-118.654052,33.987455,-118.450805,34.171152",
        startDate: "2025-01-07",
        endDate: "2025-01-31"
    }
];

router.get('/historical-wildfires', async (req, res) => {
    try {
        // Change the default product to one that may provide more complete historical data.
        // Here, we use "VIIRS_SNPP_NRT" (Near Real-Time) in lieu of the standard "VIIRS_SNPP_SP".
        const product = req.query.product || 'VIIRS_SNPP_NRT';
        const overallResults = [];

        // Loop through each fire event.
        for (const fireEvent of historicalFires) {
            let eventRecordsTotal = 0;
            const chunks = chunkDateRange(fireEvent.startDate, fireEvent.endDate, 10);

            for (const chunk of chunks) {
                // Calculate the number of days in this chunk (max 10)
                const days = Math.min(
                    (new Date(chunk.endDate) - new Date(chunk.startDate)) / (1000 * 60 * 60 * 24) + 1,
                    10
                );

                const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${MAP_KEY}/${product}/${fireEvent.boundingBox}/${days}/${chunk.startDate}`;
                console.log(`Fetching data for ${fireEvent.name} from: ${url}`);

                try {
                    const response = await axios.get(url, { responseType: 'text' });
                    const csvData = response.data;

                    if (!csvData || csvData.trim() === "") {
                        console.warn(`No CSV data returned for ${fireEvent.name} between ${chunk.startDate} and ${chunk.endDate}`);
                        continue;
                    }

                    const records = await parseCSVData(csvData);
                    console.log(`Fetched ${records.length} records for ${fireEvent.name} (${chunk.startDate} to ${chunk.endDate})`);

                    if (records.length > 0) {
                        records.forEach(record => {
                            record.fireName = fireEvent.name;
                            record.source = 'NASA FIRMS';
                        });

                        await historicalWildfire.insertMany(records);
                        eventRecordsTotal += records.length;
                    }
                } catch (chunkErr) {
                    console.error(
                        `Error processing chunk from ${chunk.startDate} to ${chunk.endDate} for ${fireEvent.name}:`,
                        chunkErr.message
                    );
                }
            }

            overallResults.push({ fireName: fireEvent.name, totalRecords: eventRecordsTotal });
        }

        res.json({
            message: "Historical wildfire data processed successfully.",
            results: overallResults
        });
    } catch (err) {
        console.error("Error in historical-wildfires endpoint:", err.message);
        res.status(500).json({
            error: "Failed to process historical wildfire data",
            details: err.message
        });
    }
});

module.exports = router;