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