// backend/models/HistoricalWildfire.js
const mongoose = require('mongoose');

const HistoricalWildfireSchema = new mongoose.Schema({
    latitude: { type: Number, required: true },
    longitude: { type: Number, required: true },
    brightness: { type: Number, required: false }, // Recommended to store brightness
    confidence: { type: Number, required: false },
    satellite: { type: String, required: false },
    instrument: { type: String, required: false },
    acq_date: { type: String, required: false },
    acq_time: { type: String, required: false },
    year: { type: Number, required: false },
    fireName: { type: String, required: false },  // useful for historical reference
    source: { type: String, required: false },
    timestamp: { type: Date, required: false },
});

module.exports = mongoose.model('HistoricalWildfire', HistoricalWildfireSchema);