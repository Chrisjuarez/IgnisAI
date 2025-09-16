//ignis-ai-backend/models/NDVI.js
const mongoose = require('mongoose');

const NDVISchema = new mongoose.Schema({
  latitude: Number,
  longitude: Number,
  date: Date,
  value: Number
});

module.exports = mongoose.model('NDVI', NDVISchema);