const mongoose = require('mongoose');

const TopographySchema = new mongoose.Schema({
  latitude: Number,
  longitude: Number,
  fetchedAt: { type: Date, default: Date.now },
  elevation: Number,
  slope_deg: Number,
  aspect_deg: Number
}, { timestamps: true });

module.exports = mongoose.model('Topography', TopographySchema);
