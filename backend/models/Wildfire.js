// backend/models/Wildfire.js
const mongoose = require('mongoose');
const WildfireSchema = new mongoose.Schema({
  latitude: { type: Number, required: true, index: true },
  longitude: { type: Number, required: true, index: true },
  brightness: { type: Number, required: true },
  confidence: { type: Number, required: true },  // ← number 0..100
  satellite:  { type: String, required: true },
  instrument: { type: String },
  product: { type: String },
  scan: { type: Number },
  track: { type: Number },
  frp: { type: Number },
  daynight: { type: String },
  version: { type: String },
  brightTi5: { type: Number },
  timestamp: { type: Date, required: true, index: true }
}, { timestamps: false });
module.exports = mongoose.model('Wildfire', WildfireSchema);
