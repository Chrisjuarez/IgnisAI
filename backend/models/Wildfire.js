// backend/models/Wildfire.js
const mongoose = require('mongoose');
const WildfireSchema = new mongoose.Schema({
  latitude:   { type: Number, required: true },
  longitude:  { type: Number, required: true },
  brightness: { type: Number, required: true },
  confidence: { type: Number, required: true },  // ← number 0..100
  satellite:  { type: String, required: true },
  timestamp:  { type: Date,   required: true }
}, { timestamps: false });
module.exports = mongoose.model('Wildfire', WildfireSchema);