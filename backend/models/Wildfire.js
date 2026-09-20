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

// FIRMS near-real-time products re-serve the same detections on every poll, so
// a detection's identity — where and when it was observed, and by which sensor
// — must be unique. Without this the archive grew by a full snapshot per
// request. Run scripts/dedupe-wildfires.js before this index can build on a
// collection that already contains duplicates.
WildfireSchema.index(
  { latitude: 1, longitude: 1, timestamp: 1, satellite: 1, instrument: 1 },
  { unique: true, name: 'firms_detection_identity' }
);

module.exports = mongoose.model('Wildfire', WildfireSchema);
