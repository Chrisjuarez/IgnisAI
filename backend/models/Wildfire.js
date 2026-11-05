// backend/models/Wildfire.js
const mongoose = require('mongoose');

// The map reads at most the last 5 days of detections, so the TTL index on
// `timestamp` lets MongoDB delete anything much older. The floor stops a bad
// env value from expiring detections the map still reads, or ones the next
// FIRMS poll would re-serve and the ingest re-insert. On a database that
// already has a plain `timestamp_1` index (production does), this definition
// is not applied automatically; see "Detection Retention" in docs/runbook.md.
const DEFAULT_RETENTION_DAYS = 30;
const MIN_RETENTION_DAYS = 7;

function retentionDays(raw = process.env.WILDFIRE_RETENTION_DAYS) {
  const n = Number(raw || DEFAULT_RETENTION_DAYS);
  if (!Number.isFinite(n)) return DEFAULT_RETENTION_DAYS;
  return Math.max(MIN_RETENTION_DAYS, Math.round(n));
}

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
  timestamp: {
    type: Date,
    required: true,
    index: { expireAfterSeconds: retentionDays() * 24 * 60 * 60 },
  }
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
