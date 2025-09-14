// ignis-ai-backend/models/Wildfire.js
const mongoose = require('mongoose');

const WildfireSchema = new mongoose.Schema({
  latitude:   { type: Number, required: true },
  longitude:  { type: Number, required: true },
  brightness: { type: Number, required: true },
  confidence: { type: String, required: true },   // still storing as string
  satellite:  { type: String, required: true },
  timestamp:  { type: Date,   required: true }    // now required
}, {
  timestamps: false   // you can enable createdAt/updatedAt if you’d like
});

module.exports = mongoose.model('Wildfire', WildfireSchema);