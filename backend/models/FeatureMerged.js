// models/FeatureMerged.js
const mongoose = require('mongoose');

const FeatureMergedSchema = new mongoose.Schema({
  location: {
    type: { type: String, enum: ['Point'], default: 'Point' },
    coordinates: { type: [Number], required: true } // [lon, lat]
  },
  timestamp: { type: Date, required: true },

  // ——— HistoricalWildfire fields ———
  brightness:   { type: Number },  // bright_ti4
  brightness5:  { type: Number },  // bright_ti5
  scan:         { type: Number },
  track:        { type: Number },
  frp:          { type: Number },
  confidence:   { type: String },  // "n" | "l" | "h"
  daynight:     { type: String },  // "D" | "N"
  version:      { type: Number },
  type:         { type: Number },

  satellite:    { type: String },
  instrument:   { type: String },
  acq_date:     { type: String },
  acq_time:     { type: String },

  // ——— HistoricalWeather observations ———
  weatherObservations: [{
    station: String,
    name:    String,
    date:    Date,
    awnd:    Number,
    pgtm:    Number,
    prcp:    Number,
    rhav:    Number,
    tmax:    Number,
    tmin:    Number,
    wdfg:    Number
  }],

  // ——— Topography ———
  elevation:      { type: Number },
  slope:          { type: Number },
  vegetationType: { type: String },

  // ——— NDVI ———
  ndvi: { type: Number }
});

// geospatial index on location
FeatureMergedSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('FeatureMerged', FeatureMergedSchema);