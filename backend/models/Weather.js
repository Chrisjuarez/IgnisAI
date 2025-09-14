// backend/models/Weather.js
const mongoose = require('mongoose');

const HourlySchema = new mongoose.Schema({
  time: [String],
  temperature_2m: [Number],
  relative_humidity_2m: [Number],
  wind_speed_10m: [Number],
  wind_gusts_10m: [Number],
  precipitation: [Number]
}, { _id: false });

const WeatherSchema = new mongoose.Schema({
  latitude: Number,
  longitude: Number,
  fetchedAt: { type: Date, default: Date.now },
  current: { type: mongoose.Schema.Types.Mixed, default: {} },
  hourly:  { type: HourlySchema, default: {} }
}, { timestamps: true });

module.exports = mongoose.model('Weather', WeatherSchema);
