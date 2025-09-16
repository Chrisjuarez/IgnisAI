// ignis-ai-backend/models/HistoricalWeather.js
const mongoose = require('mongoose');

const HistoricalWeatherSchema = new mongoose.Schema({
  station: { type: String },
  name: { type: String },
  latitude: { type: Number },
  longitude: { type: Number },
  elevation: { type: Number },
  date: { type: Date, required: true },
  adpt: { type: String },
  awnd: { type: Number },  // Average Daily Wind Speed
  pgtm: { type: Number },  // PGTM (often wind direction or pressure; check NOAA docs)
  prcp: { type: Number },  // Precipitation
  rhav: { type: Number },  // Relative Humidity (if present)
  tmax: { type: Number },  // Max Temp
  tmin: { type: Number },  // Min Temp
  wdfg: { type: Number }   // Additional wind field
});

module.exports = mongoose.model('HistoricalWeather', HistoricalWeatherSchema);