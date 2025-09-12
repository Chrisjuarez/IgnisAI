// backend/app.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./db'); 

dotenv.config();

// Connect to MongoDB
connectDB();

const app = express();
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('Ignis AI Backend is running');
});

app.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok' });
});

// export the app so tests can import it
module.exports = app;

// Import API routes
const fireDataRoutes = require('./routes/fireData'); // Fire Data
const weatherRoutes = require('./routes/weather'); // Ensure this is imported
const topographyRoutes = require('./routes/topography'); // Topography Data
//const humanFactorsRoutes = require('./routes/humanFactors'); // Human Factors Data
const predictFireSpreadRouter = require('./routes/predictFireSpread');

app.use('/api', fireDataRoutes);
app.use('/api/weather', weatherRoutes);
app.use('/api/topography', topographyRoutes);
//app.use('/api/human-factors', humanFactorsRoutes);
app.use('/api/predict-fire-spread', predictFireSpreadRouter);

// Only start the server if this file is run directly: `node backend/app.js`
if (require.main === module) {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () => {
    console.log(`Backend listening on http://localhost:${PORT}`);
  });
}