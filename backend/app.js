// backend/app.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./db');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use('/api/auth', require('./routes/auth'));

// Health for CI/tests
app.get('/health', (_req, res) => res.status(200).json({ status: 'ok' }));

// Safely mount routes if they exist (avoids Jest crashing when a file is missing)
function tryMount(mountPath, modulePath) {
  try {
    app.use(mountPath, require(modulePath));
  } catch (e) {
    console.warn(`Skipping ${modulePath}: ${e.message}`);
  }
}

// IMPORTANT: match file name casing under backend/routes
tryMount('/api', './routes/fireData');                // or './routes/firedata' if lowercase
tryMount('/api/weather', './routes/weather');
tryMount('/api/topography', './routes/topography');
tryMount('/api/ndvi', './routes/ndvi');
tryMount('/api/predict-fire-spread', './routes/predictFireSpread');
tryMount('/api/auth', './routes/auth');

// Export app for tests
module.exports = app;

// Only start server when run directly
if (require.main === module) {
  (async () => {
    await connectDB(); // db.js no-ops in tests
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => console.log(`🚀 Backend on :${PORT}`));
  })().catch(err => {
    console.error('❌ Startup error:', err);
    process.exit(1);
  });
}