// backend/app.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./db');

dotenv.config();

const app = express();
app.set('trust proxy', 1);

// CORS (frontend can be http://<EC2> because nginx serves it on :80)
const FRONTEND_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:3000';
app.use(cors({
  origin: FRONTEND_ORIGIN,
  methods: ['GET','POST','PUT','DELETE','OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization'],
  credentials: true
}));

app.use(express.json());

// Auth route (safe even if DB not ready yet)
try {
  app.use('/api/auth', require('./routes/auth'));
} catch (e) {
  console.warn(`Skipping /api/auth: ${e.message}`);
}

// Health for CI/tests — stays responsive even before DB connects
app.get(['/health', '/api/health'], (_req, res) => res.status(200).json({ status: 'ok' }));

// Helper to mount optional routes without crashing
function tryMount(mountPath, modulePath) {
  try {
    app.use(mountPath, require(modulePath));
  } catch (e) {
    console.warn(`Skipping ${modulePath}: ${e.message}`);
  }
}

// IMPORTANT: match file name casing under backend/routes
tryMount('/api', './routes/fireData');        // or './routes/firedata' if you renamed
tryMount('/api/weather', './routes/weather');
tryMount('/api/topography', './routes/topography');
tryMount('/api/ndvi', './routes/ndvi');

// 🔹 proxy to tilesvc
tryMount('/api/predict-fire-spread', './routes/predictFireSpread');

// Export app for tests (supertest uses this path and won't start the server)
module.exports = app;

// ---- Run server directly (not in tests) ----
if (require.main === module) {
  const PORT = process.env.PORT || 5000;

  // 1) Start HTTP server first so health checks pass immediately
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Backend listening on :${PORT}`);
  });

  // 2) Connect to Mongo in the background with retry
  const tryConnect = async (delayMs = 3000) => {
    try {
      await connectDB(); // use MONGODB_URI from env
      console.log('✅ Mongo connected');
    } catch (err) {
      console.error(`⚠️  Mongo connect failed: ${err.message} — retrying in ${delayMs}ms`);
      setTimeout(() => tryConnect(delayMs), delayMs);
    }
  };
  tryConnect();

  // Graceful shutdown logs (optional)
  process.on('SIGINT', () => { console.log('SIGINT received'); process.exit(0); });
  process.on('SIGTERM', () => { console.log('SIGTERM received'); process.exit(0); });
  process.on('unhandledRejection', err => console.error('Unhandled rejection:', err));
}