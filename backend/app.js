// backend/app.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./db');

dotenv.config();

const app = express();
app.set('trust proxy', 1);

const monitoring = require('./middleware/monitoring');
app.use(monitoring);
app.get('/metrics', monitoring.metricsHandler);

app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.path}`);
  next();
});

// CORS
const FRONTEND_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:3000';
app.use(cors({
  origin: FRONTEND_ORIGIN,
  methods: ['GET','POST','PUT','DELETE','OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization'],
  credentials: true
}));

app.use(express.json());

// Auth route
try {
  app.use('/api/auth', require('./routes/auth'));
} catch (e) {
  console.warn(`Skipping /api/auth: ${e.message}`);
}

app.use('/health', require('./routes/health')); // Advanced health with DB check

// Helper to mount optional routes
function tryMount(mountPath, modulePath) {
  try {
    const mod = require(modulePath);
    const handler =
      typeof mod === 'function'
        ? mod
        : (mod && typeof mod.default === 'function' ? mod.default : null);

    if (!handler) {
      throw new Error('argument handler must be a function');
    }
    app.use(mountPath, handler);
  } catch (e) {
    console.warn(`Skipping ${modulePath}: ${e.message}`);
  }
}

// API routes
tryMount('/api', './routes/fireData');
tryMount('/api/weather', './routes/weather');
tryMount('/api/topography', './routes/topography');
tryMount('/api/ndvi', './routes/ndvi');
tryMount('/api/predict-fire-spread', './routes/predictFireSpread');

// Export app for tests (your existing code - unchanged)
module.exports = app;

// ---- Run server directly (not in tests) ----
if (require.main === module) {
  const PORT = process.env.PORT || 5000;

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Backend listening on :${PORT}`);
  });

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

  process.on('SIGINT', () => { console.log('SIGINT received'); process.exit(0); });
  process.on('SIGTERM', () => { console.log('SIGTERM received'); process.exit(0); });
  process.on('unhandledRejection', err => console.error('Unhandled rejection:', err));
}