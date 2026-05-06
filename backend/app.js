// backend/app.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./db');

if (process.env.NODE_ENV !== 'test') {
  dotenv.config({ quiet: true });
}

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
const DEFAULT_CORS_ORIGINS = [
  'https://ignisai-frontend.onrender.com',
  'http://localhost:3000',
  'http://127.0.0.1:3000'
];
const CONFIGURED_CORS_ORIGINS = (process.env.CORS_ORIGIN || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);
const CORS_ORIGINS = [...new Set([...DEFAULT_CORS_ORIGINS, ...CONFIGURED_CORS_ORIGINS])];

app.use(cors({
  origin(origin, callback) {
    if (!origin || CORS_ORIGINS.includes(origin)) {
      return callback(null, true);
    }
    return callback(new Error(`CORS origin not allowed: ${origin}`));
  },
  methods: ['GET','POST','PUT','DELETE','OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization'],
  credentials: true
}));

app.use(express.json());

app.use('/health', require('./routes/health')); // Advanced health with DB check

function mountRoute(mountPath, modulePath) {
  const mod = require(modulePath);
  const handler =
    typeof mod === 'function'
      ? mod
      : (mod && typeof mod.default === 'function' ? mod.default : null);

  if (!handler) {
    throw new Error(`Failed to mount ${modulePath} at ${mountPath}: route module must export a function`);
  }

  app.use(mountPath, handler);
}

// API routes
mountRoute('/api/auth', './routes/auth');
mountRoute('/api', './routes/mapData');
mountRoute('/api', './routes/fireData');
mountRoute('/api/weather', './routes/weather');
mountRoute('/api/topography', './routes/topography');
mountRoute('/api/ndvi', './routes/ndvi');
mountRoute('/api/fire-perimeters', './routes/firePerimeters');
mountRoute('/api/predict-fire-spread', './routes/predictFireSpread');

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
