// backend/db.js
const dotenv = require('dotenv');
const mongoose = require('mongoose');

if (process.env.NODE_ENV !== 'test') {
  dotenv.config({ quiet: true });
}

let listenersAttached = false;

async function connectDB() {
  // Prefer production-style variable, but allow CI/tests to use MONGODB_TEST_URI.
  const mongoURI =
    process.env.MONGODB_URI ||
    process.env.MONGODB_TEST_URI ||
    process.env.MONGO_URL;

  // Allow explicit opt-out (useful for pure unit tests that mock DB/models).
  if (process.env.MONGO_SKIP_CONNECT === 'true') {
    console.log('⚠️  MONGO_SKIP_CONNECT=true — skipping Mongo connection');
    return mongoose.connection;
  }

  if (!mongoURI) {
    throw new Error('MONGODB_URI is not set (or MONGODB_TEST_URI)');
  }

  if (mongoose.connection.readyState === 1) {
    return mongoose.connection; // already connected
  }

  if (!listenersAttached) {
    listenersAttached = true;
    mongoose.connection.on('connected', () => console.log('✅ Mongo connected'));
    mongoose.connection.on('error', (err) =>
      console.error('❌ Mongo error:', err.message)
    );
    mongoose.connection.on('disconnected', () =>
      console.warn('⚠️  Mongo disconnected')
    );
  }

  await mongoose.connect(mongoURI, {
    serverSelectionTimeoutMS: 5000,
    socketTimeoutMS: 20000,
    maxPoolSize: 10,
    retryWrites: true,
    w: 'majority',
    appName: 'IgnisAI',
  });

  return mongoose.connection;
}

module.exports = connectDB;
