// backend/db.js
require('dotenv').config();
const mongoose = require('mongoose');

const mongoURI = process.env.MONGODB_URI;

async function connectDB() {
  if (process.env.NODE_ENV === 'test') {
    console.log('🧪 Test mode: skipping Mongo connection');
    return;
  }
  if (!mongoURI) {
    throw new Error('MONGODB_URI is not set');
  }
  if (mongoose.connection.readyState === 1) {
    console.log('✅ Mongo already connected');
    return mongoose.connection;
  }

  // Helpful logs
  mongoose.connection.on('connected', () => console.log('✅ Mongo connected'));
  mongoose.connection.on('error', (err) => console.error('❌ Mongo error:', err.message));
  mongoose.connection.on('disconnected', () => console.warn('⚠️ Mongo disconnected'));

  // Reasonable connect settings; Atlas uses TLS automatically with SRV
  await mongoose.connect(mongoURI, {
    serverSelectionTimeoutMS: 5000,  // fail fast if Atlas not reachable/allowed
    socketTimeoutMS: 20000,
    maxPoolSize: 10,
    retryWrites: true,
    w: 'majority',
    appName: 'IgnisAI'
  });

  return mongoose.connection;
}

module.exports = connectDB;