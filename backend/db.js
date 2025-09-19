// backend/db.js
require('dotenv').config();
const mongoose = require('mongoose');

const mongoURI = process.env.MONGODB_URI;

async function connectDB() {
  if (process.env.NODE_ENV === 'test') return; // <-- no DB in tests
  if (!mongoURI) throw new Error('MONGODB_URI is not set');
  if (mongoose.connection.readyState >= 1) return mongoose.connection;
  await mongoose.connect(mongoURI);
  return mongoose.connection;
}

module.exports = connectDB;