// backend/db.js
require('dotenv').config();
const mongoose = require('mongoose');

const mongoURI = process.env.MONGODB_URI;

async function connectDB() {
  if (process.env.NODE_ENV === 'test') {
    console.log('🧪 Test mode: Skipping MongoDB connection');
    return;
  }
  
  if (!mongoURI) {
    throw new Error('❌ MONGODB_URI is not set in environment variables');
  }
  
  if (mongoose.connection.readyState >= 1) {
    console.log('✅ MongoDB already connected');
    return mongoose.connection;
  }
  
  try {
    await mongoose.connect(mongoURI);
    console.log(`✅ MongoDB Connected: ${mongoose.connection.host}:${mongoose.connection.port}/${mongoose.connection.name}`);
    return mongoose.connection;
  } catch (error) {
    console.error('❌ MongoDB connection failed:', error.message);
    process.exit(1);
  }
}

module.exports = connectDB;