const mongoose = require('mongoose');

// Increase timeout for all tests
jest.setTimeout(30000);

// Global test setup
beforeAll(async () => {
  // Ensure clean test environment
  if (mongoose.connection.readyState !== 0) {
    await mongoose.connection.close();
  }
});

afterAll(async () => {
  // Clean up after all tests
  if (mongoose.connection.readyState !== 0) {
    await mongoose.connection.close();
  }
});
