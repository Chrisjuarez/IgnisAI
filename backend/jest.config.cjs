module.exports = {
  testEnvironment: 'node',
  testTimeout: 30000,
  collectCoverage: true,
  setupFilesAfterEnv: ['<rootDir>/__tests__/setupTests.js'],
  testMatch: [
    '**/__tests__/**/*.test.js',
    '**/?(*.)+(spec|test).js'
  ], // Only run actual test files
  testPathIgnorePatterns: [
    '/node_modules/',
    '/__tests__/setupTests.js' // Explicitly ignore setup file
  ],
  collectCoverageFrom: ['app.js', 'routes/**/*.js', 'models/**/*.js'],
  coverageReporters: ['text', 'lcov']
};