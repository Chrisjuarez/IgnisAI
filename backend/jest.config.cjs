module.exports = {
  testEnvironment: 'node',
  testTimeout: 30000,
  collectCoverage: true,
  setupFilesAfterEnv: ['<rootDir>/__tests__/setupTests.js'],
  collectCoverageFrom: [
    'app.js', 
    'routes/**/*.js', 
    'models/**/*.js',
    '!routes/ndvi.js'
  ],
  coverageReporters: ['text', 'lcov']
};