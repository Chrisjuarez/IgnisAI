module.exports = {
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: ['app.js', 'routes/**/*.js', 'models/**/*.js'],
  coverageReporters: ['text', 'lcov']
};