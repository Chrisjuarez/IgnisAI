module.exports = {
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: ['app.js'],   // expand later as you add tests
  coverageReporters: ['text','lcov'],
};
