// backend/jest.unit.cjs
//
// Lightweight Jest config for unit-only tests that do NOT require a
// MongoDB instance. Use this instead of the default jest.config.cjs
// (which loads the @shelf/jest-mongodb preset) when you only want to
// exercise pure-logic modules: config loaders, utilities, validators.
//
// Why a second config?
//   The default preset spins up an in-process mongod on every run.
//   That's the right default for the route/integration tests, but it
//   adds 5–15 s of cold-start overhead and downloads a binary on first
//   use. Unit tests don't need any of that.
//
// Run via:
//   npm run test:unit
//   npm run test:unit -- --testPathPatterns='lruCache'
//
// To run the full suite (incl. Mongo-backed integration tests):
//   npm test
//
module.exports = {
  testEnvironment: 'node',
  rootDir: __dirname,
  // Only the pure-unit suites. Add patterns here as you write more
  // dependency-free tests; keep route/integration tests on the default
  // jest.config.cjs which has the Mongo preset.
  testMatch: [
    '<rootDir>/__tests__/jwt.config.test.js',
    '<rootDir>/__tests__/lruCache.test.js',
  ],
  verbose: true,
  testTimeout: 15000,
};
