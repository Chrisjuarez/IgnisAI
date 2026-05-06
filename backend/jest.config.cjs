// backend/jest.config.cjs
module.exports = {
  testEnvironment: 'node',
  coveragePathIgnorePatterns: ['/node_modules/'],
  testMatch: ['**/__tests__/**/*.test.js'],
  collectCoverageFrom: [
    '**/*.js',
    '!**/node_modules/**',
    '!**/coverage/**',
    '!jest.config.cjs'
  ],
  verbose: true,
  testTimeout: 30000, // Increase timeout to 30 seconds
  preset: '@shelf/jest-mongodb'
};
