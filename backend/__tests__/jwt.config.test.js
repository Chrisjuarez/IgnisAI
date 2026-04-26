// backend/__tests__/jwt.config.test.js
//
// Verifies the fail-fast behavior of backend/config/jwt.js. The previous
// implementation silently fell back to a public string when JWT_SECRET was
// missing in production, which is the kind of regression we never want back.
//
// We re-require the module on each scenario via jest.isolateModules so the
// loader's require-time side effects (throw / warn / generate) actually run.

'use strict';

const path = require('path');

const JWT_MODULE = path.resolve(__dirname, '..', 'config', 'jwt.js');

function loadJwtConfig() {
  let mod;
  jest.isolateModules(() => {
    // eslint-disable-next-line global-require
    mod = require(JWT_MODULE);
  });
  return mod;
}

describe('config/jwt fail-fast loader', () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...ORIGINAL_ENV };
    delete process.env.JWT_SECRET;
    delete process.env.JWT_EXPIRES_IN;
  });

  afterAll(() => {
    process.env = ORIGINAL_ENV;
  });

  test('production + missing JWT_SECRET throws at require time', () => {
    process.env.NODE_ENV = 'production';
    expect(() => loadJwtConfig()).toThrow(/JWT_SECRET is not set/);
  });

  test('production + short JWT_SECRET throws at require time', () => {
    process.env.NODE_ENV = 'production';
    process.env.JWT_SECRET = 'too-short';
    expect(() => loadJwtConfig()).toThrow(/minimum 32/);
  });

  test('production + valid JWT_SECRET loads without throwing', () => {
    process.env.NODE_ENV = 'production';
    process.env.JWT_SECRET = 'a'.repeat(48);
    const cfg = loadJwtConfig();
    expect(cfg.JWT_SECRET).toHaveLength(48);
    expect(cfg.JWT_EXPIRES_IN).toBe('7d'); // default
  });

  test('test env supplies a deterministic secret without warning', () => {
    process.env.NODE_ENV = 'test';
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const cfg = loadJwtConfig();
    expect(cfg.JWT_SECRET.length).toBeGreaterThanOrEqual(32);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  test('development warns and generates an ephemeral secret', () => {
    process.env.NODE_ENV = 'development';
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const cfg = loadJwtConfig();
    expect(cfg.JWT_SECRET.length).toBeGreaterThanOrEqual(32);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringMatching(/ephemeral secret/i),
    );
    warnSpy.mockRestore();
  });

  test('JWT_EXPIRES_IN env override is respected', () => {
    process.env.NODE_ENV = 'production';
    process.env.JWT_SECRET = 'b'.repeat(40);
    process.env.JWT_EXPIRES_IN = '12h';
    const cfg = loadJwtConfig();
    expect(cfg.JWT_EXPIRES_IN).toBe('12h');
  });

  test('config object is frozen', () => {
    process.env.NODE_ENV = 'test';
    const cfg = loadJwtConfig();
    expect(Object.isFrozen(cfg)).toBe(true);
  });
});
