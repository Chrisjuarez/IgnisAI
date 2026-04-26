// backend/__tests__/lruCache.test.js
'use strict';

const { LruTtlCache } = require('../utils/lruCache');

describe('LruTtlCache', () => {
  test('rejects non-positive max', () => {
    expect(() => new LruTtlCache({ max: 0, ttl: 1000 })).toThrow();
    expect(() => new LruTtlCache({ max: -1, ttl: 1000 })).toThrow();
    expect(() => new LruTtlCache({ max: NaN, ttl: 1000 })).toThrow();
  });

  test('rejects negative ttl', () => {
    expect(() => new LruTtlCache({ max: 5, ttl: -1 })).toThrow();
  });

  test('get returns null for missing keys', () => {
    const c = new LruTtlCache({ max: 3, ttl: 1000 });
    expect(c.get('missing')).toBeNull();
  });

  test('set then get returns the stored value', () => {
    const c = new LruTtlCache({ max: 3, ttl: 1000 });
    c.set('a', 1);
    expect(c.get('a')).toBe(1);
  });

  test('evicts the oldest entry once capacity is exceeded', () => {
    const c = new LruTtlCache({ max: 2, ttl: 1000 });
    c.set('a', 1);
    c.set('b', 2);
    c.set('c', 3); // should evict 'a'
    expect(c.get('a')).toBeNull();
    expect(c.get('b')).toBe(2);
    expect(c.get('c')).toBe(3);
    expect(c.size).toBe(2);
  });

  test('get bumps an entry to MRU and protects it from eviction', () => {
    const c = new LruTtlCache({ max: 2, ttl: 1000 });
    c.set('a', 1);
    c.set('b', 2);
    // Access 'a' so 'b' becomes the LRU.
    expect(c.get('a')).toBe(1);
    c.set('c', 3); // should evict 'b' (now LRU)
    expect(c.get('b')).toBeNull();
    expect(c.get('a')).toBe(1);
    expect(c.get('c')).toBe(3);
  });

  test('expired entries are evicted lazily on access', () => {
    jest.useFakeTimers();
    const c = new LruTtlCache({ max: 5, ttl: 100 });
    c.set('a', 'stale');
    jest.advanceTimersByTime(101);
    expect(c.get('a')).toBeNull();
    expect(c.size).toBe(0);
    jest.useRealTimers();
  });

  test('re-setting a key refreshes the TTL', () => {
    jest.useFakeTimers();
    const c = new LruTtlCache({ max: 5, ttl: 100 });
    c.set('a', 1);
    jest.advanceTimersByTime(80);
    c.set('a', 2);
    jest.advanceTimersByTime(80); // 160ms total since first set, 80 since second
    expect(c.get('a')).toBe(2);
    jest.useRealTimers();
  });

  test('ttl=0 disables expiration', () => {
    jest.useFakeTimers();
    const c = new LruTtlCache({ max: 5, ttl: 0 });
    c.set('a', 1);
    jest.advanceTimersByTime(10_000_000);
    expect(c.get('a')).toBe(1);
    jest.useRealTimers();
  });

  test('clear empties the cache', () => {
    const c = new LruTtlCache({ max: 5, ttl: 1000 });
    c.set('a', 1);
    c.set('b', 2);
    c.clear();
    expect(c.size).toBe(0);
    expect(c.get('a')).toBeNull();
  });

  test('stats reflect current state', () => {
    const c = new LruTtlCache({ max: 4, ttl: 500 });
    c.set('a', 1);
    expect(c.stats()).toEqual({ size: 1, max: 4, ttl: 500 });
  });
});
