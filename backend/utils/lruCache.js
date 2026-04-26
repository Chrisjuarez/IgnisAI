// backend/utils/lruCache.js
//
// Tiny dependency-free LRU + TTL cache.
//
// Why we have this:
//   Multiple routes (mapData, predictFireSpread, ...) keep an in-memory
//   cache to absorb upstream FIRMS/WFIGS/tilesvc latency. Until now they
//   were unbounded `Map`s, which on a Render free instance with 512 MB
//   RAM and a long uptime can OOM the process — every distinct bbox /
//   threshold combination becomes a permanent entry.
//
// Semantics:
//   * `get(key)` returns the stored value or null. Expired entries are
//     evicted lazily on access.
//   * `set(key, value)` (re)inserts and bumps the entry to MRU. When the
//     map exceeds `max` entries the oldest entry is evicted.
//   * Inserting the same key twice replaces the value AND refreshes the
//     TTL — that is the desired behaviour for "we just refetched".
//   * `keys() / size / clear()` are exposed for /metrics and tests.
//
// Insertion-order property of `Map` (ES2015+) is the LRU primitive: by
// re-inserting on access we keep the most-recently-used keys at the tail
// and evict the head when over capacity. This matches lru-cache@7's
// behaviour without the dependency.

'use strict';

class LruTtlCache {
  /**
   * @param {object} opts
   * @param {number} opts.max  Maximum number of entries. Must be > 0.
   * @param {number} opts.ttl  Entry TTL in ms. Use 0 for "no TTL".
   */
  constructor({ max, ttl }) {
    if (!Number.isFinite(max) || max <= 0) {
      throw new TypeError(`LruTtlCache: max must be a positive number, got ${max}`);
    }
    if (!Number.isFinite(ttl) || ttl < 0) {
      throw new TypeError(`LruTtlCache: ttl must be >= 0, got ${ttl}`);
    }
    this.max = max;
    this.ttl = ttl;
    /** @type {Map<string, { value: any, expiresAt: number }>} */
    this._store = new Map();
  }

  get size() {
    return this._store.size;
  }

  /**
   * Return the cached value for `key` or null. Promote the entry to MRU
   * when it's a hit; lazily evict if expired.
   */
  get(key) {
    const entry = this._store.get(key);
    if (!entry) return null;
    if (this.ttl > 0 && Date.now() > entry.expiresAt) {
      this._store.delete(key);
      return null;
    }
    // Re-insert to bump MRU position.
    this._store.delete(key);
    this._store.set(key, entry);
    return entry.value;
  }

  /**
   * Store `value` under `key`. Evicts the oldest entry if at capacity.
   * Returns the value (so callers can `return setCached(...)`).
   */
  set(key, value) {
    if (this._store.has(key)) {
      this._store.delete(key);
    } else if (this._store.size >= this.max) {
      // Evict the oldest entry — `keys()` returns insertion-order so the
      // first key is the LRU.
      const oldest = this._store.keys().next().value;
      if (oldest !== undefined) this._store.delete(oldest);
    }
    const expiresAt = this.ttl > 0 ? Date.now() + this.ttl : Number.POSITIVE_INFINITY;
    this._store.set(key, { value, expiresAt });
    return value;
  }

  delete(key) {
    return this._store.delete(key);
  }

  clear() {
    this._store.clear();
  }

  keys() {
    return Array.from(this._store.keys());
  }

  /** Snapshot used by /metrics: { size, max, ttl, hitRate? } */
  stats() {
    return { size: this._store.size, max: this.max, ttl: this.ttl };
  }
}

module.exports = { LruTtlCache };
