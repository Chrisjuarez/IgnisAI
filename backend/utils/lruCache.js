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

/** Serialized size of a cached value, in bytes. */
function sizeOf(value) {
  try {
    return Buffer.byteLength(JSON.stringify(value) || '');
  } catch (_) {
    return 0; // Unserializable values are not the ones blowing the budget.
  }
}

class LruTtlCache {
  /**
   * @param {object} opts
   * @param {number} opts.max       Maximum number of entries. Must be > 0.
   * @param {number} opts.ttl       Entry TTL in ms. Use 0 for "no TTL".
   * @param {number} [opts.maxBytes] Budget for the total serialized size of
   *   all entries. Counting entries is a weak bound when their sizes differ
   *   by two orders of magnitude: a city-sized map bootstrap is ~0.4 MB and a
   *   western-CONUS one is tens of MB, so a 200-entry cap permitted gigabytes
   *   on a 512 MB instance.
   */
  constructor({ max, ttl, maxBytes = 0 }) {
    if (!Number.isFinite(max) || max <= 0) {
      throw new TypeError(`LruTtlCache: max must be a positive number, got ${max}`);
    }
    if (!Number.isFinite(ttl) || ttl < 0) {
      throw new TypeError(`LruTtlCache: ttl must be >= 0, got ${ttl}`);
    }
    if (!Number.isFinite(maxBytes) || maxBytes < 0) {
      throw new TypeError(`LruTtlCache: maxBytes must be >= 0, got ${maxBytes}`);
    }
    this.max = max;
    this.ttl = ttl;
    this.maxBytes = maxBytes;
    this._bytes = 0;
    /** @type {Map<string, { value: any, expiresAt: number, bytes: number }>} */
    this._store = new Map();
  }

  get size() {
    return this._store.size;
  }

  /** Total serialized size of everything currently held. */
  get bytes() {
    return this._bytes;
  }

  /**
   * Return the cached value for `key` or null. Promote the entry to MRU
   * when it's a hit; lazily evict if expired.
   */
  get(key) {
    const entry = this._store.get(key);
    if (!entry) return null;
    if (this.ttl > 0 && Date.now() > entry.expiresAt) {
      this._evict(key);
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
    this._evict(key);

    // Measured once per fill rather than per request. Estimating from record
    // counts is guesswork when one perimeter polygon outweighs a thousand
    // hotspots.
    const bytes = this.maxBytes > 0 ? sizeOf(value) : 0;
    const expiresAt = this.ttl > 0 ? Date.now() + this.ttl : Number.POSITIVE_INFINITY;

    this._store.set(key, { value, expiresAt, bytes });
    this._bytes += bytes;
    this._trim();
    return value;
  }

  /** Drop entries, oldest first, until both bounds are satisfied. */
  _trim() {
    while (this._store.size > this.max || (this.maxBytes > 0 && this._bytes > this.maxBytes)) {
      // A lone oversized entry stays: evicting it would empty the cache and
      // the next request would just refetch and re-store the same thing.
      if (this._store.size <= 1) break;
      const oldest = this._store.keys().next().value;
      if (oldest === undefined) break;
      this._evict(oldest);
    }
  }

  _evict(key) {
    const entry = this._store.get(key);
    if (!entry) return false;
    this._bytes -= entry.bytes;
    return this._store.delete(key);
  }

  delete(key) {
    return this._evict(key);
  }

  clear() {
    this._store.clear();
    this._bytes = 0;
  }

  keys() {
    return Array.from(this._store.keys());
  }

  /** Snapshot used by /metrics. */
  stats() {
    return {
      size: this._store.size,
      max: this.max,
      ttl: this.ttl,
      bytes: this._bytes,
      maxBytes: this.maxBytes,
    };
  }
}

module.exports = { LruTtlCache };
