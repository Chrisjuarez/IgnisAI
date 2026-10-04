// backend/utils/geo.js
//
// Distance from a point to a GeoJSON polygon, in metres, without a geometry
// dependency. Built for matching an incident's reported origin to the
// perimeter mapped around it - distances of tens to hundreds of metres - so it
// projects onto a local tangent plane at the point rather than doing geodesy.
// Over a few kilometres the error is well under a percent, which is far finer
// than the 1 km radius the caller matches within.

'use strict';

const METRES_PER_DEG_LAT = 110540;
const METRES_PER_DEG_LON_AT_EQUATOR = 111320;

function polygonRings(geometry) {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates || []];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates || [];
  return [];
}

/** [west, south, east, north] of a Polygon or MultiPolygon, or null. */
function bboxOf(geometry) {
  let w = Infinity; let s = Infinity; let e = -Infinity; let n = -Infinity;
  for (const polygon of polygonRings(geometry)) {
    for (const ring of polygon) {
      for (const [x, y] of ring) {
        if (x < w) w = x;
        if (x > e) e = x;
        if (y < s) s = y;
        if (y > n) n = y;
      }
    }
  }
  return Number.isFinite(w) ? [w, s, e, n] : null;
}

/** Whether a bbox, grown by `radiusM`, can contain the point at all. */
function bboxWithinRadius(bbox, lon, lat, radiusM) {
  if (!bbox) return false;
  const dLat = radiusM / METRES_PER_DEG_LAT;
  const dLon = radiusM / (METRES_PER_DEG_LON_AT_EQUATOR * Math.cos((lat * Math.PI) / 180));
  return lon >= bbox[0] - dLon && lon <= bbox[2] + dLon && lat >= bbox[1] - dLat && lat <= bbox[3] + dLat;
}

// Even-odd ray cast. Holes fall out naturally: a point inside a hole crosses
// the outer ring and the hole's ring, which cancel.
function insidePolygon(x, y, polygon) {
  let inside = false;
  for (const ring of polygon) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
  }
  return inside;
}

function segmentDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/**
 * Metres from (lon, lat) to the nearest part of a Polygon or MultiPolygon:
 * zero inside it, Infinity for anything that is not a polygon.
 */
function distanceToPolygonMetres(lon, lat, geometry) {
  const kx = METRES_PER_DEG_LON_AT_EQUATOR * Math.cos((lat * Math.PI) / 180);
  const toLocal = ([x, y]) => [(x - lon) * kx, (y - lat) * METRES_PER_DEG_LAT];

  let nearest = Infinity;
  for (const polygon of polygonRings(geometry)) {
    const local = polygon.map((ring) => ring.map(toLocal));
    if (insidePolygon(0, 0, local)) return 0;
    for (const ring of local) {
      for (let i = 1; i < ring.length; i += 1) {
        const d = segmentDistance(0, 0, ring[i - 1][0], ring[i - 1][1], ring[i][0], ring[i][1]);
        if (d < nearest) nearest = d;
      }
    }
  }
  return nearest;
}

module.exports = { bboxOf, bboxWithinRadius, distanceToPolygonMetres };
