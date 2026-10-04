const { bboxOf, bboxWithinRadius, distanceToPolygonMetres } = require('../utils/geo');

// About 1.11 km a side at 34.5 deg N.
const SQUARE = { type: 'Polygon', coordinates: [[[-118.41, 34.55], [-118.40, 34.55], [-118.40, 34.56], [-118.41, 34.56], [-118.41, 34.55]]] };
const METRES_PER_DEG_LON = 111320 * Math.cos((34.555 * Math.PI) / 180);

describe('distanceToPolygonMetres', () => {
  it('is zero inside the polygon', () => {
    expect(distanceToPolygonMetres(-118.405, 34.555, SQUARE)).toBe(0);
  });

  it('measures to the nearest edge from outside', () => {
    // 0.001 deg east of the east edge.
    expect(distanceToPolygonMetres(-118.399, 34.555, SQUARE)).toBeCloseTo(0.001 * METRES_PER_DEG_LON, 0);
  });

  it('measures to a corner when that is nearest', () => {
    const d = distanceToPolygonMetres(-118.399, 34.561, SQUARE);
    expect(d).toBeCloseTo(Math.hypot(0.001 * METRES_PER_DEG_LON, 0.001 * 110540), 0);
  });

  it('treats a point inside a hole as outside, measured to the hole edge', () => {
    const withHole = {
      type: 'Polygon',
      coordinates: [
        SQUARE.coordinates[0],
        [[-118.407, 34.553], [-118.403, 34.553], [-118.403, 34.557], [-118.407, 34.557], [-118.407, 34.553]],
      ],
    };
    const d = distanceToPolygonMetres(-118.405, 34.555, withHole);
    expect(d).toBeGreaterThan(0);
    // Nearest hole edge is east/west: a degree of longitude is the shorter one here.
    expect(d).toBeCloseTo(0.002 * METRES_PER_DEG_LON, -1);
  });

  it('takes the nearest part of a MultiPolygon', () => {
    const far = { type: 'Polygon', coordinates: [[[-117, 34], [-116.9, 34], [-116.9, 34.1], [-117, 34.1], [-117, 34]]] };
    const multi = { type: 'MultiPolygon', coordinates: [far.coordinates, SQUARE.coordinates] };
    expect(distanceToPolygonMetres(-118.405, 34.555, multi)).toBe(0);
  });

  it('is Infinity for anything that is not a polygon', () => {
    expect(distanceToPolygonMetres(-118.4, 34.5, { type: 'Point', coordinates: [-118.4, 34.5] })).toBe(Infinity);
    expect(distanceToPolygonMetres(-118.4, 34.5, null)).toBe(Infinity);
  });
});

describe('bbox prefilter', () => {
  it('computes bounds across every ring and part', () => {
    expect(bboxOf(SQUARE)).toEqual([-118.41, 34.55, -118.40, 34.56]);
    expect(bboxOf(null)).toBeNull();
  });

  it('admits points within the radius and rejects ones beyond it', () => {
    const bbox = bboxOf(SQUARE);
    expect(bboxWithinRadius(bbox, -118.399, 34.555, 1000)).toBe(true);   // ~92 m east
    expect(bboxWithinRadius(bbox, -118.38, 34.555, 1000)).toBe(false);   // ~1.8 km east
  });
});
