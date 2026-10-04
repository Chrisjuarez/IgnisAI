import { HOTSPOT_AGE_BANDS, HOTSPOT_AGE_COLOR, hotspotAgeHours } from '../hotspotAge';

const NOW = Date.parse('2026-10-04T16:00:00Z');

describe('hotspotAgeHours', () => {
  test('measures hours since the detection', () => {
    expect(hotspotAgeHours('2026-10-04T13:00:00Z', NOW)).toBe(3);
    expect(hotspotAgeHours(new Date('2026-10-03T16:00:00Z'), NOW)).toBe(24);
  });

  test('never reports a negative age for a clock a little ahead', () => {
    expect(hotspotAgeHours('2026-10-04T16:05:00Z', NOW)).toBe(0);
  });

  test('returns null when there is no usable time', () => {
    [null, undefined, '', 'not a date'].forEach((value) => {
      expect(hotspotAgeHours(value, NOW)).toBeNull();
    });
  });
});

describe('HOTSPOT_AGE_COLOR', () => {
  test('steps through the legend bands in order', () => {
    expect(HOTSPOT_AGE_COLOR).toEqual([
      'step', ['coalesce', ['get', 'ageHours'], expect.any(Number)],
      '#dc2626',
      6, '#f97316',
      12, '#facc15',
      24, '#78716c',
    ]);
    expect(HOTSPOT_AGE_COLOR.filter((part) => typeof part === 'string' && part.startsWith('#')))
      .toEqual(HOTSPOT_AGE_BANDS.map((band) => band.color));
  });

  test('draws a detection of unknown age in the oldest colour', () => {
    const [, [, , unknownAge]] = HOTSPOT_AGE_COLOR;
    expect(unknownAge).toBeGreaterThan(24);
  });
});
