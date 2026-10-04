// frontend/src/utils/hotspotAge.js
//
// Hotspots coloured by how long ago a satellite saw them - the convention fire
// maps share. A detection from this morning and one from yesterday afternoon
// otherwise look identical, and only one of them says where the fire is now.
// The map and its legend both read these bands, so they cannot drift apart.

const MS_PER_HOUR = 3_600_000;

// A detection with no usable time is drawn as the oldest band: claiming it is
// fresh would be worse than understating it.
const UNKNOWN_AGE_HOURS = 1e9;

export const HOTSPOT_AGE_BANDS = Object.freeze([
  { maxHours: 6, color: '#dc2626', label: 'Under 6 hours' },
  { maxHours: 12, color: '#f97316', label: '6 to 12 hours' },
  { maxHours: 24, color: '#facc15', label: '12 to 24 hours' },
  { maxHours: Infinity, color: '#78716c', label: 'Over 24 hours' },
]);

/** Hours since a detection, or null when its timestamp cannot be read. */
export function hotspotAgeHours(timestamp, now = Date.now()) {
  if (timestamp == null || timestamp === '') return null;
  const seenAt = new Date(timestamp).getTime();
  if (!Number.isFinite(seenAt)) return null;
  return Math.max(0, (now - seenAt) / MS_PER_HOUR);
}

/** Mapbox colour expression over a feature's `ageHours` property. */
export const HOTSPOT_AGE_COLOR = HOTSPOT_AGE_BANDS.slice(1).reduce(
  (expression, band, index) => [...expression, HOTSPOT_AGE_BANDS[index].maxHours, band.color],
  ['step', ['coalesce', ['get', 'ageHours'], UNKNOWN_AGE_HOURS], HOTSPOT_AGE_BANDS[0].color],
);
