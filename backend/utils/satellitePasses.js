// backend/utils/satellitePasses.js
//
// When the satellites behind FIRMS next look at a point.
//
// Hotspots arrive in bursts - a few passes around 1-3 AM and 1-3 PM local - so
// between them the freshest detection can be half a day old. The satellites
// are polar orbiters on published orbits, which makes the next look
// predictable to the minute. Checked against Bouquet, 2026-10-03/04: every
// pass that produced a detection is predicted within a minute.
//
// A pass is not a promise of a detection. Smoke, cloud or a small cool fire
// can be missed; four of Bouquet's predicted passes saw nothing.

'use strict';

const axios = require('axios');
const satellite = require('satellite.js');

// The satellites FIRMS takes its NRT products from, with the elevation above
// which a point lies inside the instrument's swath. VIIRS sweeps about
// 3,000 km and MODIS about 2,300 km, so MODIS has to be higher in the sky.
const SATELLITES = Object.freeze([
  { name: 'NOAA-20', noradId: 43013, instrument: 'VIIRS', minElevationDeg: 20 },
  { name: 'NOAA-21', noradId: 54234, instrument: 'VIIRS', minElevationDeg: 20 },
  { name: 'Aqua', noradId: 27424, instrument: 'MODIS', minElevationDeg: 30 },
  { name: 'Terra', noradId: 25994, instrument: 'MODIS', minElevationDeg: 30 },
]);

const CELESTRAK_GP_URL = process.env.CELESTRAK_GP_URL || 'https://celestrak.org/NORAD/elements/gp.php';
// Orbital elements drift slowly. Twice a day keeps predictions to the minute,
// and a set that failed to refresh stays usable for days.
const ELEMENTS_TTL_MS = 12 * 60 * 60 * 1000;
const STEP_MS = 30 * 1000;
const LOOKAHEAD_HOURS = 24;
// A scan that starts during a pass sees its first sample as the peak, wrong by
// however long ago the real one was. Starting early finds the true peak, and a
// pass whose peak is already behind `from` is over, not next. Passes inside a
// swath last well under this.
const PASS_LEAD_MS = 15 * 60 * 1000;

let elementsCache = { fetchedAt: 0, satellites: null };
let elementsRequest = null;

async function fetchSatellite(sat) {
  const { data } = await axios.get(CELESTRAK_GP_URL, {
    params: { CATNR: sat.noradId, FORMAT: 'TLE' },
    responseType: 'text',
    timeout: 15000,
  });
  const [, line1, line2] = String(data).trim().split(/\r?\n/).map((line) => line.trim());
  return { ...sat, satrec: satellite.twoline2satrec(line1, line2) };
}

/** Every satellite with its orbit, fetched at most twice a day. */
async function currentSatellites(now) {
  if (elementsCache.satellites && now - elementsCache.fetchedAt < ELEMENTS_TTL_MS) {
    return elementsCache.satellites;
  }
  elementsRequest = elementsRequest || Promise.all(SATELLITES.map(fetchSatellite))
    .then((satellites) => {
      elementsCache = { fetchedAt: Date.now(), satellites };
      return satellites;
    })
    .finally(() => { elementsRequest = null; });
  try {
    return await elementsRequest;
  } catch (err) {
    if (elementsCache.satellites) return elementsCache.satellites;
    throw err;
  }
}

function elevationDeg(satrec, observer, date) {
  const { position } = satellite.propagate(satrec, date);
  if (!position) return -90;
  const look = satellite.ecfToLookAngles(observer, satellite.eciToEcf(position, satellite.gstime(date)));
  return satellite.radiansToDegrees(look.elevation);
}

/**
 * When a satellite is highest over the point while the point is in its swath,
 * between two epoch milliseconds. Those are the times FIRMS stamps its
 * detections with.
 */
function overheadTimes(satrec, observer, minElevationDeg, from, until) {
  const times = [];
  let peak = null;
  for (let t = from; t <= until; t += STEP_MS) {
    const elevation = elevationDeg(satrec, observer, new Date(t));
    if (elevation >= minElevationDeg && (!peak || elevation > peak.elevation)) {
      peak = { t, elevation };
    } else if (elevation < minElevationDeg && peak) {
      times.push(new Date(peak.t));
      peak = null;
    }
  }
  return times;
}

function observerAt(lat, lon) {
  return { latitude: satellite.degreesToRadians(lat), longitude: satellite.degreesToRadians(lon), height: 0 };
}

/**
 * The next satellite looks at a point, soonest first, as
 * [{ satellite, instrument, at }]. Empty rather than an error when orbits
 * cannot be had: not knowing the next pass must not cost an incident its
 * detail.
 */
async function nextPasses(lat, lon, { from = Date.now(), hours = LOOKAHEAD_HOURS, limit = 3 } = {}) {
  let satellites;
  try {
    satellites = await currentSatellites(from);
  } catch (err) {
    console.warn(`satellite orbits unavailable: ${err.message}`);
    return [];
  }
  const observer = observerAt(lat, lon);
  const until = from + hours * 60 * 60 * 1000;
  return satellites
    .flatMap(({ name, instrument, minElevationDeg, satrec }) => overheadTimes(satrec, observer, minElevationDeg, from - PASS_LEAD_MS, until)
      .filter((at) => at.getTime() >= from)
      .map((at) => ({ satellite: name, instrument, at: at.toISOString() })))
    .sort((a, b) => a.at.localeCompare(b.at))
    .slice(0, limit);
}

module.exports = { nextPasses, SATELLITES };
module.exports._private = {
  overheadTimes,
  observerAt,
  forgetOrbits: () => { elementsCache = { fetchedAt: 0, satellites: null }; },
};
