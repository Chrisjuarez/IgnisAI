jest.mock('axios', () => ({ get: jest.fn() }));

const axios = require('axios');
const satellite = require('satellite.js');
const { nextPasses, SATELLITES, _private } = require('../utils/satellitePasses');

// Orbital elements as CelesTrak published them on 2026-10-04.
const ELEMENTS = {
  43013: ['1 43013U 17073A   26277.53648533  .00000027  00000+0  33946-4 0  9997',
    '2 43013  98.7849 216.0558 0000747  30.8694 329.2526 14.19528931459980'],
  54234: ['1 54234U 22150A   26277.56751025  .00000028  00000+0  33813-4 0  9994',
    '2 54234  98.7130 214.5415 0000822 215.0525 145.0597 14.19551732202061'],
  27424: ['1 27424U 02022A   26277.61259523  .00000445  00000+0  98257-4 0  9999',
    '2 27424  98.4428 249.1614 0001098  62.5549 355.7231 14.62221960299269'],
  25994: ['1 25994U 99068A   26277.59761024  .00000169  00000+0  43371-4 0  9996',
    '2 25994  97.9336 322.7421 0001225 212.5603 266.2513 14.61169515425586'],
};
const BOUQUET = { lat: 34.561835, lon: -118.40183 };
const MINUTE = 60 * 1000;

function serveElements() {
  axios.get.mockImplementation(async (_url, { params }) => {
    const [line1, line2] = ELEMENTS[params.CATNR];
    return { data: `SATELLITE ${params.CATNR}\r\n${line1}\r\n${line2}\r\n` };
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  _private.forgetOrbits();
});

describe('overheadTimes', () => {
  // Every pass over Bouquet on 2026-10-03/04 that FIRMS reported detections
  // from, at the acquisition time FIRMS stamped on them.
  const DETECTED = [
    ['NOAA-21', '2026-10-03T09:32:00Z'], ['NOAA-20', '2026-10-03T10:28:00Z'],
    ['NOAA-20', '2026-10-03T21:48:00Z'], ['Aqua', '2026-10-03T23:48:00Z'],
    ['Terra', '2026-10-04T04:04:00Z'], ['NOAA-21', '2026-10-04T09:13:00Z'],
    ['NOAA-20', '2026-10-04T10:11:00Z'], ['NOAA-21', '2026-10-04T20:33:00Z'],
  ];

  it('predicts every pass that produced a detection to within two minutes', () => {
    const observer = _private.observerAt(BOUQUET.lat, BOUQUET.lon);
    const from = Date.parse('2026-10-03T06:00:00Z');
    const until = Date.parse('2026-10-04T22:00:00Z');

    const missed = DETECTED.filter(([name, detectedAt]) => {
      const sat = SATELLITES.find((s) => s.name === name);
      const satrec = satellite.twoline2satrec(...ELEMENTS[sat.noradId]);
      return !_private.overheadTimes(satrec, observer, sat.minElevationDeg, from, until)
        .some((t) => Math.abs(t.getTime() - Date.parse(detectedAt)) <= 2 * MINUTE);
    });

    expect(missed).toEqual([]);
  });
});

describe('nextPasses', () => {
  const REQUEST = Date.parse('2026-10-04T20:54:00Z'); // 1:54 PM PDT

  it('lists the next passes, soonest first', async () => {
    serveElements();

    const passes = await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST });

    expect(passes).toHaveLength(3);
    // NOAA-20 at 2:29 PM PDT is the next look after the 1:33 PM NOAA-21 pass.
    expect(passes[0]).toMatchObject({ satellite: 'NOAA-20', instrument: 'VIIRS' });
    expect(Math.abs(Date.parse(passes[0].at) - Date.parse('2026-10-04T21:29:00Z'))).toBeLessThanOrEqual(2 * MINUTE);
    expect(passes.map((p) => p.at)).toEqual([...passes.map((p) => p.at)].sort());
  });

  it('does not offer a pass that already peaked as the next one', async () => {
    serveElements();
    // Two minutes after NOAA-20's 2:29 PM peak, while it is still overhead.
    const justAfterPeak = Date.parse('2026-10-04T21:31:00Z');

    const [next] = await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: justAfterPeak });

    expect(next).toMatchObject({ satellite: 'Aqua', instrument: 'MODIS' });
    expect(Math.abs(Date.parse(next.at) - Date.parse('2026-10-04T22:48:00Z'))).toBeLessThanOrEqual(2 * MINUTE);
  });

  it('fetches orbits once and reuses them', async () => {
    serveElements();

    await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST });
    await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST + 60 * MINUTE });

    expect(axios.get).toHaveBeenCalledTimes(SATELLITES.length);
  });

  it('keeps using the last orbits when a refresh fails', async () => {
    serveElements();
    await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST });
    axios.get.mockRejectedValue(new Error('CelesTrak down'));

    const passes = await nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST + 13 * 60 * MINUTE });

    expect(passes).toHaveLength(3);
  });

  it('returns no passes, rather than failing, when orbits cannot be had at all', async () => {
    axios.get.mockRejectedValue(new Error('CelesTrak down'));
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(nextPasses(BOUQUET.lat, BOUQUET.lon, { from: REQUEST })).resolves.toEqual([]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
