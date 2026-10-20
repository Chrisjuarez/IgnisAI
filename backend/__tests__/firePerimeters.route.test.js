const request = require('supertest');

jest.mock('axios', () => ({ get: jest.fn() }));

const axios = require('axios');
const app = require('../app');

describe('GET /api/fire-perimeters', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns an empty FeatureCollection when authoritative sources are unavailable', async () => {
    axios.get.mockRejectedValue(new Error('ArcGIS unavailable'));

    const res = await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-119,33,-117,35', incidentName: 'Palisades' })
      .expect(200);

    expect(res.body).toMatchObject({
      geojson: { type: 'FeatureCollection', features: [] },
      count: 0,
      partial: true,
    });
  });

  it('marks the result partial when one source answers with an ArcGIS error', async () => {
    axios.get
      .mockResolvedValueOnce({ data: { type: 'FeatureCollection', features: [] } })
      .mockResolvedValueOnce({ data: { error: { code: 498, message: 'Invalid token' } } });

    const res = await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-119,33,-117,35' })
      .expect(200);

    expect(res.body.partial).toBe(true);
  });

  it('merges available perimeter FeatureCollections', async () => {
    axios.get
      .mockResolvedValueOnce({
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: { OBJECTID: 1, poly_IncidentName: 'Palisades' },
              geometry: { type: 'Polygon', coordinates: [[[-118.6, 34], [-118.5, 34], [-118.5, 34.1], [-118.6, 34.1], [-118.6, 34]]] },
            },
          ],
        },
      })
      .mockRejectedValueOnce(new Error('secondary unavailable'));

    const res = await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-119,33,-117,35', incidentName: 'Palisades' })
      .expect(200);

    expect(res.body.count).toBe(1);
    expect(res.body.partial).toBe(true);
    expect(res.body.geojson.features[0].properties.poly_IncidentName).toBe('Palisades');
  });

  it('asks every source for simplified geometry so a wide viewport stays small', async () => {
    axios.get.mockResolvedValue({ data: { type: 'FeatureCollection', features: [] } });

    await request(app)
      .get('/api/fire-perimeters')
      .query({ bbox: '-125.1,31,-101.8,49.5' })
      .expect(200);

    expect(axios.get).toHaveBeenCalledTimes(2);
    for (const [, config] of axios.get.mock.calls) {
      expect(config.params).toMatchObject({ maxAllowableOffset: 0.0001, geometryPrecision: 6 });
    }
  });

  // FIRIS keeps every flight's perimeter. Drawn together, the Dome Fire's
  // daily outlines stacked into a solid mass that hid its current shape.
  describe('draws each fire as it was last mapped', () => {
    const day = (n) => Date.parse('2026-09-16T20:00:00Z') + n * 24 * 60 * 60 * 1000;

    function square(lon, lat, size, properties) {
      return {
        type: 'Feature',
        properties,
        geometry: { type: 'Polygon', coordinates: [[[lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat]]] },
      };
    }

    function flight(OBJECTID, mappedAt, props = {}, size = 0.02) {
      return square(-119.62, 37.57, size, {
        OBJECTID, source: 'USFS', mission: 'CA-YNP-DOME-N874EB', incident_name: 'DOME', poly_DateCurrent: mappedAt, ...props,
      });
    }

    function serve({ wfigs = [], firis = [] }) {
      axios.get.mockImplementation(async (url) => ({
        data: { type: 'FeatureCollection', features: String(url).includes('WFIGS') ? wfigs : firis },
      }));
    }

    async function drawn(query = {}) {
      const res = await request(app).get('/api/fire-perimeters').query(query).expect(200);
      return res.body.geojson.features.map((feature) => feature.properties.OBJECTID).sort((a, b) => a - b);
    }

    it('keeps only the newer of two perimeters of one fire mapped a day apart', async () => {
      serve({ firis: [flight(1, day(0), {}, 0.01), flight(2, day(1))] });

      expect(await drawn()).toEqual([2]);
    });

    it('keeps two different fires', async () => {
      serve({
        firis: [
          flight(1, day(0)),
          square(-118.4, 34.49, 0.02, { OBJECTID: 2, source: 'FIRIS', mission: 'CA-ANF-BOQUET-N50X', incident_name: 'BOUQUET', poly_DateCurrent: day(1) }),
        ],
      });

      expect(await drawn()).toEqual([1, 2]);
    });

    it('keeps fires that share a name but not a place', async () => {
      serve({ firis: [flight(1, day(0)), square(-121.5, 39.5, 0.02, { OBJECTID: 2, incident_name: 'Dome', poly_DateCurrent: day(1) })] });

      expect(await drawn()).toEqual([1, 2]);
    });

    it('keeps fires that share a name and a place but not an IRWIN ID', async () => {
      serve({
        wfigs: [
          square(-119.62, 37.57, 0.02, { OBJECTID: 60001, poly_IncidentName: 'Dome', attr_IrwinID: '{A}', poly_PolygonDateTime: day(0) }),
          square(-119.61, 37.58, 0.02, { OBJECTID: 60002, poly_IncidentName: 'Dome', attr_IrwinID: '{B}', poly_PolygonDateTime: day(1) }),
        ],
      });

      expect(await drawn()).toEqual([60001, 60002]);
    });

    it('drops FIRIS flights older than the WFIGS perimeter of the same fire', async () => {
      serve({
        wfigs: [square(-119.63, 37.56, 0.1, {
          OBJECTID: 60965, poly_IncidentName: 'Dome', attr_IrwinID: '{AACFD673}', poly_PolygonDateTime: day(21), poly_DateCurrent: day(25),
        })],
        firis: [flight(1, day(17)), flight(2, day(18))],
      });

      expect(await drawn()).toEqual([60965]);
    });

    // poly_DateCurrent on a WFIGS perimeter is when its record was last
    // edited; poly_PolygonDateTime is when the shape was mapped.
    it('dates a WFIGS perimeter by when it was mapped, not when its record was edited', async () => {
      serve({
        wfigs: [square(-119.63, 37.56, 0.1, {
          OBJECTID: 60965, poly_IncidentName: 'Dome', attr_IrwinID: '{AACFD673}', poly_PolygonDateTime: day(21), poly_DateCurrent: day(25),
        })],
        firis: [flight(1, day(23))],
      });

      expect(await drawn()).toEqual([1]);
    });

    // FIRIS republishes WFIGS perimeters as NIFC rows, at the same time but
    // without the IRWIN ID: the Dome Fire's arrived twice.
    it('drops the FIRIS copy of a WFIGS perimeter', async () => {
      serve({
        wfigs: [square(-119.63, 37.56, 0.1, {
          OBJECTID: 60965, poly_IncidentName: 'Dome', attr_IrwinID: '{AACFD673}', poly_PolygonDateTime: day(21), poly_DateCurrent: day(25),
        })],
        firis: [square(-119.63, 37.56, 0.1, { OBJECTID: 1471, source: 'NIFC', mission: '2026-CAYNP-000100', incident_name: 'Dome', poly_DateCurrent: day(21) })],
      });

      expect(await drawn()).toEqual([60965]);
    });

    it('keeps every perimeter one flight mapped of a fire', async () => {
      const henderson = { source: 'USFS', mission: 'CA-PMQ-HENDERSON-N5852K', incident_name: 'HENDERSON' };
      serve({
        firis: [
          square(-120.0, 39.0, 0.03, { OBJECTID: 1, ...henderson, poly_DateCurrent: day(0) }),
          square(-120.0, 39.0, 0.04, { OBJECTID: 2, ...henderson, poly_DateCurrent: day(1) }),
          square(-119.965, 39.035, 0.002, { OBJECTID: 3, ...henderson, poly_DateCurrent: day(1) }),
        ],
      });

      expect(await drawn()).toEqual([2, 3]);
    });

    it('tells a nameless FIRIS perimeter\'s fire by its mission code', async () => {
      const unnamed = { source: 'FIRIS', incident_name: null };
      serve({
        firis: [
          square(-118.4, 34.49, 0.01, { OBJECTID: 1, ...unnamed, mission: 'CA-ANF-BOQUET-N50X', poly_DateCurrent: day(0) }),
          square(-118.4, 34.49, 0.02, { OBJECTID: 2, ...unnamed, mission: 'CA-ANF-BOQUET-N42Z', poly_DateCurrent: day(1) }),
          square(-119.98, 39.54, 0.1, { OBJECTID: 3, ...unnamed, mission: 'NV-RNOX-HAWK-N57B', poly_DateCurrent: day(0) }),
          square(-119.98, 39.54, 0.12, { OBJECTID: 4, ...unnamed, mission: 'NV-RNOX-HAWK-N57B', poly_DateCurrent: day(1) }),
        ],
      });

      expect(await drawn()).toEqual([2, 4]);
    });

    it('draws the newest perimeter within a requested window', async () => {
      serve({ firis: [flight(1, day(0)), flight(2, day(1)), flight(3, day(10))] });

      expect(await drawn({ from: new Date(day(0)).toISOString(), to: new Date(day(8)).toISOString() })).toEqual([2]);
    });
  });
});
