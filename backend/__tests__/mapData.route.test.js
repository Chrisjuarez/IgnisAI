const request = require('supertest');

jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('../models/Wildfire', () => ({ insertMany: jest.fn(), find: jest.fn() }));

const axios = require('axios');
const app = require('../app');

const FIRMS_CSV = [
  'latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight',
  '34.0522,-118.2437,330.5,0.4,0.4,2026-04-22,1430,N21,VIIRS,high,2.0,328.5,12.3,D',
].join('\n');

function collection(features = []) {
  return { type: 'FeatureCollection', features };
}

const ZONE_RING = [[[-118.45, 34.55], [-118.35, 34.55], [-118.35, 34.6], [-118.45, 34.6], [-118.45, 34.55]]];

// Shaped like the CalOES feed for the Bouquet Fire on 2026-10-03.
const EVACUATION_ZONES = [
  { type: 'Feature', geometry: { type: 'Polygon', coordinates: ZONE_RING },
    properties: { ZONE_ID: 'US-CA-XLA-LAC-E018', STATUS: 'Evacuation Order', COUNTY: 'LOS ANGELES', EDIT_DATE: Date.parse('2026-10-03T23:10:00Z') } },
  { type: 'Feature', geometry: { type: 'Polygon', coordinates: ZONE_RING },
    properties: { ZONE_ID: 'US-CA-XLA-LAC-E031-B', STATUS: 'Evacuation Warning', COUNTY: 'LOS ANGELES', EDIT_DATE: Date.parse('2026-10-03T23:10:00Z') } },
];

describe('GET /api/map/bootstrap', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    axios.get.mockImplementation((url, config = {}) => {
      if (String(url).includes('WFIGS_Incident_Locations_Current') || config.params?.resultRecordCount === 1500) {
        return Promise.resolve({
          data: collection([
            {
              type: 'Feature',
              geometry: { type: 'Point', coordinates: [-118.55, 34.05] },
              properties: {
                OBJECTID: 7,
                IrwinID: 'abc-123',
                IncidentName: 'Palisades Fire',
                IncidentTypeCategory: 'WF',
                IncidentSize: 23448,
                PercentContained: 95,
                POOCounty: 'Los Angeles County',
                POOState: 'CA',
                ModifiedOnDateTime_dt: Date.parse('2026-04-22T12:00:00Z'),
                FireDiscoveryDateTime: Date.parse('2025-01-07T18:30:00Z'),
                ActiveFireCandidate: 1,
              },
            },
          ]),
        });
      }
      if (String(url).includes('WFIGS_Interagency_Perimeters_Current')) {
        return Promise.resolve({
          data: collection([
            {
              type: 'Feature',
              geometry: { type: 'Polygon', coordinates: [[[-118.6, 34], [-118.5, 34], [-118.5, 34.1], [-118.6, 34.1], [-118.6, 34]]] },
              properties: { OBJECTID: 1, poly_IncidentName: 'Palisades Fire' },
            },
          ]),
        });
      }
      if (String(url).includes('CA_Perimeters_NIFC_FIRIS_public_view')) {
        return Promise.resolve({ data: collection([]) });
      }
      if (String(url).includes('firms.modaps.eosdis.nasa.gov')) {
        return Promise.resolve({ data: FIRMS_CSV });
      }
      if (String(url).includes('api.weather.gov')) {
        return Promise.resolve({
          data: collection([
            {
              id: 'nws-alert-1',
              type: 'Feature',
              geometry: { type: 'Polygon', coordinates: [[[-119, 33], [-117, 33], [-117, 35], [-119, 35], [-119, 33]]] },
              properties: {
                event: 'Red Flag Warning',
                headline: 'Red Flag Warning issued by NWS',
                areaDesc: 'Los Angeles County Mountains',
                status: 'Actual',
                effective: '2026-04-22T12:00:00Z',
                expires: '2026-04-23T03:00:00Z',
                sent: '2026-04-22T11:30:00Z',
              },
            },
          ]),
        });
      }
      if (String(url).includes('CA_EVACUATIONS_CalOESHosted_view')) {
        return Promise.resolve({ data: collection(EVACUATION_ZONES) });
      }
      return Promise.resolve({ data: collection([]) });
    });
  });

  it('returns normalized incidents, perimeters, hotspots, alerts, and layer status', async () => {
    const res = await request(app)
      .get('/api/map/bootstrap')
      .query({ bbox: '-119,33,-117,35' })
      .expect(200);

    expect(res.body.incidents).toHaveLength(1);
    expect(res.body.incidents[0]).toMatchObject({
      id: 'wfigs:abc-123',
      name: 'Palisades Fire',
      status: 'active',
      hasPerimeter: true,
      hasHotspots: true,
      hasPrediction: true,
    });
    // Geometry is counted, not shipped. The map fetches /wildfires and
    // /fire-perimeters directly; serialising both at full extent was
    // exhausting the instance.
    expect(res.body.perimeters).toBeUndefined();
    expect(res.body.perimeterCount).toBe(1);
    expect(res.body.hotspots).toBeUndefined();
    expect(res.body.hotspotCount).toBe(1);
    expect(res.body.alerts[0]).toMatchObject({ event: 'Red Flag Warning', sourceNames: ['NWS'] });
    expect(res.body.layerStatus).toMatchObject({
      incidents: { ok: true },
      perimeters: { ok: true },
      hotspots: { ok: true },
      alerts: { ok: true },
    });
  });

  describe('evacuation zones on the bootstrap', () => {
    const { normalizeEvacuationZone, shortZoneId } = require('../routes/mapData')._private;

    it('carries normalised Orders and Warnings and reports the layer healthy', async () => {
      const res = await request(app)
        .get('/api/map/bootstrap')
        .query({ bbox: '-118.6,34.4,-118.2,34.7' })
        .expect(200);

      expect(res.body.evacuations.type).toBe('FeatureCollection');
      expect(res.body.evacuations.features.map(f => f.properties)).toEqual([
        expect.objectContaining({ zone_id: 'LAC-E018', status: 'order', status_label: 'Order' }),
        expect.objectContaining({ zone_id: 'LAC-E031-B', status: 'warning', status_label: 'Warning' }),
      ]);
      expect(res.body.layerStatus.evacuations).toMatchObject({ ok: true, count: 2, partial: false });
    });

    it('asks for generalised geometry so the payload stays small', async () => {
      await request(app).get('/api/map/bootstrap').query({ bbox: '-118.61,34.4,-118.2,34.7' }).expect(200);

      const call = axios.get.mock.calls.find(([url]) => String(url).includes('CA_EVACUATIONS_CalOESHosted_view'));
      // Full-precision zones were 648 KB for one fire; generalised, 33 KB.
      expect(call[1].params).toMatchObject({ maxAllowableOffset: expect.any(Number), geometryPrecision: 5 });
      expect(call[1].params.outFields).not.toBe('*');
    });

    it('keeps every other layer when the evacuation feed is down', async () => {
      const base = axios.get.getMockImplementation();
      axios.get.mockImplementation((url, config) => (
        String(url).includes('CA_EVACUATIONS_CalOESHosted_view')
          ? Promise.reject(new Error('CalOES unavailable'))
          : base(url, config)
      ));

      const res = await request(app)
        .get('/api/map/bootstrap')
        .query({ bbox: '-118.62,34.4,-118.2,34.7' })
        .expect(200);

      expect(res.body.evacuations.features).toEqual([]);
      expect(res.body.layerStatus.evacuations).toMatchObject({ ok: false, error: 'CalOES unavailable' });
      expect(res.body.layerStatus.incidents.ok).toBe(true);
    });

    it('flags a truncated page instead of silently dropping zones', async () => {
      const base = axios.get.getMockImplementation();
      axios.get.mockImplementation((url, config) => (
        String(url).includes('CA_EVACUATIONS_CalOESHosted_view')
          ? Promise.resolve({ data: { ...collection(EVACUATION_ZONES), properties: { exceededTransferLimit: true } } })
          : base(url, config)
      ));

      const res = await request(app)
        .get('/api/map/bootstrap')
        .query({ bbox: '-118.63,34.4,-118.2,34.7' })
        .expect(200);

      expect(res.body.layerStatus.evacuations).toMatchObject({ ok: true, partial: true });
    });

    it.each([
      ['US-CA-XLA-LAC-E018', 'LAC-E018'],
      ['US-CA-XTU-PVL-E044', 'PVL-E044'],
      ['US-CA-XMY-MRY-F015-C', 'MRY-F015-C'],
      ['LOCAL-7', 'LOCAL-7'],
    ])('shortens zone id %s to %s, as county maps print it', (raw, short) => {
      expect(shortZoneId(raw)).toBe(short);
    });

    it('drops zones with no geometry or a status it does not recognise', () => {
      expect(normalizeEvacuationZone({ geometry: null, properties: { STATUS: 'Evacuation Order' } })).toBeNull();
      expect(normalizeEvacuationZone({ geometry: { type: 'Polygon', coordinates: ZONE_RING }, properties: { STATUS: 'Lifted' } })).toBeNull();
    });
  });

  describe('perimeter memory', () => {
    it('asks for generalised perimeter geometry on the bootstrap', async () => {
      // Full-precision perimeters for the western US were 50 MB of GeoJSON per
      // cold bootstrap on a 512 MB instance, and a redeploy could OOM the first
      // request. annotateIncidents reads only names; the polygons do not need
      // full precision here.
      await request(app).get('/api/map/bootstrap').query({ bbox: '-118.7,34.4,-118.2,34.7' }).expect(200);

      ['WFIGS_Interagency_Perimeters_Current', 'CA_Perimeters_NIFC_FIRIS_public_view'].forEach((service) => {
        const call = axios.get.mock.calls.find(([url]) => String(url).includes(service));
        expect(call[1].params).toMatchObject({ maxAllowableOffset: expect.any(Number), geometryPrecision: 5 });
      });
    });

    it('returns only the incident\'s own perimeters on the detail route', async () => {
      const base = axios.get.getMockImplementation();
      // Each perimeter where its fire actually is. Perimeters are also matched by
      // location now, so three sharing one polygon would all sit on the incident.
      const perimeter = (name, [lon, lat]) => ({
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: [[[lon - 0.05, lat - 0.05], [lon + 0.05, lat - 0.05], [lon + 0.05, lat + 0.05], [lon - 0.05, lat + 0.05], [lon - 0.05, lat - 0.05]]] },
        properties: { poly_IncidentName: name },
      });
      axios.get.mockImplementation((url, config) => (
        String(url).includes('WFIGS_Interagency_Perimeters_Current')
          ? Promise.resolve({ data: collection([
            perimeter('Palisades Fire', [-118.55, 34.05]),
            perimeter('Bouquet', [-118.40, 34.56]),
            perimeter('Eaton', [-118.10, 34.19]),
          ]) })
          : base(url, config)
      ));

      const res = await request(app)
        .get(`/api/incidents/${encodeURIComponent('wfigs:abc-123')}`)
        .query({ bbox: '-119.02,33,-117,35' })
        .expect(200);

      // It used to ship every perimeter in the bootstrap - 553 at full extent -
      // on each incident click, for a field no client reads.
      expect(res.body.perimeters.features.map(f => f.properties.poly_IncidentName)).toEqual(['Palisades Fire']);
      expect(res.body.incident).toMatchObject({ name: 'Palisades Fire', hasPerimeter: true });
    });
  });
});

describe('matching perimeters to incidents', () => {
  const { annotateIncidents } = require('../routes/mapData')._private;
  const DISCOVERED = '2026-10-03T21:16:00.000Z';
  const FLIGHT_2109 = Date.parse('2026-10-04T04:09:00Z');

  // Bouquet, as WFIGS reported it after its size was overwritten.
  const bouquet = (over = {}) => ({
    id: 'wfigs:bouquet', name: 'BOUQUET', lat: 34.561835, lon: -118.40183,
    acres: 0.1, createdAt: DISCOVERED, updatedAt: '2026-10-04T03:30:00.000Z', sourceNames: ['WFIGS', 'NIFC'],
    ...over,
  });
  // A square whose nearest edge is `gapM` metres west of the incident.
  const square = (gapM, props) => {
    const east = -118.40183 - gapM / (111320 * Math.cos((34.561835 * Math.PI) / 180));
    return {
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: [[[east - 0.02, 34.55], [east, 34.55], [east, 34.57], [east - 0.02, 34.57], [east - 0.02, 34.55]]] },
      properties: { source: 'FIRIS', mission: 'CA-ANF-BOQUET-N50X', incident_name: null, ...props },
    };
  };
  const annotate = (incident, features) => annotateIncidents([incident], collection(features), [])[0];

  it('links a nameless perimeter mapped just outside the reported origin', () => {
    // The Bouquet case: no incident name on the perimeter, a misspelled mission
    // code, and the origin 112 m outside the burned area.
    const result = annotate(bouquet(), [square(112, { area_acres: 1048.26, poly_DateCurrent: FLIGHT_2109 })]);

    expect(result.hasPerimeter).toBe(true);
    expect(result.acres).toBe(1048);
    expect(result.reportedAcres).toBe(0.1);
    expect(result.acresSource).toEqual({ kind: 'perimeter', provider: 'FIRIS', asOf: '2026-10-04T04:09:00.000Z' });
  });

  it('does not reach past the match radius', () => {
    const result = annotate(bouquet(), [square(4700, { area_acres: 5000, poly_DateCurrent: FLIGHT_2109 })]);
    expect(result.hasPerimeter).toBe(false);
    expect(result.acres).toBe(0.1);
    expect(result.acresSource.kind).toBe('reported');
  });

  it('ignores a perimeter mapped before the fire existed', () => {
    // An earlier fire's perimeter on the same hillside must not lend a new
    // incident its acreage.
    const result = annotate(bouquet(), [square(50, { area_acres: 9000, poly_DateCurrent: Date.parse('2026-08-20T00:00:00Z') })]);
    expect(result.hasPerimeter).toBe(false);
    expect(result.acres).toBe(0.1);
  });

  it('takes the newest measurement when several perimeters match', () => {
    const result = annotate(bouquet(), [
      square(112, { source: 'USFS', area_acres: 709.7, poly_DateCurrent: Date.parse('2026-10-03T23:49:00Z') }),
      square(112, { area_acres: 1048.26, poly_DateCurrent: FLIGHT_2109 }),
      square(112, { source: 'USFS', area_acres: 842.8, poly_DateCurrent: Date.parse('2026-10-04T00:27:00Z') }),
    ]);
    expect(result.acres).toBe(1048);
    expect(result.acresSource.provider).toBe('FIRIS');
  });

  it('keeps a reported size that is already larger than the measurement', () => {
    // Reported sizes include estimates beyond the last flight; a smaller,
    // older perimeter must not drag a correct figure down.
    const result = annotate(bouquet({ acres: 1500 }), [square(112, { area_acres: 1048.26, poly_DateCurrent: FLIGHT_2109 })]);
    expect(result.acres).toBe(1500);
    expect(result.acresSource.kind).toBe('reported');
    expect(result.hasPerimeter).toBe(true);
  });

  it('still links a named perimeter too far away to locate, without borrowing its acreage', () => {
    const named = square(30000, { source: undefined, poly_IncidentName: 'Bouquet', poly_GISAcres: 50000 });
    const result = annotate(bouquet(), [named]);
    expect(result.hasPerimeter).toBe(true);
    expect(result.acres).toBe(0.1);
  });

  it('rounds measured acreage the way the panel displays it', () => {
    const small = annotate(bouquet({ acres: null }), [square(0, { area_acres: 42.37, poly_DateCurrent: FLIGHT_2109 })]);
    expect(small.acres).toBe(42.4);
  });
});

describe('bootstrap response weight', () => {
  const { bootstrapResponse } = require('../routes/mapData')._private || {};

  it('withholds detections from the response but reports how many there were', () => {
    if (!bootstrapResponse) return; // not exported in this build
    const payload = {
      updatedAt: 'now',
      incidents: [{ id: 'a' }],
      perimeters: { type: 'FeatureCollection', features: [] },
      hotspots: [{ latitude: 1 }, { latitude: 2 }, { latitude: 3 }],
      alerts: [],
      layerStatus: {},
    };

    const response = bootstrapResponse(payload);

    expect(response.hotspots).toBeUndefined();
    expect(response.hotspotCount).toBe(3);
    expect(response.perimeters).toBeUndefined();
    expect(response.perimeterCount).toBe(0);
    expect(response.incidents).toEqual([{ id: 'a' }]);
    expect(response.layerStatus).toEqual({});
  });

  it('does not mutate the shared payload that /incidents/:id reads', () => {
    if (!bootstrapResponse) return;
    const payload = {
      hotspots: [{ latitude: 1 }],
      perimeters: { type: 'FeatureCollection', features: [{ id: 'p1' }] },
      incidents: [],
    };

    bootstrapResponse(payload);

    expect(payload.hotspots).toHaveLength(1);
    expect(payload.perimeters.features).toHaveLength(1);
  });
});
