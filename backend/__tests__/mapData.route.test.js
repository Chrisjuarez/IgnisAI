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
    expect(res.body.perimeters.features).toHaveLength(1);
    expect(res.body.hotspots).toHaveLength(1);
    expect(res.body.alerts[0]).toMatchObject({ event: 'Red Flag Warning', sourceNames: ['NWS'] });
    expect(res.body.layerStatus).toMatchObject({
      incidents: { ok: true },
      perimeters: { ok: true },
      hotspots: { ok: true },
      alerts: { ok: true },
    });
  });
});
