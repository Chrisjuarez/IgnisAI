// Which fires Ignis can forecast: an active fire a satellite has seen within
// 2 km in the last 24 h. The list marks the active fires it cannot.
import React from 'react';
import { render, screen, within } from '@testing-library/react';
import AdvancedFireDashboard from '../AdvancedFireDashboard';

jest.mock('../MapComponent', () => {
  const ReactActual = jest.requireActual('react');
  return ReactActual.forwardRef((_props, ref) => {
    ReactActual.useImperativeHandle(ref, () => ({
      runPredictionForIncident: jest.fn(),
      flyToIncident: jest.fn(),
      flyToAlert: jest.fn(),
    }));
    return <div data-testid="map" />;
  });
});

jest.mock('../auth/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Crew Lead' } }) }));

const NO_RECENT = "No satellite has detected this fire within 2 km in the last 24 hours, so Ignis can't forecast its spread.";
const fire = (over) => ({
  type: 'wildfire', county: 'Los Angeles', state: 'US-CA', status: 'active', acres: 1048, lat: 34.5618, lon: -118.4018,
  detectedLast24h: true, predictionEligibility: { eligible: true, reasons: [] }, ...over,
});
const mockFires = [
  fire({ id: 'wfigs:bouquet', name: 'BOUQUET' }),
  fire({ id: 'wfigs:mccully', name: 'MCCULLY', detectedLast24h: false,
    predictionEligibility: { eligible: false, reasons: [NO_RECENT] } }),
  fire({ id: 'wfigs:dome', name: 'DOME', status: 'contained', detectedLast24h: false,
    predictionEligibility: { eligible: false, reasons: ['incident is contained'] } }),
  fire({ id: 'wfigs:rx', name: 'SRF RX', type: 'prescribed', detectedLast24h: false,
    predictionEligibility: { eligible: false, reasons: ['record is not an active wildfire', NO_RECENT] } }),
];

jest.mock('../../api', () => ({
  getMapBootstrap: jest.fn(async () => ({
    data: { updatedAt: '2026-10-17T18:00:00Z', incidents: mockFires, alerts: [], layerStatus: {} },
  })),
  getIncident: jest.fn(async () => ({ data: {} })),
  getIncidentUpdates: jest.fn(async () => ({ data: { updates: [] } })),
  getSiteExposure: jest.fn(),
}));

const card = async (name) => screen.findByRole('button', { name: new RegExp(`^${name}`) });

test('marks an active fire no satellite has seen in the last day', async () => {
  render(<AdvancedFireDashboard />);

  expect(within(await card('MCCULLY')).getByText('No detection in 24 h')).toBeInTheDocument();
});

test('leaves forecastable and contained fires unmarked', async () => {
  render(<AdvancedFireDashboard />);

  expect(within(await card('BOUQUET')).queryByText('No detection in 24 h')).not.toBeInTheDocument();
  // A contained fire already says so; it is not waiting on a satellite.
  expect(within(await card('DOME')).queryByText('No detection in 24 h')).not.toBeInTheDocument();
});

test('leaves prescribed burns unmarked', async () => {
  // Never forecast, detected or not, so a satellite tag would mislead.
  render(<AdvancedFireDashboard />);

  expect(within(await card('SRF RX')).queryByText('No detection in 24 h')).not.toBeInTheDocument();
});
