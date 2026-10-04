// The dashboard's side of "new satellite data since your forecast": what it
// records when a forecast runs, when it looks again, and what it shows then.
import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AdvancedFireDashboard from '../AdvancedFireDashboard';
import { getIncident } from '../../api';
import { DETECTION_LATENCY_MS } from '../../hooks/useRecheckAfterNextPass';

const mockRunPrediction = jest.fn(async () => ({ frames: [{ lead_hours: 24 }] }));

jest.mock('../MapComponent', () => {
  const ReactActual = jest.requireActual('react');
  return ReactActual.forwardRef((_props, ref) => {
    ReactActual.useImperativeHandle(ref, () => ({
      runPredictionForIncident: mockRunPrediction,
      flyToIncident: jest.fn(),
      flyToAlert: jest.fn(),
    }));
    return <div data-testid="map" />;
  });
});

jest.mock('../auth/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Crew Lead' } }) }));

const mockBouquet = {
  id: 'wfigs:bouquet', name: 'BOUQUET', county: 'Los Angeles', state: 'US-CA', status: 'active',
  acres: 1048, lat: 34.561835, lon: -118.40183, hasPrediction: true,
  predictionEligibility: { eligible: true, reasons: [] },
};
const mockDome = { ...mockBouquet, id: 'wfigs:dome', name: 'DOME', county: 'Mariposa' };

jest.mock('../../api', () => ({
  getMapBootstrap: jest.fn(async () => ({
    data: { updatedAt: '2026-10-04T20:50:00Z', incidents: [mockBouquet, mockDome], alerts: [], layerStatus: {} },
  })),
  getIncident: jest.fn(),
  getIncidentUpdates: jest.fn(async () => ({ data: { updates: [] } })),
  getSiteExposure: jest.fn(),
}));

const NOW = Date.parse('2026-10-04T20:54:00Z');
const NEXT_PASS = '2026-10-04T21:29:00.000Z';
const detail = (lastDetectionAt, nextPassAt = NEXT_PASS) => ({
  data: {
    incident: mockBouquet,
    freshness: { lastDetectionAt, perimeterMappedAt: null, nextPasses: [{ satellite: 'NOAA-20', instrument: 'VIIRS', at: nextPassAt }] },
  },
});

beforeEach(() => {
  jest.useFakeTimers();
  jest.setSystemTime(NOW);
  jest.clearAllMocks();
});

afterEach(() => {
  jest.useRealTimers();
});

async function selectIncident(name) {
  render(<AdvancedFireDashboard />);
  fireEvent.click(await screen.findByRole('button', { name: new RegExp(name) }));
}

test('offers a re-run once the pass after a forecast brings a new detection', async () => {
  getIncident
    .mockResolvedValueOnce(detail('2026-10-04T20:33:00.000Z'))                                  // selected
    .mockResolvedValueOnce(detail('2026-10-04T20:33:00.000Z'))                                  // forecast run
    .mockResolvedValueOnce(detail('2026-10-04T21:29:00.000Z', '2026-10-04T22:48:00.000Z'));     // after the pass

  await selectIncident('BOUQUET');
  fireEvent.click(await screen.findByRole('button', { name: 'Run Ignis Prediction' }));
  await waitFor(() => expect(mockRunPrediction).toHaveBeenCalledTimes(1));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();

  await act(async () => {
    jest.advanceTimersByTime(35 * 60 * 1000 + DETECTION_LATENCY_MS);
  });

  expect(await screen.findByRole('status')).toHaveTextContent('New satellite detection since your forecast');
  expect(getIncident).toHaveBeenCalledTimes(3);
});

test('a pass that sees nothing new leaves the forecast alone', async () => {
  getIncident.mockResolvedValue(detail('2026-10-04T20:33:00.000Z'));

  await selectIncident('BOUQUET');
  fireEvent.click(await screen.findByRole('button', { name: 'Run Ignis Prediction' }));
  await waitFor(() => expect(mockRunPrediction).toHaveBeenCalledTimes(1));
  await act(async () => {
    jest.advanceTimersByTime(35 * 60 * 1000 + DETECTION_LATENCY_MS);
  });

  await waitFor(() => expect(getIncident).toHaveBeenCalledTimes(3));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

test('a detail that arrives after the user has moved on is not shown under the new incident', async () => {
  let resolveBouquet;
  getIncident
    .mockImplementationOnce(() => new Promise((resolve) => { resolveBouquet = resolve; }))
    .mockResolvedValueOnce({ data: { incident: mockDome, freshness: { lastDetectionAt: null, perimeterMappedAt: null, nextPasses: [] } } });

  await selectIncident('BOUQUET');
  fireEvent.click(screen.getByRole('button', { name: /DOME/ }));
  await screen.findByText('None in the last 2 days');

  await act(async () => resolveBouquet(detail('2026-10-04T20:33:00.000Z')));

  expect(screen.getByText('None in the last 2 days')).toBeInTheDocument();
});
