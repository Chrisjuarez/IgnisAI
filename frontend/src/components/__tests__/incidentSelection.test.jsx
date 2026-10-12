// The incident panel shows only the selected incident's details, and only the
// newest ones.
import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import AdvancedFireDashboard from '../AdvancedFireDashboard';
import { getIncident } from '../../api';

jest.mock('../MapComponent', () => {
  const ReactActual = jest.requireActual('react');
  return ReactActual.forwardRef((_props, ref) => {
    ReactActual.useImperativeHandle(ref, () => ({
      runPredictionForIncident: jest.fn(async () => ({ frames: [{ lead_hours: 24 }] })),
      flyToIncident: jest.fn(),
      flyToAlert: jest.fn(),
    }));
    return <div data-testid="map" />;
  });
});

jest.mock('../auth/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Crew Lead' } }) }));

const mockActive = {
  id: 'wfigs:bouquet', name: 'BOUQUET', county: 'Los Angeles', state: 'US-CA', status: 'active',
  acres: 1048, lat: 34.5618, lon: -118.4018, predictionEligibility: { eligible: true, reasons: [] },
};
const mockContained = {
  ...mockActive, id: 'wfigs:dome', name: 'DOME', status: 'contained',
  predictionEligibility: { eligible: false, reasons: ['incident is contained'] },
};

jest.mock('../../api', () => ({
  getMapBootstrap: jest.fn(async () => ({
    data: { updatedAt: '2026-10-11T18:00:00Z', incidents: [mockActive, mockContained], alerts: [], layerStatus: {} },
  })),
  getIncident: jest.fn(),
  getIncidentUpdates: jest.fn(async () => ({ data: { updates: [] } })),
  getSiteExposure: jest.fn(),
}));

const NOW = Date.parse('2026-10-11T18:00:00Z');
const minutesAgo = (n) => new Date(NOW - n * 60 * 1000).toISOString();
const detail = (incident, lastDetectionAt) => ({
  data: {
    incident,
    predictionEligibility: incident.predictionEligibility,
    freshness: { lastDetectionAt, perimeterMappedAt: null, nextPasses: [] },
  },
});
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};
const runButton = () => screen.getByRole('button', { name: 'Run Ignis Prediction' });

beforeEach(() => {
  jest.useFakeTimers();
  jest.setSystemTime(NOW);
  jest.clearAllMocks();
});

afterEach(() => {
  jest.useRealTimers();
});

async function openDashboard() {
  render(<AdvancedFireDashboard />);
  await screen.findByRole('button', { name: /BOUQUET/ });
}

test('while the next incident loads, the panel shows none of the previous one\'s details', async () => {
  getIncident
    .mockResolvedValueOnce(detail(mockActive, minutesAgo(21)))
    .mockImplementationOnce(() => deferred().promise);
  await openDashboard();

  fireEvent.click(screen.getByRole('button', { name: /BOUQUET/ }));
  expect(await screen.findByText(/21 min ago/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /DOME/ }));

  expect(screen.queryByText(/21 min ago/)).not.toBeInTheDocument();
  // The contained fire's own eligibility, not the active fire's.
  expect(runButton()).toBeDisabled();
});

test('a failed load leaves the panel without details rather than the previous incident\'s', async () => {
  getIncident
    .mockResolvedValueOnce(detail(mockActive, minutesAgo(21)))
    .mockRejectedValueOnce(new Error('Fire data could not be loaded.'));
  await openDashboard();

  fireEvent.click(screen.getByRole('button', { name: /BOUQUET/ }));
  await screen.findByText(/21 min ago/);
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: /DOME/ }));
  });

  expect(screen.queryByText(/21 min ago/)).not.toBeInTheDocument();
  expect(runButton()).toBeDisabled();
});

test('of two loads for the same incident, the newer one is shown even if the older lands last', async () => {
  const first = deferred();
  getIncident
    .mockImplementationOnce(() => first.promise)                    // on selecting
    .mockResolvedValueOnce(detail(mockActive, minutesAgo(4)));      // when the forecast runs
  await openDashboard();

  fireEvent.click(screen.getByRole('button', { name: /BOUQUET/ }));
  fireEvent.click(runButton());
  expect(await screen.findByText(/4 min ago/)).toBeInTheDocument();

  await act(async () => first.resolve(detail(mockActive, minutesAgo(180))));

  expect(screen.getByText(/4 min ago/)).toBeInTheDocument();
  expect(screen.queryByText(/3 hr ago/)).not.toBeInTheDocument();
});
