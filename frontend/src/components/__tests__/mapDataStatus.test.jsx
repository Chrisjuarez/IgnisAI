// How the dashboard says how current its data is, and what Refresh does.
import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AdvancedFireDashboard from '../AdvancedFireDashboard';
import { getMapBootstrap } from '../../api';

const mockRefreshWildfires = jest.fn();

jest.mock('../MapComponent', () => {
  const ReactActual = jest.requireActual('react');
  return ReactActual.forwardRef((_props, ref) => {
    ReactActual.useImperativeHandle(ref, () => ({
      refreshWildfires: mockRefreshWildfires,
      flyToIncident: jest.fn(),
      flyToAlert: jest.fn(),
    }));
    return <div data-testid="map" />;
  });
});

jest.mock('../auth/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Crew Lead' } }) }));

jest.mock('../../api', () => ({
  getMapBootstrap: jest.fn(),
  getIncident: jest.fn(async () => ({ data: {} })),
  getIncidentUpdates: jest.fn(async () => ({ data: { updates: [] } })),
  getSiteExposure: jest.fn(),
}));

const NOW = Date.parse('2026-10-11T18:00:00Z');
const BOUQUET = {
  id: 'wfigs:bouquet', name: 'BOUQUET', county: 'Los Angeles', state: 'US-CA', status: 'active',
  acres: 1048, lat: 34.5618, lon: -118.4018, predictionEligibility: { eligible: true, reasons: [] },
};
const bootstrap = ({ incidents = [BOUQUET], layerStatus = {} } = {}) => ({
  data: { updatedAt: new Date(NOW).toISOString(), incidents, alerts: [], layerStatus },
});

beforeEach(() => {
  jest.clearAllMocks();
});

test('a failed refresh keeps the incidents on screen and says the update failed', async () => {
  getMapBootstrap
    .mockResolvedValueOnce(bootstrap())
    .mockRejectedValueOnce(new Error('Fire data could not be loaded. Try again shortly.'));
  render(<AdvancedFireDashboard />);
  await screen.findByRole('button', { name: /BOUQUET/ });

  fireEvent.click(screen.getByRole('button', { name: /^Updated/ }));

  expect(await screen.findByRole('button', { name: 'Update failed · Retry' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /BOUQUET/ })).toBeInTheDocument();
});

test('both refresh controls reload the incidents and the map', async () => {
  getMapBootstrap.mockResolvedValue(bootstrap());
  render(<AdvancedFireDashboard />);
  await screen.findByRole('button', { name: /BOUQUET/ });

  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => expect(getMapBootstrap).toHaveBeenCalledTimes(2));
  expect(mockRefreshWildfires).toHaveBeenCalledTimes(1);

  fireEvent.click(await screen.findByRole('button', { name: /^Updated/ }));
  await waitFor(() => expect(getMapBootstrap).toHaveBeenCalledTimes(3));
  expect(mockRefreshWildfires).toHaveBeenCalledTimes(2);
});

test('the update time keeps counting while the page is open', async () => {
  jest.useFakeTimers();
  jest.setSystemTime(NOW);
  try {
    getMapBootstrap.mockResolvedValue(bootstrap());
    render(<AdvancedFireDashboard />);
    expect(await screen.findByRole('button', { name: 'Updated just now' })).toBeInTheDocument();

    await act(async () => {
      jest.advanceTimersByTime(5 * 60 * 1000);
    });

    expect(screen.getByRole('button', { name: 'Updated 5 min ago' })).toBeInTheDocument();
  } finally {
    jest.useRealTimers();
  }
});

test('an incident feed that is down says so instead of "no incidents"', async () => {
  getMapBootstrap.mockResolvedValue(bootstrap({ incidents: [], layerStatus: { incidents: { ok: false, error: 'WFIGS unavailable' } } }));
  render(<AdvancedFireDashboard />);

  expect(await screen.findByText('The incidents feed is not responding')).toBeInTheDocument();
});
