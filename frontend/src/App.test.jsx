// src/App.test.jsx
import React from 'react';
import { renderWithProviders, screen } from './test-utils';
import App from './App';

jest.mock('./components/MapComponent', () => {
  const ReactActual = jest.requireActual('react');
  return ReactActual.forwardRef(() => <div data-testid="map-component" />);
});
jest.mock('./components/FireControls', () => () => <div data-testid="fire-controls" />);

jest.mock('./api', () => ({
  getMapBootstrap: jest.fn(async () => ({
    data: { updatedAt: new Date().toISOString(), incidents: [], alerts: [], layerStatus: {} },
  })),
  getIncident: jest.fn(async () => ({ data: {} })),
  getIncidentUpdates: jest.fn(async () => ({ data: { updates: [] } })),
  getSiteExposure: jest.fn(),
}));

beforeEach(() => {
  window.history.pushState({}, '', '/');
});

test('a visitor without an account goes straight to the map', async () => {
  renderWithProviders(<App />);

  expect(await screen.findByTestId('map-component')).toBeInTheDocument();
  expect(window.location.pathname).toBe('/dashboard');
  expect(screen.getByRole('link', { name: 'Sign in' })).toHaveAttribute('href', '/login');
});

test('the login page is still there for anyone who wants an account', () => {
  window.history.pushState({}, '', '/login');
  renderWithProviders(<App />);

  expect(screen.getByRole('heading', { name: /IgnisAI/i })).toBeInTheDocument();
});
