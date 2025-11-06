// src/App.test.js
import React from 'react';
import { renderWithProviders, screen } from './test-utils';
import App from './App';
import { useAuth } from './components/auth/AuthContext';

jest.mock('./components/MapComponent', () => () => <div data-testid="map-component" />);
jest.mock('./components/FireControls', () => () => <div data-testid="fire-controls" />);

const createAuthState = (overrides = {}) => ({
  user: null,
  isAuthenticated: false,
  loading: false,
  login: jest.fn(),
  register: jest.fn(),
  logout: jest.fn(),
  forgotPassword: jest.fn(),
  ...overrides
});

beforeEach(() => {
  useAuth.mockReturnValue(createAuthState());
  window.history.pushState({}, '', '/');
});

afterEach(() => {
  useAuth.mockReturnValue(createAuthState());
  window.history.pushState({}, '', '/');
});

test('renders the login page logo text', () => {
  renderWithProviders(<App />);
  const logo = screen.getByRole('heading', { name: /IgnisAI/i });
  expect(logo).toBeInTheDocument();
});

test('renders dashboard components when an authenticated user visits the dashboard', async () => {
  useAuth.mockReturnValue(createAuthState({
    user: { fullName: 'Fire Chief', email: 'chief@example.com' },
    isAuthenticated: true
  }));

  window.history.pushState({}, 'Dashboard', '/dashboard');

  const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

  const { container } = renderWithProviders(<App />);

  const mapHeading = await screen.findByRole('heading', { name: /interactive fire map/i });
  expect(mapHeading).toBeInTheDocument();

  const predictionHeading = screen.getByRole('heading', { name: /fire spread prediction/i });
  expect(predictionHeading).toBeInTheDocument();

  const mapContainer = container.querySelector('.map-container');
  expect(mapContainer).not.toBeNull();
  expect(mapContainer).toBeVisible();

  const predictionSection = container.querySelector('.fire-prediction-section');
  expect(predictionSection).not.toBeNull();
  expect(predictionSection).toBeVisible();

  expect(consoleErrorSpy).not.toHaveBeenCalled();

  consoleErrorSpy.mockRestore();
});
