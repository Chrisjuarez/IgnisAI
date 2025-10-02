// src/App.test.js
import React from 'react';
import { renderWithProviders, screen, within } from './test-utils';
import App from './App';

jest.mock('./components/MapComponent', () => () => <div data-testid="map-component" />);
jest.mock('./components/FireControls', () => () => <div data-testid="fire-controls" />);

test('renders the header logo text', () => {
  renderWithProviders(<App />);
  const header = screen.getByRole('banner');
  expect(within(header).getByText(/IgnisAI/i)).toBeInTheDocument();
});