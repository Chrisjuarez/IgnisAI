// src/App.test.js
import React from 'react';
import { renderWithProviders, screen } from './test-utils';
import App from './App';

jest.mock('./components/MapComponent', () => () => <div data-testid="map-component" />);
jest.mock('./components/FireControls', () => () => <div data-testid="fire-controls" />);

test('renders the login page logo text', () => {
  renderWithProviders(<App />);
  // The login page has an <h1> with "IgnisAI" text
  const logo = screen.getByRole('heading', { name: /IgnisAI/i });
  expect(logo).toBeInTheDocument();
});