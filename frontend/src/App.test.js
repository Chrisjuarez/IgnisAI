import React from 'react';
import { render, screen, within } from '@testing-library/react';
import App from './App';

jest.mock('./components/MapComponent', () => () => <div data-testid="map-component" />);
jest.mock('./components/FireControls', () => () => <div data-testid="fire-controls" />);

test('renders the header logo text', () => {
  render(<App />);
  const header = screen.getByRole('banner');
  expect(within(header).getByText(/IgnisAI/i)).toBeInTheDocument();
});
