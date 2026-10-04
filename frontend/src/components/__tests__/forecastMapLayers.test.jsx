import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ForecastMapLayers, { FORECAST_CONTEXT_LAYERS } from '../ForecastMapLayers';

function renderLayers(overrides = {}) {
  const props = {
    visibility: { prediction: true, perimeters: true, hotspots: false },
    onToggle: jest.fn(),
    opacity: 1,
    onOpacityChange: jest.fn(),
    ...overrides,
  };
  render(<ForecastMapLayers {...props} />);
  return props;
}

describe('ForecastMapLayers', () => {
  test('lists every layer a forecast is read against, checked to match the map', () => {
    renderLayers();

    FORECAST_CONTEXT_LAYERS.forEach(({ label }) => {
      expect(screen.getByRole('checkbox', { name: label })).toBeInTheDocument();
    });
    expect(screen.getByRole('checkbox', { name: 'Official perimeter' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Satellite hotspots' })).not.toBeChecked();
    // A layer the dashboard has never switched is on, as it is on the map.
    expect(screen.getByRole('checkbox', { name: 'Evacuation zones' })).toBeChecked();
  });

  test('reports which layer was switched', () => {
    const { onToggle } = renderLayers();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Satellite hotspots' }));
    expect(onToggle).toHaveBeenCalledWith('hotspots');
  });

  test('reports opacity as a number', () => {
    const { onOpacityChange } = renderLayers();
    fireEvent.change(screen.getByRole('slider', { name: 'Forecast opacity' }), { target: { value: '0.5' } });
    expect(onOpacityChange).toHaveBeenCalledWith(0.5);
  });

  test('disables opacity while the forecast is hidden', () => {
    renderLayers({ visibility: { prediction: false } });
    expect(screen.getByRole('slider', { name: 'Forecast opacity' })).toBeDisabled();
  });
});
