// frontend/src/components/ForecastMapLayers.jsx
//
// What can be seen under a forecast, switched from the forecast panel itself.
// Each entry drives the same visibility as its switch in the Layers drawer, so
// the two can never disagree about whether a layer is on.

import React from 'react';

export const FORECAST_CONTEXT_LAYERS = Object.freeze([
  { id: 'prediction', label: 'Forecast' },
  { id: 'perimeters', label: 'Official perimeter' },
  { id: 'hotspots', label: 'Satellite hotspots' },
  { id: 'evacuations', label: 'Evacuation zones' },
  {
    id: 'forecastStart',
    label: "Model's starting fire",
    hint: 'The hotspots the forecast grew from. It can differ from the official perimeter.',
  },
]);

export const MIN_FORECAST_OPACITY = 0.15;

export default function ForecastMapLayers({ visibility, onToggle, opacity, onOpacityChange }) {
  const forecastShown = visibility.prediction !== false;

  return (
    <fieldset className="forecast-map-layers">
      <legend>Show on map</legend>
      <div className="forecast-map-layers__list">
        {FORECAST_CONTEXT_LAYERS.map(({ id, label, hint }) => (
          <label key={id} className="forecast-map-layers__item" title={hint}>
            <input
              type="checkbox"
              checked={visibility[id] !== false}
              onChange={() => onToggle(id)}
            />
            {label}
          </label>
        ))}
      </div>
      <label className="forecast-map-layers__opacity">
        <span>Forecast opacity</span>
        <input
          type="range"
          min={MIN_FORECAST_OPACITY}
          max="1"
          step="0.05"
          value={opacity}
          disabled={!forecastShown}
          aria-label="Forecast opacity"
          onChange={(event) => onOpacityChange(Number(event.target.value))}
        />
      </label>
    </fieldset>
  );
}
