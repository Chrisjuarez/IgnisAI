// A forecast run that has timed out, been replaced or outlived its map draws
// nothing when its response finally arrives.
import React, { createRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import MapComponent from '../MapComponent';
import { predictFireSpreadMultistep } from '../../api';
import { renderPredictionRasterFrame } from '../../utils/addPredictionOverlay';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(async () => ({ data: { data: [] } })),
  getWildfireFootprints: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  getFirePerimeters: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  predictFireSpreadMultistep: jest.fn(),
}));

// The renderer honors `isCurrent` the way the real one does once the map is ready.
jest.mock('../../utils/addPredictionOverlay', () => ({
  prepareMultistepRasterFrames: jest.fn(async (payload) => ({
    bounds: payload.bounds,
    scene: null,
    frames: payload.steps.map((step) => ({ ...step, bounds: payload.bounds, meta: {} })),
  })),
  renderPredictionRasterFrame: jest.fn(async (_map, _frame, opts = {}) => (
    opts.isCurrent && !opts.isCurrent() ? null : { kind: 'raster' }
  )),
  renderPredictionScene: jest.fn(),
  removePredictionOverlays: jest.fn(),
  removePredictionRaster: jest.fn(),
  removePredictionScene: jest.fn(),
  setPredictionRasterOpacity: jest.fn(),
}));

const FORECAST_TIMEOUT_MS = 180_000;
const BOUQUET = { id: 'wfigs:bouquet', name: 'BOUQUET', lat: 34.5618, lon: -118.4018, status: 'active' };
const DOME = { id: 'wfigs:dome', name: 'DOME', lat: 37.79, lon: -119.6, status: 'active' };
const forecastFor = (incident) => ({
  bounds: [incident.lon - 0.1, incident.lat - 0.1, incident.lon + 0.1, incident.lat + 0.1],
  steps: [{ index: 0, lead_hours: 24, label: 'Day 1', image_base64: 'frame' }],
});
const deferred = () => {
  let resolve;
  const promise = new Promise((res) => { resolve = res; });
  return { promise, resolve };
};

async function renderMap() {
  const ref = createRef();
  const view = render(<MapComponent ref={ref} watchShell incidents={[BOUQUET, DOME]} />);
  await act(async () => { jest.advanceTimersByTime(50); });  // map load and style
  return { ref, view };
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
});

afterEach(() => {
  jest.useRealTimers();
});

test('a forecast that arrives after its timeout draws nothing', async () => {
  const late = deferred();
  predictFireSpreadMultistep.mockImplementationOnce(() => late.promise);
  const { ref } = await renderMap();

  let run;
  await act(async () => { run = ref.current.runPredictionForIncident(BOUQUET); });
  await act(async () => { jest.advanceTimersByTime(FORECAST_TIMEOUT_MS); });
  await act(async () => { await run; });
  expect(screen.getByText(/Forecast timed out/)).toBeInTheDocument();

  await act(async () => { late.resolve(forecastFor(BOUQUET)); });

  expect(renderPredictionRasterFrame).not.toHaveBeenCalled();
  expect(screen.queryByText('BOUQUET Ignis Advisory')).not.toBeInTheDocument();
});

test('an older run that lands after a newer one does not replace it', async () => {
  const late = deferred();
  predictFireSpreadMultistep
    .mockImplementationOnce(() => late.promise)
    .mockResolvedValueOnce(forecastFor(DOME));
  const { ref } = await renderMap();

  let first;
  await act(async () => { first = ref.current.runPredictionForIncident(BOUQUET); });
  await act(async () => { jest.advanceTimersByTime(FORECAST_TIMEOUT_MS); });
  await act(async () => { await first; });
  await act(async () => { await ref.current.runPredictionForIncident(DOME); });
  expect(screen.getByText('DOME Ignis Advisory')).toBeInTheDocument();
  const drawsBefore = renderPredictionRasterFrame.mock.calls.length;

  await act(async () => { late.resolve(forecastFor(BOUQUET)); });

  expect(screen.getByText('DOME Ignis Advisory')).toBeInTheDocument();
  expect(renderPredictionRasterFrame.mock.calls.length).toBe(drawsBefore);
  const drawnBounds = renderPredictionRasterFrame.mock.calls.map(([, frame]) => frame.bounds);
  expect(drawnBounds).not.toContainEqual(forecastFor(BOUQUET).bounds);
});

test('a run still loading when the map unmounts draws nothing', async () => {
  const late = deferred();
  predictFireSpreadMultistep.mockImplementationOnce(() => late.promise);
  const { ref, view } = await renderMap();

  let run;
  await act(async () => { run = ref.current.runPredictionForIncident(BOUQUET); });
  view.unmount();
  await act(async () => { late.resolve(forecastFor(BOUQUET)); await run; });

  expect(renderPredictionRasterFrame).not.toHaveBeenCalled();
});
