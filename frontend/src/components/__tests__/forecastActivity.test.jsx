// A fire's forecast says how many satellite detections it was built on. Fires
// with few detections in the day before stopped more often than the bands
// suggest, so a quiet fire is named as one - as an observation, not a forecast.
import React, { createRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import MapComponent from '../MapComponent';
import { predictFireSpreadMultistep } from '../../api';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(async () => ({ data: { data: [] } })),
  getWildfireFootprints: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  getFirePerimeters: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  predictFireSpreadMultistep: jest.fn(),
}));

jest.mock('../../utils/addPredictionOverlay', () => ({
  prepareMultistepRasterFrames: jest.fn(async (payload) => ({
    bounds: payload.bounds,
    scene: null,
    frames: payload.steps.map((step) => ({ ...step, bounds: payload.bounds, meta: {} })),
  })),
  renderPredictionRasterFrame: jest.fn(async () => ({ kind: 'raster' })),
  renderPredictionScene: jest.fn(),
  removePredictionOverlays: jest.fn(),
  removePredictionRaster: jest.fn(),
  removePredictionScene: jest.fn(),
  setPredictionRasterOpacity: jest.fn(),
}));

const fire = (detectionsLast24h) => ({
  id: 'wfigs:norte', name: 'NORTE', lat: 37.5, lon: -121.4, status: 'active', detectionsLast24h,
});
const forecast = {
  bounds: [-121.5, 37.4, -121.3, 37.6],
  steps: [{ index: 0, lead_hours: 24, label: 'Day 1', image_base64: 'frame' }],
};

async function forecastFor(incident) {
  const ref = createRef();
  render(<MapComponent ref={ref} watchShell incidents={[incident]} />);
  await act(async () => { jest.advanceTimersByTime(50); });  // map load and style
  await act(async () => { await ref.current.runPredictionForIncident(incident); });
  return ref;
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  predictFireSpreadMultistep.mockResolvedValue(forecast);
});

afterEach(() => {
  jest.useRealTimers();
});

test('a quiet fire is named as one, with its count', async () => {
  await forecastFor(fire(4));

  const note = screen.getByRole('note', { name: 'Fire activity' });
  expect(note).toHaveTextContent('Quiet fire');
  expect(note).toHaveTextContent('4 satellite detections at this fire in the last 24 h');
});

test('a busy fire shows its count alone', async () => {
  await forecastFor(fire(117));

  const note = screen.getByRole('note', { name: 'Fire activity' });
  expect(note).toHaveTextContent('117 satellite detections at this fire in the last 24 h');
  expect(note).not.toHaveTextContent('Quiet');
});

test('one detection reads in the singular', async () => {
  await forecastFor(fire(1));

  expect(screen.getByRole('note', { name: 'Fire activity' }))
    .toHaveTextContent('1 satellite detection at this fire in the last 24 h');
});

test('a forecast without a count says nothing about activity', async () => {
  await forecastFor(fire(undefined));

  expect(screen.getByText('NORTE Ignis Advisory')).toBeInTheDocument();
  expect(screen.queryByRole('note', { name: 'Fire activity' })).not.toBeInTheDocument();
});

test('clearing the forecast clears its count', async () => {
  await forecastFor(fire(4));

  act(() => { screen.getByRole('button', { name: 'Clear forecast timeline' }).click(); });

  expect(screen.queryByRole('note', { name: 'Fire activity' })).not.toBeInTheDocument();
});
