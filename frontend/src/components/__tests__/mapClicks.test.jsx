// The map's own layers, and what a click on them does.
import React from 'react';
import { act, render, waitFor } from '@testing-library/react';
import mapboxgl from 'mapbox-gl';
import MapComponent from '../MapComponent';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(async () => ({ data: { data: [] } })),
  getWildfireFootprints: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  getFirePerimeters: jest.fn(async () => ({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  predictFireSpreadMultistep: jest.fn(),
}));

const BOUQUET = { id: 'wfigs:bouquet', name: 'BOUQUET', lat: 34.5618, lon: -118.4018, status: 'active' };
const CLICK = { point: { x: 10, y: 10 }, lngLat: { lng: BOUQUET.lon, lat: BOUQUET.lat } };
const hotspot = (layerId) => ({
  layer: { id: layerId },
  properties: { latitude: BOUQUET.lat, longitude: BOUQUET.lon, brightnessCat: 'High', confidencePct: 90 },
  geometry: { type: 'Point', coordinates: [BOUQUET.lon, BOUQUET.lat] },
});

async function renderMap(props = {}) {
  render(<MapComponent watchShell incidents={[BOUQUET]} {...props} />);
  const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
  await waitFor(() => expect(map.getLayer('ignis-incidents-layer')).toBeTruthy());
  mapboxgl.Popup.mockClear();
  return map;
}

beforeEach(() => {
  global.fetch = jest.fn(async () => ({ json: async () => ({ features: [] }) }));
});

afterEach(() => {
  delete global.fetch;
});

test('an incident drawn over a hotspot is selected, and no hotspot popup opens', async () => {
  const onIncidentSelect = jest.fn();
  const map = await renderMap({ onIncidentSelect });
  map.queryRenderedFeatures.mockReturnValue([
    { layer: { id: 'ignis-incidents-layer' }, properties: { id: BOUQUET.id } },
    hotspot('wildfires-layer'),
  ]);

  await act(async () => map.trigger('click', CLICK));

  expect(onIncidentSelect).toHaveBeenCalledTimes(1);
  expect(onIncidentSelect).toHaveBeenCalledWith(expect.objectContaining({ id: BOUQUET.id }));
  expect(mapboxgl.Popup).not.toHaveBeenCalled();
});

test('a hotspot drawn over its own footprint opens one popup', async () => {
  const map = await renderMap();
  map.queryRenderedFeatures.mockReturnValue([hotspot('wildfires-layer'), hotspot('wildfire-footprints-fill')]);

  await act(async () => map.trigger('click', CLICK));

  await waitFor(() => expect(mapboxgl.Popup).toHaveBeenCalledTimes(1));
});

test('a click on bare map does nothing', async () => {
  const onIncidentSelect = jest.fn();
  const map = await renderMap({ onIncidentSelect });

  await act(async () => map.trigger('click', CLICK));

  expect(onIncidentSelect).not.toHaveBeenCalled();
  expect(mapboxgl.Popup).not.toHaveBeenCalled();
});

test('keeps its layers when the load event and the style both set them up', async () => {
  const map = await renderMap();
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });

  ['ignis-incidents-layer', 'wildfires-layer', 'fire-perimeters-fill', 'evacuation-zones-fill', 'nws-alerts-fill']
    .forEach((id) => expect(map.getLayer(id)).toBeTruthy());
});
