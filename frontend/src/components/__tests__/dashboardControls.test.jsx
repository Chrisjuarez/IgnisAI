import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import mapboxgl from 'mapbox-gl';
import FireControls from '../FireControls';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(() => Promise.resolve({ data: { data: [] } })),
  predictFireSpread: jest.fn(() => Promise.resolve({}))
}));

describe('Dashboard controls', () => {
  let consoleErrorSpy;

  beforeEach(() => {
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    expect(consoleErrorSpy).not.toHaveBeenCalled();
    consoleErrorSpy.mockRestore();
  });

  const baseProps = {
    onRefresh: jest.fn(),
    isFetching: false,
    fireCount: 0,
    onChangeBrightness: jest.fn(),
    onChangeConfidence: jest.fn(),
    onChangeMapStyle: jest.fn(),
    mapboxToken: 'test-token',
    onSelectLocation: jest.fn(),
    range: 10,
    onChangeRange: jest.fn(),
    nearbyFires: []
  };

  test('filter dropdowns load and call their change handlers', () => {
    const onChangeBrightness = jest.fn();
    const onChangeConfidence = jest.fn();
    const onChangeMapStyle = jest.fn();

    render(
      <FireControls
        {...baseProps}
        onChangeBrightness={onChangeBrightness}
        onChangeConfidence={onChangeConfidence}
        onChangeMapStyle={onChangeMapStyle}
      />
    );

    const brightnessSelect = screen.getByText(/Brightness Filter/i).nextElementSibling;
    const confidenceSelect = screen.getByText(/Confidence Filter/i).nextElementSibling;
    const styleSelect = screen.getByText(/Map Style/i).nextElementSibling;

    fireEvent.change(brightnessSelect, { target: { value: 'Severe' } });
    fireEvent.change(confidenceSelect, { target: { value: 'High' } });
    fireEvent.change(styleSelect, {
      target: { value: 'mapbox://styles/mapbox/satellite-streets-v12' }
    });

    expect(onChangeBrightness).toHaveBeenCalledWith('Severe');
    expect(onChangeConfidence).toHaveBeenCalledWith('High');
    expect(onChangeMapStyle).toHaveBeenCalledWith('mapbox://styles/mapbox/satellite-streets-v12');
  });

  test('refresh and use-my-location buttons execute their handlers', () => {
    const onRefresh = jest.fn();
    const onSelectLocation = jest.fn();
    const originalGeo = navigator.geolocation;
    const geolocationMock = {
      getCurrentPosition: jest.fn(success =>
        success({ coords: { latitude: 12.34, longitude: 56.78 } })
      )
    };
    Object.defineProperty(window.navigator, 'geolocation', {
      value: geolocationMock,
      configurable: true
    });

    render(
      <FireControls
        {...baseProps}
        onRefresh={onRefresh}
        onSelectLocation={onSelectLocation}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /refresh fire data/i }));
    expect(onRefresh).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: /use my location/i }));
    expect(geolocationMock.getCurrentPosition).toHaveBeenCalledTimes(1);
    expect(onSelectLocation).toHaveBeenCalledWith({ lat: 12.34, lng: 56.78 });

    if (originalGeo) {
      Object.defineProperty(window.navigator, 'geolocation', {
        value: originalGeo,
        configurable: true
      });
    } else {
      delete window.navigator.geolocation;
    }
  });

  test('falls back to approximate IP location when geolocation fails', async () => {
    const onSelectLocation = jest.fn();
    const originalGeo = navigator.geolocation;
    const originalFetch = global.fetch;

    const geolocationMock = {
      getCurrentPosition: jest.fn((_, error) => error(new Error('blocked')))
    };
    Object.defineProperty(window.navigator, 'geolocation', {
      value: geolocationMock,
      configurable: true
    });

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            latitude: 37.7749,
            longitude: -122.4194,
            city: 'San Francisco',
            region: 'California'
          })
      })
    );

    render(
      <FireControls
        {...baseProps}
        onSelectLocation={onSelectLocation}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /use my location/i }));

    await waitFor(() =>
      expect(onSelectLocation).toHaveBeenCalledWith({
        lat: 37.7749,
        lng: -122.4194
      })
    );
    expect(global.fetch).toHaveBeenCalledWith('https://ipapi.co/json/');

    if (originalGeo) {
      Object.defineProperty(window.navigator, 'geolocation', {
        value: originalGeo,
        configurable: true
      });
    } else {
      delete window.navigator.geolocation;
    }

    if (originalFetch) {
      global.fetch = originalFetch;
    } else {
      delete global.fetch;
    }
  });

  test('map zoom controls zoom the map in and out', async () => {
    const MapComponent = require('../MapComponent').default;
    const setIsFetching = jest.fn();
    const onFiresUpdated = jest.fn();
    const onNearbyFiresUpdate = jest.fn();

    render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={onFiresUpdated}
        setIsFetching={setIsFetching}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={onNearbyFiresUpdate}
      />
    );

    const zoomInButton = await screen.findByRole('button', { name: /zoom in/i });
    const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
    const mapInstance = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
    const initialZoom = mapInstance._zoom;

    fireEvent.click(zoomInButton);
    expect(mapInstance.zoomIn).toHaveBeenCalledTimes(1);
    expect(mapInstance._zoom).toBe(initialZoom + 1);

    fireEvent.click(zoomOutButton);
    expect(mapInstance.zoomOut).toHaveBeenCalledTimes(1);
    expect(mapInstance._zoom).toBe(initialZoom);
  });
});
