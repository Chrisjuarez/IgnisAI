import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import mapboxgl from 'mapbox-gl';
import FireControls from '../FireControls';
import {
  getWildfireData,
  getWildfireFootprints,
  getFirePerimeters,
  predictFireSpreadMultistep,
} from '../../api';
import {
  prepareMultistepRasterFrames,
  renderPredictionRasterFrame,
  removePredictionOverlays,
} from '../../utils/addPredictionOverlay';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(() => Promise.resolve({ data: { data: [] } })),
  getWildfireFootprints: jest.fn(() => Promise.resolve({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  getFirePerimeters: jest.fn(() => Promise.resolve({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  predictFireSpread: jest.fn(() => Promise.resolve({})),
  predictFireSpreadMultistep: jest.fn(() => Promise.resolve({
    bounds: [-118.6, 34.0, -118.1, 34.4],
    threshold: 0.01,
    step_hours: 6,
    steps: [
      { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1', prob_max: 0.11, prob_mean: 0.03, area_fraction: 0.08 },
      { index: 1, lead_hours: 12, label: '12 hours', image_base64: 'frame-2', prob_max: 0.19, prob_mean: 0.05, area_fraction: 0.11 },
    ]
  }))
}));

jest.mock('../../utils/addPredictionOverlay', () => ({
  addPredictionOverlay: jest.fn(() => Promise.resolve({ bounds: [-118.6, 34.0, -118.1, 34.4] })),
  prepareMultistepRasterFrames: jest.fn(async payload => ({
    bounds: payload.bounds,
    threshold: payload.threshold,
    stepHours: payload.step_hours,
    frames: payload.steps.map(step => ({
      ...step,
      bounds: payload.bounds,
      heatmapUrl: `data:image/png;base64,${step.image_base64}`,
      meta: {
        prob_max: step.prob_max,
        prob_mean: step.prob_mean,
      }
    }))
  })),
  renderPredictionRasterFrame: jest.fn(() => Promise.resolve({ kind: 'raster' })),
  removePredictionOverlays: jest.fn(),
}));

describe('Dashboard controls', () => {
  let consoleErrorSpy;
  const multistepPayload = {
    bounds: [-118.6, 34.0, -118.1, 34.4],
    threshold: 0.01,
    step_hours: 6,
    steps: [
      { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1', prob_max: 0.11, prob_mean: 0.03, area_fraction: 0.08 },
      { index: 1, lead_hours: 12, label: '12 hours', image_base64: 'frame-2', prob_max: 0.19, prob_mean: 0.05, area_fraction: 0.11 },
    ]
  };

  beforeEach(() => {
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.useRealTimers();
    mapboxgl.Popup.mockImplementation(() => ({
      setLngLat: jest.fn().mockReturnThis(),
      setHTML: jest.fn().mockReturnThis(),
      addTo: jest.fn().mockReturnThis(),
      remove: jest.fn(),
    }));
    getWildfireData.mockResolvedValue({ data: { data: [] } });
    getWildfireFootprints.mockResolvedValue({ data: { geojson: { type: 'FeatureCollection', features: [] } } });
    getFirePerimeters.mockResolvedValue({ data: { geojson: { type: 'FeatureCollection', features: [] } } });
    predictFireSpreadMultistep.mockResolvedValue(multistepPayload);
    prepareMultistepRasterFrames.mockImplementation(async payload => ({
      bounds: payload.bounds,
      threshold: payload.threshold,
      stepHours: payload.step_hours,
      frames: payload.steps.map(step => ({
        ...step,
        bounds: payload.bounds,
        heatmapUrl: `data:image/png;base64,${step.image_base64}`,
        meta: {
          prob_max: step.prob_max,
          prob_mean: step.prob_mean,
        }
      }))
    }));
    renderPredictionRasterFrame.mockResolvedValue({ kind: 'raster' });
    removePredictionOverlays.mockImplementation(() => {});
  });

  afterEach(() => {
    jest.useRealTimers();
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

  test('map initializes FIRMS footprint and perimeter layers', async () => {
    const MapComponent = require('../MapComponent').default;

    render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={jest.fn()}
        setIsFetching={jest.fn()}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={jest.fn()}
      />
    );

    const mapInstance = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
    await waitFor(() => {
      expect(mapInstance.getLayer('wildfire-footprints-fill')).toBeTruthy();
      expect(mapInstance.getLayer('fire-perimeters-outline')).toBeTruthy();
    });
    expect(getWildfireFootprints).toHaveBeenCalled();
    expect(getFirePerimeters).toHaveBeenCalled();
  });

  test('forecast panel appears and slider updates the active forecast frame', async () => {
    const MapComponent = require('../MapComponent').default;

    render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={jest.fn()}
        setIsFetching={jest.fn()}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={jest.fn()}
      />
    );

    fireEvent.click(await screen.findByRole('button', { name: /history/i }));
    fireEvent.click(screen.getByRole('button', { name: /camp\/paradise fire/i }));

    expect(await screen.findByTestId('forecast-panel')).toBeInTheDocument();
    expect(screen.getByText('6 hours')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/forecast timeline slider/i), { target: { value: '1' } });

    await waitFor(() => expect(screen.getByText('12 hours')).toBeInTheDocument());
    expect(renderPredictionRasterFrame).toHaveBeenCalled();
  });

  test('forecast playback advances and clear removes the overlay state', async () => {
    jest.useFakeTimers();
    const MapComponent = require('../MapComponent').default;

    render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={jest.fn()}
        setIsFetching={jest.fn()}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={jest.fn()}
      />
    );

    fireEvent.click(await screen.findByRole('button', { name: /history/i }));
    fireEvent.click(screen.getByRole('button', { name: /camp\/paradise fire/i }));
    expect(await screen.findByTestId('forecast-panel')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /play/i }));
    act(() => {
      jest.advanceTimersByTime(1300);
    });

    await waitFor(() => expect(screen.getByText('12 hours')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /clear forecast timeline/i }));
    await waitFor(() => expect(screen.queryByTestId('forecast-panel')).not.toBeInTheDocument());
    expect(removePredictionOverlays).toHaveBeenCalled();

    jest.useRealTimers();
  });

  test('custom historical runs call the multistep api with date', async () => {
    const MapComponent = require('../MapComponent').default;
    const { container } = render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={jest.fn()}
        setIsFetching={jest.fn()}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={jest.fn()}
      />
    );

    fireEvent.click(await screen.findByRole('button', { name: /history/i }));
    fireEvent.change(container.querySelector('input[type="date"]'), { target: { value: '2021-08-14' } });
    fireEvent.click(screen.getByRole('button', { name: /run custom date/i }));

    await waitFor(() => {
      expect(predictFireSpreadMultistep).toHaveBeenCalledWith(expect.objectContaining({
        date: '2021-08-14',
        steps: 6,
      }));
    });
    expect(await screen.findByTestId('forecast-panel')).toBeInTheDocument();
    expect(prepareMultistepRasterFrames).toHaveBeenCalled();
  });
});
