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
  setPredictionRasterOpacity,
} from '../../utils/addPredictionOverlay';
import { HOTSPOT_AGE_COLOR } from '../../utils/hotspotAge';

jest.mock('../../api', () => ({
  getWildfireData: jest.fn(() => Promise.resolve({ data: { data: [] } })),
  getWildfireFootprints: jest.fn(() => Promise.resolve({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  getFirePerimeters: jest.fn(() => Promise.resolve({ data: { geojson: { type: 'FeatureCollection', features: [] } } })),
  predictFireSpread: jest.fn(() => Promise.resolve({})),
  predictFireSpreadMultistep: jest.fn(() => Promise.resolve({
    bounds: [-118.6, 34.0, -118.1, 34.4],
    threshold: 0.85,
    display_floor: 0.02,
    step_hours: 6,
    scene: {
      ignition: { type: 'Feature', geometry: { type: 'Point', coordinates: [-118.55, 34.07] }, properties: {} },
      observed: { type: 'FeatureCollection', features: [] },
      forecast: {
        type: 'FeatureCollection',
        features: [
          { type: 'Feature', properties: { day: 1, lead_hours: 6, color: '#bd0026' },
            geometry: { type: 'Polygon', coordinates: [[[-118.6, 34.0], [-118.5, 34.0], [-118.5, 34.1], [-118.6, 34.1], [-118.6, 34.0]]] } },
        ],
      },
    },
    steps: [
      { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1', prob_max: 0.11, prob_mean: 0.03, area_fraction: 0.00, display_area_fraction: 0.08, display_floor: 0.02 },
      { index: 1, lead_hours: 12, label: '12 hours', image_base64: 'frame-2', prob_max: 0.19, prob_mean: 0.05, area_fraction: 0.00, display_area_fraction: 0.11, display_floor: 0.02 },
    ]
  }))
}));

jest.mock('../../utils/addPredictionOverlay', () => ({
  addPredictionOverlay: jest.fn(() => Promise.resolve({ bounds: [-118.6, 34.0, -118.1, 34.4] })),
  prepareMultistepRasterFrames: jest.fn(async payload => ({
    bounds: payload.bounds,
    scene: payload.scene || null,
    burnedArea: payload.burned_area || null,
    threshold: payload.threshold,
    stepHours: payload.step_hours,
    frames: payload.steps.map(step => ({
      ...step,
      bounds: payload.bounds,
      heatmapUrl: `data:image/png;base64,${step.image_base64}`,
      meta: {
        threshold: payload.threshold,
        display_floor: step.display_floor ?? payload.display_floor,
        prob_max: step.prob_max,
        prob_mean: step.prob_mean,
        display_area_fraction: step.display_area_fraction,
      }
    }))
  })),
  renderPredictionRasterFrame: jest.fn(() => Promise.resolve({ kind: 'raster' })),
  renderPredictionScene: jest.fn(() => Promise.resolve({ kind: 'scene', bands: 0 })),
  removePredictionOverlays: jest.fn(),
  removePredictionRaster: jest.fn(),
  removePredictionScene: jest.fn(),
  setPredictionRasterOpacity: jest.fn(),
}));

describe('Dashboard controls', () => {
  let consoleErrorSpy;
  const scene = {
    ignition: { type: 'Feature', geometry: { type: 'Point', coordinates: [-118.55, 34.07] }, properties: {} },
    observed: { type: 'FeatureCollection', features: [] },
    forecast: {
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', properties: { day: 1, lead_hours: 6, color: '#bd0026' },
          geometry: { type: 'Polygon', coordinates: [[[-118.6, 34.0], [-118.5, 34.0], [-118.5, 34.1], [-118.6, 34.1], [-118.6, 34.0]]] } },
      ],
    },
  };
  const multistepPayload = {
    bounds: [-118.6, 34.0, -118.1, 34.4],
    threshold: 0.85,
    display_floor: 0.02,
    step_hours: 6,
    scene,
    steps: [
      { index: 0, lead_hours: 6, label: '6 hours', image_base64: 'frame-1', prob_max: 0.11, prob_mean: 0.03, area_fraction: 0.00, display_area_fraction: 0.08, display_floor: 0.02 },
      { index: 1, lead_hours: 12, label: '12 hours', image_base64: 'frame-2', prob_max: 0.19, prob_mean: 0.05, area_fraction: 0.00, display_area_fraction: 0.11, display_floor: 0.02 },
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
      // The component reads this to decide whether the band view has anything
      // to draw. Omitting it here silently routed every forecast test down the
      // heatmap branch, which is why no test caught the band view not
      // rendering. The burned area is passed through for the same reason.
      scene: payload.scene || null,
      burnedArea: payload.burned_area || null,
      threshold: payload.threshold,
      stepHours: payload.step_hours,
      frames: payload.steps.map(step => ({
        ...step,
        bounds: payload.bounds,
        heatmapUrl: `data:image/png;base64,${step.image_base64}`,
        meta: {
          threshold: payload.threshold,
          display_floor: step.display_floor ?? payload.display_floor,
          prob_max: step.prob_max,
          prob_mean: step.prob_mean,
          display_area_fraction: step.display_area_fraction,
        }
      }))
    }));
    renderPredictionRasterFrame.mockResolvedValue({ kind: 'raster' });
    removePredictionOverlays.mockImplementation(() => {});
  });

  afterEach(() => {
    jest.useRealTimers();
    const unexpectedErrors = consoleErrorSpy.mock.calls.filter(([message]) =>
      !String(message).includes('not wrapped in act')
    );
    expect(unexpectedErrors).toEqual([]);
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
      expect(mapInstance.getLayer('observed-fire-cells-fill')).toBeTruthy();
      expect(mapInstance.getLayer('wildfire-footprints-fill')).toBeTruthy();
      expect(mapInstance.getLayer('fire-perimeters-outline')).toBeTruthy();
    });
    expect(getWildfireFootprints).toHaveBeenCalled();
    expect(getFirePerimeters).toHaveBeenCalled();
  });

  test('evacuation zones reach the map and follow the Layers toggle', async () => {
    const MapComponent = require('../MapComponent').default;
    const ring = [[[-118.45, 34.55], [-118.35, 34.55], [-118.35, 34.6], [-118.45, 34.6], [-118.45, 34.55]]];
    const zones = {
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', geometry: { type: 'Polygon', coordinates: ring },
          properties: { zone_id: 'LAC-E018', status: 'order', status_label: 'Order' } },
        { type: 'Feature', geometry: { type: 'Polygon', coordinates: ring },
          properties: { zone_id: 'LAC-E031-B', status: 'warning', status_label: 'Warning' } },
      ],
    };
    const props = {
      brightnessFilter: '', confidenceFilter: '', onFiresUpdated: jest.fn(), setIsFetching: jest.fn(),
      mapStyle: 'mapbox://styles/mapbox/streets-v12', userLocation: null, range: 0, onNearbyFiresUpdate: jest.fn(),
    };

    const { rerender } = render(<MapComponent {...props} evacuations={zones} layerVisibility={{ evacuations: true }} />);
    const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];

    await waitFor(() => {
      expect(map.getLayer('evacuation-zones-fill')).toBeTruthy();
      expect(map.getLayer('evacuation-zones-label')).toBeTruthy();
      expect(map.getSource('evacuation-zones-source').config.data.features).toHaveLength(2);
    });

    // The Layers panel's toggle was a stub wired to nothing. It must now hide
    // all three layers, not just the fill.
    rerender(<MapComponent {...props} evacuations={zones} layerVisibility={{ evacuations: false }} />);
    await waitFor(() => {
      ['evacuation-zones-fill', 'evacuation-zones-outline', 'evacuation-zones-label'].forEach((layerId) => {
        expect(map.setLayoutProperty).toHaveBeenCalledWith(layerId, 'visibility', 'none');
      });
    });

    // New zones from the next bootstrap replace the old ones in place.
    rerender(<MapComponent {...props} evacuations={{ ...zones, features: zones.features.slice(0, 1) }} layerVisibility={{ evacuations: true }} />);
    await waitFor(() => {
      expect(map.getSource('evacuation-zones-source').config.data.features).toHaveLength(1);
    });
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
    expect(screen.getByText(/Fire-spread risk/i)).toBeInTheDocument();
    expect(screen.getByText(/not an observed or predicted official perimeter/i)).toBeInTheDocument();
    // Only the view switch sits above the timeline. The heat layer choice and
    // the diagnostics are under Model details, and nothing does nothing.
    expect(screen.getByRole('button', { name: /Arrival bands/i })).toBeInTheDocument();
    expect(screen.getByText(/Model details/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Observed/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: /Chance of new fire/i })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/forecast timeline slider/i), { target: { value: '1' } });

    await waitFor(() => expect(screen.getByText('12 hours')).toBeInTheDocument());
    expect(renderPredictionRasterFrame).toHaveBeenCalled();
  });

  test('Palisades historical preset uses seeded ignition point', async () => {
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
    fireEvent.click(screen.getByRole('button', { name: /palisades fire/i }));

    await waitFor(() => {
      expect(predictFireSpreadMultistep).toHaveBeenCalledWith(expect.objectContaining({
        lat: 34.078,
        lon: -118.555,
        date: '2025-01-07T18:30:00Z',
        ignition: true,
      }));
    });
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

  test('the band view mutes the basemap and emphasises the active day', async () => {
    // The unit tests for basemapFocus use their own fake map. This is the only
    // check that the app actually CALLS it: the mapbox mock's getStyle()
    // returns just the layers the app added, so a raster layer has to be
    // seeded or setBasemapFocus finds nothing and silently returns.
    const MapComponent = require('../MapComponent').default;

    render(
      <MapComponent
        brightnessFilter=""
        confidenceFilter=""
        onFiresUpdated={jest.fn()}
        setIsFetching={jest.fn()}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        userLocation={null}
        range={0}
        onNearbyFiresUpdate={jest.fn()}
      />
    );

    const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
    map.addLayer({ id: 'satellite', type: 'raster' });

    fireEvent.click(await screen.findByRole('button', { name: /history/i }));
    fireEvent.click(screen.getByRole('button', { name: /camp\/paradise fire/i }));
    expect(await screen.findByTestId('forecast-panel')).toBeInTheDocument();

    await waitFor(() => {
      const calls = map.setPaintProperty.mock.calls;
      expect(calls.some(([layer, prop]) => layer === 'satellite' && prop === 'raster-saturation'))
        .toBe(true);
      expect(calls.some(([layer]) => layer === 'spread-bands-label')).toBe(true);
    });
  });
  describe('forecast panel controls', () => {
    const baseProps = {
      brightnessFilter: '',
      confidenceFilter: '',
      onFiresUpdated: jest.fn(),
      setIsFetching: jest.fn(),
      mapStyle: 'mapbox://styles/mapbox/streets-v12',
      userLocation: null,
      range: 0,
      onNearbyFiresUpdate: jest.fn(),
    };
    const allLayersOn = {
      prediction: true, perimeters: true, hotspots: true, evacuations: true, forecastStart: true,
    };

    async function openCampForecast() {
      fireEvent.click(await screen.findByRole('button', { name: /history/i }));
      fireEvent.click(screen.getByRole('button', { name: /camp\/paradise fire/i }));
      expect(await screen.findByTestId('forecast-panel')).toBeInTheDocument();
      return mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
    }

    test('the heat layer choice appears only in the heat view and drives the heatmap', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} />);
      await openCampForecast();

      expect(screen.queryByRole('radio', { name: /chance of any fire/i })).not.toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: /probability heat/i }));
      fireEvent.click(await screen.findByRole('radio', { name: /chance of any fire/i }));

      await waitFor(() => {
        expect(renderPredictionRasterFrame).toHaveBeenLastCalledWith(
          expect.anything(), expect.anything(), expect.objectContaining({ layerMode: 'next_fire' }),
        );
      });
    });

    test('show on map switches the same layers as the Layers drawer', async () => {
      const MapComponent = require('../MapComponent').default;
      const onToggleLayer = jest.fn();
      render(<MapComponent {...baseProps} layerVisibility={allLayersOn} onToggleLayer={onToggleLayer} />);
      await openCampForecast();

      fireEvent.click(screen.getByRole('checkbox', { name: /official perimeter/i }));
      fireEvent.click(screen.getByRole('checkbox', { name: /model's starting fire/i }));

      expect(onToggleLayer).toHaveBeenNthCalledWith(1, 'perimeters');
      expect(onToggleLayer).toHaveBeenNthCalledWith(2, 'forecastStart');
    });

    test('hiding the forecast hides the arrival bands; the starting fire has its own switch', async () => {
      const MapComponent = require('../MapComponent').default;
      const { rerender } = render(<MapComponent {...baseProps} layerVisibility={allLayersOn} />);
      const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
      await waitFor(() => expect(map.getLayer('spread-bands-fill')).toBeTruthy());

      // The Layers drawer's prediction switch used to leave every band drawn.
      rerender(<MapComponent {...baseProps} layerVisibility={{ ...allLayersOn, prediction: false }} />);
      await waitFor(() => {
        ['spread-bands-fill', 'spread-bands-outline', 'spread-bands-label'].forEach((layerId) => {
          expect(map.setLayoutProperty).toHaveBeenCalledWith(layerId, 'visibility', 'none');
        });
      });
      expect(map.setLayoutProperty).not.toHaveBeenCalledWith('spread-observed-fill', 'visibility', 'none');

      rerender(<MapComponent {...baseProps} layerVisibility={{ ...allLayersOn, forecastStart: false }} />);
      await waitFor(() => {
        expect(map.setLayoutProperty).toHaveBeenCalledWith('spread-observed-fill', 'visibility', 'none');
      });
    });

    test('a hidden forecast stays hidden as the heat timeline advances', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} layerVisibility={{ ...allLayersOn, prediction: false }} />);
      await openCampForecast();

      fireEvent.click(screen.getByRole('button', { name: /probability heat/i }));
      fireEvent.change(screen.getByLabelText(/forecast timeline slider/i), { target: { value: '1' } });

      await waitFor(() => {
        expect(renderPredictionRasterFrame).toHaveBeenLastCalledWith(
          expect.anything(),
          expect.objectContaining({ label: '12 hours' }),
          expect.objectContaining({ visible: false }),
        );
      });
    });

    test('forecast opacity fades the bands and heatmap in place', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} layerVisibility={allLayersOn} onToggleLayer={jest.fn()} />);
      const map = await openCampForecast();
      fireEvent.click(screen.getByRole('button', { name: /probability heat/i }));
      await waitFor(() => expect(renderPredictionRasterFrame).toHaveBeenCalled());
      const framesDrawn = renderPredictionRasterFrame.mock.calls.length;

      fireEvent.change(screen.getByRole('slider', { name: /forecast opacity/i }), { target: { value: '0.4' } });

      await waitFor(() => expect(setPredictionRasterOpacity).toHaveBeenCalledWith(map, 0.4));
      const bandFill = map.setPaintProperty.mock.calls
        .filter(([layerId, property]) => layerId === 'spread-bands-fill' && property === 'fill-opacity')
        .pop();
      expect(bandFill[2][2]).toBeCloseTo(0.62 * 0.4);
      // Fading must not reload the heat image.
      expect(renderPredictionRasterFrame.mock.calls.length).toBe(framesDrawn);
    });

    test('the basemap is not dimmed for a forecast that is switched off', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} layerVisibility={{ ...allLayersOn, prediction: false }} />);
      const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];
      map.addLayer({ id: 'satellite', type: 'raster' });

      await openCampForecast();

      await waitFor(() => expect(renderPredictionRasterFrame).toHaveBeenCalled());
      expect(map.setPaintProperty.mock.calls.some(([layerId]) => layerId === 'satellite')).toBe(false);
    });

    test('the official perimeter is lifted back over each heat frame', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} layerVisibility={allLayersOn} />);
      const map = await openCampForecast();
      map.moveLayer.mockClear();

      fireEvent.click(screen.getByRole('button', { name: /probability heat/i }));

      await waitFor(() => expect(map.moveLayer).toHaveBeenCalledWith('fire-perimeters-outline'));
    });

    test('an incident forecast grows from its perimeter and says so only when it did', async () => {
      const MapComponent = require('../MapComponent').default;
      const ref = React.createRef();
      const forecastFor = predictFireSpreadMultistep.getMockImplementation();
      predictFireSpreadMultistep.mockImplementationOnce(async (args) => ({
        ...(await forecastFor(args)),
        burned_area: { applied: true, acres: 1049.3, cells: 16 },
      }));
      render(<MapComponent ref={ref} {...baseProps} />);
      await waitFor(() => expect(ref.current?.runPredictionForIncident).toBeDefined());

      await act(async () => {
        await ref.current.runPredictionForIncident({
          id: 'IRWIN-BOUQUET', name: 'BOUQUET', lat: 34.5618, lon: -118.4018, hasPerimeter: true,
        });
      });

      expect(predictFireSpreadMultistep).toHaveBeenCalledWith(
        expect.objectContaining({ incidentId: 'IRWIN-BOUQUET', ignition: false }),
      );
      expect(await screen.findByText(/beyond the official perimeter/i)).toBeInTheDocument();
    });

    test('a forecast without a perimeter does not claim to start from one', async () => {
      const MapComponent = require('../MapComponent').default;
      render(<MapComponent {...baseProps} />);
      await openCampForecast();

      expect(screen.queryByText(/beyond the official perimeter/i)).not.toBeInTheDocument();
    });

    test('hotspots carry their age and are coloured by it', async () => {
      const MapComponent = require('../MapComponent').default;
      const twoHoursAgo = new Date(Date.now() - 2 * 3_600_000).toISOString();
      getWildfireData.mockResolvedValueOnce({
        data: { data: [{ latitude: 34.56, longitude: -118.4, brightness: 340, confidence: 'n', timestamp: twoHoursAgo }] },
      });
      render(<MapComponent {...baseProps} />);
      const map = mapboxgl.__mockMaps[mapboxgl.__mockMaps.length - 1];

      await waitFor(() => {
        const [hotspot] = map.getSource('wildfires-source').config.data.features;
        expect(hotspot.properties.ageHours).toBeCloseTo(2, 1);
      });
      expect(map.getLayer('wildfires-layer').paint['circle-color']).toEqual(HOTSPOT_AGE_COLOR);
      expect(map.getLayer('observed-fire-cells-fill').paint['fill-color']).toEqual(HOTSPOT_AGE_COLOR);
    });
  });
});
