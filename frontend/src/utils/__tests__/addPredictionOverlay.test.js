import { renderPredictionRasterFrame, setPredictionRasterOpacity } from '../addPredictionOverlay';

function makeMap(overrides = {}) {
  return {
    loaded: jest.fn(() => false),
    isStyleLoaded: jest.fn(() => true),
    on: jest.fn(),
    off: jest.fn(),
    getStyle: jest.fn(() => ({ layers: [] })),
    getLayer: jest.fn(() => false),
    getSource: jest.fn(() => null),
    removeLayer: jest.fn(),
    removeSource: jest.fn(),
    addSource: jest.fn(),
    addLayer: jest.fn(),
    ...overrides,
  };
}

describe('prediction raster overlay rendering', () => {
  test('does not wait forever when the style is ready but map.loaded() is false', async () => {
    const map = makeMap();
    const frame = {
      bounds: [-118.58, 33.97, -118.36, 34.15],
      heatmapUrl: 'data:image/png;base64,abc123',
    };

    await renderPredictionRasterFrame(map, frame);

    expect(map.loaded).not.toHaveBeenCalled();
    expect(map.isStyleLoaded).toHaveBeenCalled();
    expect(map.on).not.toHaveBeenCalled();
    expect(map.addSource).toHaveBeenCalledWith(
      'ignis-pred-raster-src',
      expect.objectContaining({
        type: 'image',
        url: frame.heatmapUrl,
        coordinates: [
          [-118.58, 34.15],
          [-118.36, 34.15],
          [-118.36, 33.97],
          [-118.58, 33.97],
        ],
      }),
    );
    expect(map.addLayer).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'ignis-pred-raster-layer',
        type: 'raster',
      }),
    );
  });

  test('uses rotated raster coordinates from the backend payload when present', async () => {
    const map = makeMap();
    const coordinates = [
      [-118.5793, 34.1125],
      [-118.4094, 34.1461],
      [-118.3686, 34.0069],
      [-118.5383, 33.9734],
    ];

    await renderPredictionRasterFrame(map, {
      bounds: [-118.58, 33.97, -118.36, 34.15],
      coordinates,
      heatmapUrl: 'data:image/png;base64,abc123',
    });

    expect(map.addSource).toHaveBeenCalledWith(
      'ignis-pred-raster-src',
      expect.objectContaining({ coordinates }),
    );
  });
});

describe('forecast visibility and opacity on heat frames', () => {
  const frame = {
    bounds: [-118.58, 33.97, -118.36, 34.15],
    heatmapUrl: 'data:image/png;base64,abc123',
    contour: {
      type: 'FeatureCollection',
      features: [{ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: [[-118.5, 34], [-118.4, 34.1]] } }],
    },
  };

  test('a frame drawn while the forecast is hidden arrives hidden, at the chosen opacity', async () => {
    const map = makeMap();

    await renderPredictionRasterFrame(map, frame, { visible: false, opacity: 0.4 });

    expect(map.addLayer).toHaveBeenCalledWith(expect.objectContaining({
      id: 'ignis-pred-raster-layer',
      layout: { visibility: 'none' },
      paint: expect.objectContaining({ 'raster-opacity': 0.4 }),
    }));
    expect(map.addLayer).toHaveBeenCalledWith(expect.objectContaining({
      id: 'ignis-pred-contour-line',
      layout: { visibility: 'none' },
    }));
  });

  test('defaults to visible at full strength', async () => {
    const map = makeMap();

    await renderPredictionRasterFrame(map, frame);

    expect(map.addLayer).toHaveBeenCalledWith(expect.objectContaining({
      id: 'ignis-pred-raster-layer',
      layout: { visibility: 'visible' },
      paint: expect.objectContaining({ 'raster-opacity': 1 }),
    }));
  });

  test('opacity changes in place and is clamped', () => {
    const map = makeMap({ getLayer: jest.fn(() => true), setPaintProperty: jest.fn() });

    setPredictionRasterOpacity(map, 1.6);

    expect(map.setPaintProperty).toHaveBeenCalledWith('ignis-pred-raster-layer', 'raster-opacity', 1);
  });

  test('opacity is a no-op before any frame is drawn', () => {
    const map = makeMap({ setPaintProperty: jest.fn() });

    setPredictionRasterOpacity(map, 0.5);

    expect(map.setPaintProperty).not.toHaveBeenCalled();
  });
});
