import { renderPredictionRasterFrame } from '../addPredictionOverlay';

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
