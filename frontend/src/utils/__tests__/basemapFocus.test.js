import { isBasemapFocused, setBasemapFocus } from '../basemapFocus';

function fakeMap(layers = [{ id: 'satellite', type: 'raster' }, { id: 'roads', type: 'line' }]) {
  const paint = {};
  return {
    getStyle: () => ({ layers }),
    getPaintProperty: (layerId, property) => paint[`${layerId}.${property}`],
    setPaintProperty: (layerId, property, value) => { paint[`${layerId}.${property}`] = value; },
    _paint: paint,
  };
}

describe('basemapFocus', () => {
  test('mutes raster layers and leaves vector layers alone', () => {
    const map = fakeMap();
    setBasemapFocus(map, true);

    expect(map._paint['satellite.raster-saturation']).toBeLessThan(0);
    expect(map._paint['roads.raster-saturation']).toBeUndefined();
    expect(isBasemapFocused(map)).toBe(true);
  });

  test('restores the style\'s own values rather than a guess', () => {
    const map = fakeMap();
    map.setPaintProperty('satellite', 'raster-saturation', 0.2); // style had its own value
    setBasemapFocus(map, true);
    setBasemapFocus(map, false);

    expect(map._paint['satellite.raster-saturation']).toBe(0.2);
    expect(isBasemapFocused(map)).toBe(false);
  });

  test('dimming twice does not overwrite the saved originals', () => {
    // The bug this guards: a second dim captures the dimmed values as the
    // originals, and the restore then puts the dimmed values back.
    const map = fakeMap();
    map.setPaintProperty('satellite', 'raster-saturation', 0.2);
    setBasemapFocus(map, true);
    setBasemapFocus(map, true);
    setBasemapFocus(map, false);

    expect(map._paint['satellite.raster-saturation']).toBe(0.2);
  });

  test('restoring without having dimmed is a no-op', () => {
    const map = fakeMap();
    setBasemapFocus(map, false);
    expect(map._paint).toEqual({});
  });

  test('survives a style that is not ready yet', () => {
    const map = { getStyle: () => { throw new Error('style not loaded'); } };
    expect(() => setBasemapFocus(map, true)).not.toThrow();
    expect(isBasemapFocused(map)).toBe(false);
  });

  test('survives a layer that rejects the property', () => {
    const map = fakeMap();
    map.setPaintProperty = () => { throw new Error('unsupported'); };
    expect(() => setBasemapFocus(map, true)).not.toThrow();
  });

  test('two maps keep their own saved state', () => {
    const a = fakeMap();
    const b = fakeMap();
    setBasemapFocus(a, true);
    expect(isBasemapFocused(a)).toBe(true);
    expect(isBasemapFocused(b)).toBe(false);
  });

  test('a null map is ignored', () => {
    expect(() => setBasemapFocus(null, true)).not.toThrow();
    expect(isBasemapFocused(null)).toBe(false);
  });
});
