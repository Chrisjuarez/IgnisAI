import { forgetBasemapFocus, setBasemapFocus } from '../basemapFocus';

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
    expect(map._paint['satellite.raster-brightness-max']).toBeLessThan(1);
  });

  test('restores the style\'s own values rather than a guess', () => {
    const map = fakeMap();
    map.setPaintProperty('satellite', 'raster-saturation', 0.2); // style had its own value
    setBasemapFocus(map, true);
    setBasemapFocus(map, false);

    expect(map._paint['satellite.raster-saturation']).toBe(0.2);
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
  });

  test('forgetting drops the capture without writing anything back', () => {
    // A style reload brings its own paint. Restoring the previous style's
    // values onto the new style's layers is the bug this prevents.
    const map = fakeMap();
    map.setPaintProperty('satellite', 'raster-saturation', 0.2);
    setBasemapFocus(map, true);
    const dimmed = map._paint['satellite.raster-saturation'];

    forgetBasemapFocus(map);
    setBasemapFocus(map, false);

    expect(map._paint['satellite.raster-saturation']).toBe(dimmed);
  });

  test('after forgetting, the next focus re-captures from the new style', () => {
    const map = fakeMap();
    setBasemapFocus(map, true);
    forgetBasemapFocus(map);
    map.setPaintProperty('satellite', 'raster-saturation', 0.5); // the new style's value
    setBasemapFocus(map, true);
    setBasemapFocus(map, false);

    expect(map._paint['satellite.raster-saturation']).toBe(0.5);
  });

  test('forgetting a null map is a no-op', () => {
    expect(() => forgetBasemapFocus(null)).not.toThrow();
  });

  test('survives a layer that rejects the property', () => {
    const map = fakeMap();
    map.setPaintProperty = () => { throw new Error('unsupported'); };
    expect(() => setBasemapFocus(map, true)).not.toThrow();
  });

  test('two maps keep their own saved state', () => {
    const a = fakeMap();
    const b = fakeMap();
    a.setPaintProperty('satellite', 'raster-saturation', 0.2);
    b.setPaintProperty('satellite', 'raster-saturation', 0.7);
    setBasemapFocus(a, true);
    setBasemapFocus(a, false);
    setBasemapFocus(b, true);
    setBasemapFocus(b, false);

    expect(a._paint['satellite.raster-saturation']).toBe(0.2);
    expect(b._paint['satellite.raster-saturation']).toBe(0.7);
  });

  test('a null map is ignored', () => {
    expect(() => setBasemapFocus(null, true)).not.toThrow();
  });
});
