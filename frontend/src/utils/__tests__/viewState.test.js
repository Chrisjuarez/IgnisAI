import { parseViewState, serializeViewState, viewStateChanged } from '../viewState';

describe('round trip', () => {
  it('survives a reload with the view intact', () => {
    const view = { center: { lat: 34.078, lon: -118.555 }, zoom: 11.5, incidentId: 'wfigs:abc', day: 3 };

    expect(parseViewState(serializeViewState(view))).toEqual(view);
  });

  it('omits what is absent rather than writing empty keys', () => {
    expect(serializeViewState({ incidentId: 'x' })).toBe('?incident=x');
    expect(serializeViewState({})).toBe('');
  });
});

describe('parsing hostile input', () => {
  it('ignores a half-specified centre', () => {
    expect(parseViewState('?lat=34.0').center).toBeUndefined();
    expect(parseViewState('?lon=-118.5').center).toBeUndefined();
  });

  it('rejects out-of-range coordinates', () => {
    expect(parseViewState('?lat=999&lon=-118').center).toBeUndefined();
    expect(parseViewState('?lat=34&lon=-999').center).toBeUndefined();
  });

  it('rejects a nonsense zoom', () => {
    expect(parseViewState('?z=99').zoom).toBeUndefined();
    expect(parseViewState('?z=abc').zoom).toBeUndefined();
  });

  it('returns an empty view for an empty query', () => {
    expect(parseViewState('')).toEqual({});
    expect(parseViewState(undefined)).toEqual({});
  });
});

describe('what is worth recording', () => {
  const base = { center: { lat: 34.0, lon: -118.0 }, zoom: 10, incidentId: 'a', day: 1 };

  it('does not record a hairline pan', () => {
    // The map fires move events continuously; recording each one would make
    // the back button useless.
    const nudged = { ...base, center: { lat: 34.00001, lon: -118.00001 } };

    expect(viewStateChanged(base, nudged)).toBe(false);
  });

  it('records a real pan', () => {
    expect(viewStateChanged(base, { ...base, center: { lat: 34.5, lon: -118.0 } })).toBe(true);
  });

  it('records selecting a different fire immediately', () => {
    expect(viewStateChanged(base, { ...base, incidentId: 'b' })).toBe(true);
  });

  it('records stepping to another forecast day', () => {
    expect(viewStateChanged(base, { ...base, day: 2 })).toBe(true);
  });

  it('treats a first view as a change', () => {
    expect(viewStateChanged(null, base)).toBe(true);
  });
});

describe('precision', () => {
  it('does not imply accuracy the map does not have', () => {
    const qs = serializeViewState({ center: { lat: 34.0780123456, lon: -118.5551987654 } });

    expect(qs).toContain('lat=34.07801');
    expect(qs).not.toContain('34.0780123');
  });
});
