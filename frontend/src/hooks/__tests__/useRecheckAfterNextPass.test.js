import { renderHook } from '@testing-library/react';
import useRecheckAfterNextPass, { DETECTION_LATENCY_MS } from '../useRecheckAfterNextPass';

const NOW = Date.parse('2026-10-04T20:54:00Z');
const MINUTE = 60 * 1000;
const freshnessWithPassAt = (iso) => ({ nextPasses: [{ satellite: 'NOAA-20', instrument: 'VIIRS', at: iso }] });

beforeEach(() => {
  jest.useFakeTimers();
  jest.setSystemTime(NOW);
});

afterEach(() => {
  jest.useRealTimers();
});

test('re-checks once the next pass has had time to reach FIRMS, not before', () => {
  const recheck = jest.fn();
  renderHook(() => useRecheckAfterNextPass(freshnessWithPassAt('2026-10-04T21:29:00Z'), recheck));

  jest.advanceTimersByTime(35 * MINUTE + DETECTION_LATENCY_MS - 1);
  expect(recheck).not.toHaveBeenCalled();

  jest.advanceTimersByTime(1);
  expect(recheck).toHaveBeenCalledTimes(1);
});

test('a new answer replaces the pending re-check rather than adding one', () => {
  const recheck = jest.fn();
  const { rerender } = renderHook(({ freshness }) => useRecheckAfterNextPass(freshness, recheck), {
    initialProps: { freshness: freshnessWithPassAt('2026-10-04T21:29:00Z') },
  });

  rerender({ freshness: freshnessWithPassAt('2026-10-04T22:48:00Z') });
  jest.advanceTimersByTime(35 * MINUTE + DETECTION_LATENCY_MS);
  expect(recheck).not.toHaveBeenCalled();

  jest.advanceTimersByTime(79 * MINUTE);
  expect(recheck).toHaveBeenCalledTimes(1);
});

test('re-checks straight away when that time has already passed', () => {
  const recheck = jest.fn();
  renderHook(() => useRecheckAfterNextPass(freshnessWithPassAt('2026-10-04T19:00:00Z'), recheck));

  jest.advanceTimersByTime(0);
  expect(recheck).toHaveBeenCalledTimes(1);
});

test('sets nothing without a predicted pass', () => {
  const recheck = jest.fn();
  renderHook(() => useRecheckAfterNextPass({ nextPasses: [] }, recheck));
  renderHook(() => useRecheckAfterNextPass(undefined, recheck));

  jest.advanceTimersByTime(48 * 60 * MINUTE);
  expect(recheck).not.toHaveBeenCalled();
});
