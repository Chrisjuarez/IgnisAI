import { clockTime, timeAgo, timeUntil } from '../relativeTime';

const NOW = Date.parse('2026-10-04T20:54:00Z');
const minutesBefore = (n) => new Date(NOW - n * 60 * 1000).toISOString();
const minutesAfter = (n) => new Date(NOW + n * 60 * 1000).toISOString();

describe('timeAgo', () => {
  test.each([
    [minutesBefore(0.5), 'just now'],
    [minutesBefore(21), '21 min ago'],
    [minutesBefore(17 * 60), '17 hr ago'],
    [minutesBefore(3 * 24 * 60), '3 d ago'],
  ])('%s reads as %s', (iso, expected) => {
    expect(timeAgo(iso, NOW)).toBe(expected);
  });

  test('a clock running slightly ahead is not "in the future"', () => {
    expect(timeAgo(minutesAfter(2), NOW)).toBe('just now');
  });
});

describe('timeUntil', () => {
  test.each([
    [minutesAfter(35), 'in 35 min'],
    [minutesAfter(0.4), 'in 1 min'],
    [minutesAfter(9 * 60), 'in 9 hr'],
    [minutesBefore(1), 'now'],
  ])('%s reads as %s', (iso, expected) => {
    expect(timeUntil(iso, NOW)).toBe(expected);
  });
});

describe('clockTime', () => {
  test('gives the date only when it is not today', () => {
    const today = clockTime(minutesBefore(21), NOW);
    const yesterday = clockTime(minutesBefore(24 * 60), NOW);

    expect(today).not.toMatch(/Oct/);
    expect(yesterday).toMatch(/Oct 3/);
  });

  test('names the time zone', () => {
    expect(clockTime(minutesBefore(21), NOW)).toMatch(/M\s(?:[A-Z]{2,5}|GMT[+-][\d:]+)$/);
  });
});

test.each([null, undefined, '', 'not a time'])('%p is unreadable everywhere', (value) => {
  expect(timeAgo(value, NOW)).toBeNull();
  expect(timeUntil(value, NOW)).toBeNull();
  expect(clockTime(value, NOW)).toBeNull();
});
