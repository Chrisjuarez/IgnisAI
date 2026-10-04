import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import DataFreshness, { hasNewDetectionsSince } from '../DataFreshness';

const NOW = Date.parse('2026-10-04T20:54:00Z');

describe('DataFreshness', () => {
  test('dates each source on its own clock and says when satellites next look', () => {
    render(
      <DataFreshness
        now={NOW}
        freshness={{
          lastDetectionAt: '2026-10-04T20:33:00.000Z',
          perimeterMappedAt: '2026-10-04T04:09:00.000Z',
          nextPasses: [
            { satellite: 'NOAA-20', instrument: 'VIIRS', at: '2026-10-04T21:29:00.000Z' },
            { satellite: 'Aqua', instrument: 'MODIS', at: '2026-10-04T22:48:00.000Z' },
          ],
        }}
      />
    );

    expect(screen.getByText(/21 min ago/)).toBeInTheDocument();
    expect(screen.getByText(/NOAA-20 VIIRS · in 35 min/)).toBeInTheDocument();
    expect(screen.queryByText(/Aqua/)).not.toBeInTheDocument();
    expect(screen.getByText(/17 hr ago/)).toBeInTheDocument();
  });

  test('says plainly when a source has nothing to show', () => {
    render(<DataFreshness now={NOW} freshness={{ lastDetectionAt: null, perimeterMappedAt: null, nextPasses: [] }} />);

    expect(screen.getByText('None in the last 2 days')).toBeInTheDocument();
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
    expect(screen.getByText('No mapped perimeter')).toBeInTheDocument();
  });

  test('renders nothing before the incident detail arrives', () => {
    const { container } = render(<DataFreshness now={NOW} freshness={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe('hasNewDetectionsSince', () => {
  const at = (iso) => ({ lastDetectionAt: iso });

  test.each([
    ['a later detection', at('2026-10-04T09:13:00Z'), at('2026-10-04T20:33:00Z'), true],
    ['the same detection the forecast had', at('2026-10-04T20:33:00Z'), at('2026-10-04T20:33:00Z'), false],
    ['a first detection after a forecast that had none', at(null), at('2026-10-04T20:33:00Z'), true],
    ['still nothing detected', at(null), at(null), false],
    ['no forecast run', null, at('2026-10-04T20:33:00Z'), false],
  ])('%s', (_label, basis, freshness, expected) => {
    expect(hasNewDetectionsSince(basis, freshness)).toBe(expected);
  });
});

describe('re-run notice', () => {
  const freshness = { lastDetectionAt: '2026-10-04T20:33:00.000Z', perimeterMappedAt: null, nextPasses: [] };

  test('offers a re-run when satellites have seen the fire since the forecast', () => {
    const onRerunForecast = jest.fn();
    render(<DataFreshness now={NOW} freshness={freshness} newDetectionsSinceForecast onRerunForecast={onRerunForecast} />);

    expect(screen.getByRole('status')).toHaveTextContent('New satellite detection since your forecast');
    fireEvent.click(screen.getByRole('button', { name: 'Re-run' }));
    expect(onRerunForecast).toHaveBeenCalledTimes(1);
  });

  test('cannot be pressed twice while the forecast is running', () => {
    render(<DataFreshness now={NOW} freshness={freshness} newDetectionsSinceForecast rerunning onRerunForecast={jest.fn()} />);

    expect(screen.getByRole('button', { name: 'Running...' })).toBeDisabled();
  });

  test('stays quiet otherwise', () => {
    render(<DataFreshness now={NOW} freshness={freshness} />);

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});

