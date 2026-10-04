import React from 'react';
import { render, screen } from '@testing-library/react';
import DataFreshness from '../DataFreshness';

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
