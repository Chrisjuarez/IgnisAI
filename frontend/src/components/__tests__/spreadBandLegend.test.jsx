import React from 'react';
import { render, screen } from '@testing-library/react';

import SpreadBandLegend from '../SpreadBandLegend';

const bands = {
  type: 'FeatureCollection',
  features: [
    { properties: { day: 3, color: '#fd8d3c', lead_hours: 72 } },
    { properties: { day: 3, color: '#fd8d3c', lead_hours: 72 } },
    { properties: { day: 1, color: '#bd0026', lead_hours: 24 } },
    { properties: { day: 2, color: '#f03b20', lead_hours: 48 } },
  ],
};

test('lists each day once, in chronological order', () => {
  render(<SpreadBandLegend bands={bands} />);

  const labels = screen.getAllByText(/^Day \d/).map((n) => n.textContent);
  expect(labels).toEqual(['Day 1 · +24h', 'Day 2 · +48h', 'Day 3 · +72h']);
});

test('says the shading means arrival day, not intensity', () => {
  render(<SpreadBandLegend bands={bands} />);

  expect(screen.getByText(/day fire first reaches an area, not intensity/i)).toBeInTheDocument();
});

test('warns that later days are less certain', () => {
  render(<SpreadBandLegend bands={bands} />);

  expect(screen.getByText(/later days are less certain/i)).toBeInTheDocument();
});

test('surfaces degraded weather next to the colours it affects', () => {
  render(<SpreadBandLegend bands={bands} degraded />);

  expect(screen.getByText(/treat as indicative/i)).toBeInTheDocument();
});

test('renders nothing when there are no bands, rather than an empty key', () => {
  const { container } = render(<SpreadBandLegend bands={{ type: 'FeatureCollection', features: [] }} />);

  expect(container).toBeEmptyDOMElement();
});

test('tolerates a missing collection', () => {
  const { container } = render(<SpreadBandLegend bands={undefined} />);

  expect(container).toBeEmptyDOMElement();
});
