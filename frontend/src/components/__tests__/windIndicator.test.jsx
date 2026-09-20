import React from 'react';
import { render, screen } from '@testing-library/react';

import WindIndicator from '../WindIndicator';

const santaAna = {
  available: true, calm: false,
  toward_deg: 248.5, toward: 'WSW', speed_ms: 1.5, speed_mph: 3.3, gust_ms: 4.29,
};

test('states the direction the wind blows toward, not from', () => {
  render(<WindIndicator wind={santaAna} />);

  // FROM vs TOWARD is the easiest thing to invert, and inverting it makes a
  // wrong forecast look right.
  expect(screen.getByText(/wind toward wsw/i)).toBeInTheDocument();
});

test('rotates the arrow to the bearing so it can be compared to the spread', () => {
  const { container } = render(<WindIndicator wind={santaAna} />);

  expect(container.querySelector('.wind-indicator__arrow')).toHaveStyle('transform: rotate(248.5deg)');
});

test('tells the reader what the arrow is for', () => {
  render(<WindIndicator wind={santaAna} />);

  expect(screen.getByText(/spread should run with this arrow/i)).toBeInTheDocument();
});

test('reports speed and gusts in mph', () => {
  render(<WindIndicator wind={santaAna} />);

  expect(screen.getByText(/3.3 mph/)).toBeInTheDocument();
  expect(screen.getByText(/gusts 10 mph/i)).toBeInTheDocument();
});

test('calm air says so instead of drawing a confident arrow', () => {
  const { container } = render(<WindIndicator wind={{ available: true, calm: true }} />);

  expect(screen.getByText(/calm/i)).toBeInTheDocument();
  expect(container.querySelector('.wind-indicator__arrow')).toBeNull();
});

test('renders nothing when the model had no wind channels', () => {
  const { container } = render(<WindIndicator wind={{ available: false, reason: 'wind_channels_missing' }} />);

  expect(container).toBeEmptyDOMElement();
});

test('is announced to screen readers with direction and speed', () => {
  render(<WindIndicator wind={santaAna} />);

  expect(screen.getByRole('img', { name: /toward WSW, 3.3 miles per hour/i })).toBeInTheDocument();
});
