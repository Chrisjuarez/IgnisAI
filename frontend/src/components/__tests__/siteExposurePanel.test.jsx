import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

import SiteExposurePanel from '../SiteExposurePanel';
import { getSiteExposure } from '../../api';

jest.mock('../../api', () => ({ getSiteExposure: jest.fn() }));

const SITE = { name: 'Mesa Verde PV', lat: 34.078, lon: -118.555 };
const IGNITION = { lat: 34.055, lon: -118.52 };

// A site that burns on day 2: the model's daily value collapses on day 3,
// while cumulative exposure stays high. This is the case that must not be
// rendered as "risk went away".
const BURNED_ON_DAY_TWO = {
  covered: true,
  separation_km: 4.11,
  series: [
    { day: 1, lead_hours: 24, daily_probability: 0.0794, cumulative_probability: 0.0794, risk: 'medium' },
    { day: 2, lead_hours: 48, daily_probability: 0.5885, cumulative_probability: 0.5885, risk: 'extreme' },
    { day: 3, lead_hours: 72, daily_probability: 0.0217, cumulative_probability: 0.5885, risk: 'extreme' },
  ],
  probability_within_horizon: 0.5885,
  risk_within_horizon: 'extreme',
  arrival: { threshold: 0.1, reached: true, day: 2, lead_hours: 48 },
  quality: { degraded: false },
};

async function runPanel(payload, props = {}) {
  getSiteExposure.mockResolvedValue(payload);
  render(<SiteExposurePanel site={SITE} ignition={IGNITION} {...props} />);
  fireEvent.click(screen.getByRole('button', { name: /run forecast/i }));
  await waitFor(() => expect(getSiteExposure).toHaveBeenCalled());
}

beforeEach(() => jest.clearAllMocks());

test('headline reports cumulative exposure, not the final day delta', async () => {
  await runPanel(BURNED_ON_DAY_TWO);

  // 58.9% is cumulative. 2.2% is day 3's delta and must not be the headline.
  const headline = await screen.findByRole('group', { name: /exposure within horizon/i });
  expect(within(headline).getByText('58.9%')).toBeInTheDocument();
  expect(within(headline).queryByText('2.2%')).not.toBeInTheDocument();
});

test('day 3 still reads extreme for a site that burned on day 2', async () => {
  await runPanel(BURNED_ON_DAY_TWO);

  const dayThree = (await screen.findByText('Day 3')).closest('tr');
  expect(dayThree).toHaveTextContent('58.9%');
  expect(dayThree).toHaveTextContent('Extreme');
});

test('arrival day is surfaced so the user knows when to act', async () => {
  await runPanel(BURNED_ON_DAY_TWO);

  const arrival = await screen.findByText(/reaches the site on/i);
  expect(arrival).toHaveTextContent('day 2');
  expect(arrival).toHaveTextContent('+48h');
});

test('a site outside the tile shows no forecast rather than a low number', async () => {
  await runPanel({
    covered: false,
    reason: 'site_outside_forecast_tile',
    detail: 'The forecast covers a 32 km tile centred on the ignition; this site falls outside it.',
    separation_km: 68.24,
    series: [],
    probability_within_horizon: null,
    risk_within_horizon: null,
  });

  expect(await screen.findByText(/no forecast for this site/i)).toBeInTheDocument();
  expect(screen.getByText(/68.24 km from the ignition/i)).toBeInTheDocument();
  expect(screen.queryByText(/chance of fire reaching the site/i)).not.toBeInTheDocument();
});

test('a fire that never arrives says so explicitly', async () => {
  await runPanel({
    ...BURNED_ON_DAY_TWO,
    probability_within_horizon: 0.004,
    risk_within_horizon: 'low',
    arrival: { threshold: 0.1, reached: false, day: null, lead_hours: null },
  });

  expect(await screen.findByText(/does not reach this site/i)).toBeInTheDocument();
});

test('degraded weather is disclosed alongside the number', async () => {
  await runPanel({ ...BURNED_ON_DAY_TWO, quality: { degraded: true } });

  expect(await screen.findByText(/treat as indicative/i)).toBeInTheDocument();
});

test('request failures surface instead of leaving a stale figure', async () => {
  getSiteExposure.mockRejectedValue(new Error('tilesvc unreachable'));
  render(<SiteExposurePanel site={SITE} ignition={IGNITION} />);
  fireEvent.click(screen.getByRole('button', { name: /run forecast/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent('tilesvc unreachable');
});

test('horizon choice is passed through to the request', async () => {
  getSiteExposure.mockResolvedValue(BURNED_ON_DAY_TWO);
  render(<SiteExposurePanel site={SITE} ignition={IGNITION} />);

  fireEvent.click(screen.getByRole('button', { name: '1d' }));
  fireEvent.click(screen.getByRole('button', { name: /run forecast/i }));

  await waitFor(() => expect(getSiteExposure).toHaveBeenCalledWith(
    expect.objectContaining({ days: 1, siteLat: SITE.lat, siteLon: SITE.lon }),
  ));
});
