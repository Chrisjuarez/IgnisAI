// frontend/src/components/SiteExposurePanel.jsx
//
// The map answers "where will this fire go". An asset owner asks the narrower
// question: does it reach MY site, and when. This panel is that question.
//
// Two things it must not do, because both mislead someone making a decision:
//
//   1. Show the model's daily value as if it were the risk. The model predicts
//      a delta (cells that newly burn each day), so a site that burns on day 2
//      reports near zero on day 3 — "low risk" for a site already lost. The
//      headline is always cumulative; the daily figure is shown beside it as
//      the model's own output, labelled as such.
//
//   2. Render a number when the site is outside the forecast tile. A rollout
//      covers 32 km around the ignition; beyond that there is no forecast, and
//      the panel says so rather than implying safety.

import React, { useCallback, useState } from 'react';

import { getSiteExposure } from '../api';

const RISK_TIERS = {
  low: { label: 'Low', tone: 'low' },
  medium: { label: 'Medium', tone: 'medium' },
  high: { label: 'High', tone: 'high' },
  extreme: { label: 'Extreme', tone: 'extreme' },
};

const HORIZON_CHOICES = [1, 2, 3];

function tierOf(risk) {
  return RISK_TIERS[String(risk || '').toLowerCase()] || { label: '—', tone: 'unknown' };
}

function asPercent(probability) {
  if (probability == null || Number.isNaN(Number(probability))) return '—';
  return `${(Number(probability) * 100).toFixed(1)}%`;
}

function ArrivalSummary({ arrival }) {
  if (!arrival) return null;
  if (!arrival.reached) {
    return (
      <p className="site-exposure__arrival site-exposure__arrival--clear">
        Fire does not reach this site within the forecast horizon.
      </p>
    );
  }
  return (
    <p className="site-exposure__arrival">
      Reaches the site on <strong>day {arrival.day}</strong> (+{arrival.lead_hours}h), at a{' '}
      {asPercent(arrival.threshold)} probability threshold.
    </p>
  );
}

function ExposureTable({ series }) {
  return (
    <table className="site-exposure__table">
      <thead>
        <tr>
          <th scope="col">Day</th>
          <th scope="col">Cumulative</th>
          <th scope="col">Risk</th>
          <th scope="col" title="The model's own per-day output. Falls once the site has burned.">
            Daily
          </th>
        </tr>
      </thead>
      <tbody>
        {series.map((entry) => {
          const tier = tierOf(entry.risk);
          return (
            <tr key={entry.day}>
              <th scope="row">Day {entry.day}</th>
              <td className="site-exposure__cumulative">{asPercent(entry.cumulative_probability)}</td>
              <td>
                <span className={`site-exposure__tier site-exposure__tier--${tier.tone}`}>{tier.label}</span>
              </td>
              <td className="site-exposure__daily">{asPercent(entry.daily_probability)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Result({ exposure }) {
  if (!exposure) return null;

  if (exposure.covered === false) {
    return (
      <div className="site-exposure__uncovered" role="status">
        <p><strong>No forecast for this site.</strong></p>
        <p>{exposure.detail}</p>
        <p className="site-exposure__separation">
          Site is {exposure.separation_km} km from the ignition.
        </p>
      </div>
    );
  }

  const tier = tierOf(exposure.risk_within_horizon);
  return (
    <div className="site-exposure__result">
      <div
        className={`site-exposure__headline site-exposure__headline--${tier.tone}`}
        role="group"
        aria-label="Exposure within horizon"
      >
        <span className="site-exposure__headline-value">{asPercent(exposure.probability_within_horizon)}</span>
        <span className="site-exposure__headline-label">
          chance of fire reaching the site &middot; {tier.label}
        </span>
      </div>
      <ArrivalSummary arrival={exposure.arrival} />
      <ExposureTable series={exposure.series || []} />
      <p className="site-exposure__meta">
        Ignition {exposure.separation_km} km away.
        {exposure.quality?.degraded ? ' Weather inputs degraded — treat as indicative.' : ''}
      </p>
    </div>
  );
}

export default function SiteExposurePanel({ site, ignition, date, className = '' }) {
  const [days, setDays] = useState(3);
  const [exposure, setExposure] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);

  const run = useCallback(async () => {
    if (!site) return;
    setRunning(true);
    setError(null);
    try {
      const data = await getSiteExposure({
        siteLat: site.lat,
        siteLon: site.lon,
        ignitionLat: ignition?.lat,
        ignitionLon: ignition?.lon,
        days,
        date,
      });
      setExposure(data);
    } catch (err) {
      setExposure(null);
      setError(err?.response?.data?.detail || err.message || 'Exposure request failed');
    } finally {
      setRunning(false);
    }
  }, [site, ignition, days, date]);

  return (
    <section className={`site-exposure ${className}`.trim()}>
      <header className="site-exposure__header">
        <h3>Site fire risk</h3>
        {site ? (
          <p className="site-exposure__site">
            {site.name || 'Selected site'} &middot; {Number(site.lat).toFixed(3)}, {Number(site.lon).toFixed(3)}
          </p>
        ) : (
          <p className="site-exposure__site">Select a site to assess.</p>
        )}
      </header>

      <div className="site-exposure__controls">
        <fieldset className="site-exposure__horizon">
          <legend>Horizon</legend>
          {HORIZON_CHOICES.map((choice) => (
            <button
              key={choice}
              type="button"
              className={days === choice ? 'active' : ''}
              onClick={() => setDays(choice)}
              disabled={running}
            >
              {choice}d
            </button>
          ))}
        </fieldset>
        <button type="button" className="site-exposure__run" onClick={run} disabled={!site || running}>
          {running ? 'Running…' : 'Run forecast'}
        </button>
      </div>

      {error && <p className="site-exposure__error" role="alert">{error}</p>}
      <Result exposure={exposure} />
    </section>
  );
}
