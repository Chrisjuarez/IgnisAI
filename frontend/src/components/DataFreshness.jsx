// frontend/src/components/DataFreshness.jsx
//
// How old each source behind an incident is. Perimeters come from aircraft and
// detections from satellites, on different clocks, so each gets its own line -
// and the next satellite pass says when the picture can next change.

import React from 'react';
import { clockTime, timeAgo, timeUntil } from '../utils/relativeTime';

/**
 * Whether satellites have seen the fire since a forecast was run. `basis` is
 * the incident's freshness as it stood then; both sides come from the same
 * backend rule, so a detection the forecast already had never counts as new.
 */
export function hasNewDetectionsSince(basis, freshness) {
  if (!basis) return false;
  const latest = Date.parse(freshness?.lastDetectionAt);
  if (!Number.isFinite(latest)) return false;
  const seen = Date.parse(basis.lastDetectionAt);
  return !Number.isFinite(seen) || latest > seen;
}

export default function DataFreshness({
  freshness,
  newDetectionsSinceForecast = false,
  onRerunForecast,
  rerunning = false,
  now = Date.now(),
}) {
  if (!freshness) return null;
  const { lastDetectionAt, perimeterMappedAt } = freshness;
  const [nextPass] = freshness.nextPasses || [];

  return (
    <>
      {newDetectionsSinceForecast && (
        <div className="data-freshness-notice" role="status">
          <span>New satellite detection since your forecast</span>
          <button type="button" onClick={onRerunForecast} disabled={rerunning}>
            {rerunning ? 'Running...' : 'Re-run'}
          </button>
        </div>
      )}
      <dl className="data-freshness" aria-label="Data freshness">
        <dt>Satellite detection</dt>
        <dd>
          {lastDetectionAt
            ? `${clockTime(lastDetectionAt, now)} · ${timeAgo(lastDetectionAt, now)}`
            : 'None in the last 2 days'}
        </dd>
        <dt>Next satellite pass</dt>
        <dd>
          {nextPass
            ? `${clockTime(nextPass.at, now)} · ${nextPass.satellite} ${nextPass.instrument} · ${timeUntil(nextPass.at, now)}`
            : 'Unavailable'}
        </dd>
        <dt>Perimeter mapped</dt>
        <dd>
          {perimeterMappedAt
            ? `${clockTime(perimeterMappedAt, now)} · ${timeAgo(perimeterMappedAt, now)}`
            : 'No mapped perimeter'}
        </dd>
      </dl>
    </>
  );
}
