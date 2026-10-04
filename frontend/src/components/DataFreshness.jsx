// frontend/src/components/DataFreshness.jsx
//
// How old each source behind an incident is. Perimeters come from aircraft and
// detections from satellites, on different clocks, so each gets its own line -
// and the next satellite pass says when the picture can next change.

import React from 'react';
import { clockTime, timeAgo, timeUntil } from '../utils/relativeTime';

export default function DataFreshness({ freshness, now = Date.now() }) {
  if (!freshness) return null;
  const { lastDetectionAt, perimeterMappedAt } = freshness;
  const [nextPass] = freshness.nextPasses || [];

  return (
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
  );
}
