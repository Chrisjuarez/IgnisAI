// frontend/src/components/SpreadBandLegend.jsx
//
// The key for the day bands drawn on the map.
//
// A progression map is unreadable without one: the colours mean days, and
// nothing on the map itself says so. Two things this has to get across, because
// both are easy to misread and expensive to misread:
//
//   1. Bands are arrival days, not intensity. A cell sits in the band for the
//      day the fire FIRST reaches it, so the shading is time, not heat.
//   2. Later days are less certain. The palette fades outward for that reason,
//      and the legend says so rather than leaving the fade to be inferred.

import React from 'react';

const DAY_LABEL = (day, leadHours) =>
  leadHours ? `Day ${day} · +${leadHours}h` : `Day ${day}`;

function bandsByDay(collection) {
  const seen = new Map();
  for (const feature of collection?.features || []) {
    const { day, color, lead_hours: leadHours } = feature.properties || {};
    if (day == null || seen.has(day)) continue;
    seen.set(day, { day, color, leadHours });
  }
  return Array.from(seen.values()).sort((a, b) => a.day - b.day);
}

export default function SpreadBandLegend({ bands, degraded = false, className = '' }) {
  const entries = bandsByDay(bands);

  if (!entries.length) return null;

  return (
    <section className={`spread-legend ${className}`.trim()} aria-label="Fire spread day bands">
      <h4 className="spread-legend__title">Forecast spread</h4>
      <ol className="spread-legend__scale">
        {entries.map(({ day, color, leadHours }) => (
          <li key={day} className="spread-legend__entry">
            <span
              className="spread-legend__swatch"
              style={{ backgroundColor: color }}
              aria-hidden="true"
            />
            <span className="spread-legend__label">{DAY_LABEL(day, leadHours)}</span>
          </li>
        ))}
      </ol>
      <p className="spread-legend__note">
        Shading is the day fire first reaches an area, not intensity. Later days are
        less certain.
      </p>
      {degraded && (
        <p className="spread-legend__degraded">
          Weather inputs degraded — treat as indicative.
        </p>
      )}
    </section>
  );
}
