// frontend/src/components/SourceHealthPanel.js
//
// Renders the per-source data freshness/health badges that come back
// from /api/map/bootstrap (`layerStatus`). The dashboard only displayed
// FIRMS + NWS in the status strip; all five sources matter when the
// user is making a decision about whether to trust the map.
//
// Status semantics (from backend/routes/mapData.js):
//   ok=true,  partial=false, stale=false  -> "Live"   (green)
//   ok=true,  partial=true                -> "Partial" (amber)
//   ok=true,  stale=true                  -> "Stale"   (amber)
//   ok=false                              -> "Down"    (red), tooltip = error
//
// We render an inline pill list with a colored dot, the source label,
// the tier word, and a "fetched 4 min ago" relative timestamp. Hover
// on the pill reveals the full source name and any error message.

import React from 'react';

const TIER_LIVE = 'live';
const TIER_PARTIAL = 'partial';
const TIER_STALE = 'stale';
const TIER_DOWN = 'down';

function classifyStatus(status) {
  if (!status || typeof status !== 'object') return TIER_DOWN;
  if (status.ok === false) return TIER_DOWN;
  if (status.partial) return TIER_PARTIAL;
  if (status.stale) return TIER_STALE;
  return TIER_LIVE;
}

function tierLabel(tier) {
  switch (tier) {
    case TIER_LIVE: return 'Live';
    case TIER_PARTIAL: return 'Partial';
    case TIER_STALE: return 'Stale';
    case TIER_DOWN: return 'Down';
    default: return 'Unknown';
  }
}

// "fetched 4 min ago" — accepts ISO string, falls back to em-dash.
function formatRelative(iso) {
  if (!iso) return null;
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) return null;
  const seconds = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${hours} hr ago`;
  return `${Math.round(hours / 24)} d ago`;
}

// Map of layerStatus keys -> short user-facing label. The key list is
// intentionally fixed: if the backend adds a new key it is rendered with
// its raw key as the label (better than silently dropping it).
const LABELS = {
  incidents: 'Incidents',
  perimeters: 'Perimeters',
  hotspots: 'FIRMS',
  alerts: 'NWS',
  evacuations: 'Evacuations',
};

// Order is meaningful (most-prominent first).
const ORDER = ['incidents', 'perimeters', 'hotspots', 'alerts', 'evacuations'];

function StatusPill({ statusKey, status }) {
  const tier = classifyStatus(status);
  const label = LABELS[statusKey] || statusKey;
  const fetched = formatRelative(status?.lastFetchedAt);
  const title = [
    status?.source || label,
    `Status: ${tierLabel(tier)}`,
    Number.isFinite(status?.count) ? `Records: ${status.count.toLocaleString()}` : null,
    status?.error ? `Error: ${status.error}` : null,
    fetched ? `Fetched ${fetched}` : null,
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <span
      className={`source-pill source-pill--${tier}`}
      title={title}
      role="status"
      aria-label={`${label} ${tierLabel(tier)}`}
      data-testid={`source-pill-${statusKey}`}
    >
      <span className="source-pill__dot" aria-hidden="true" />
      <span className="source-pill__label">{label}</span>
      <span className="source-pill__tier">{tierLabel(tier)}</span>
      {fetched ? <span className="source-pill__age">· {fetched}</span> : null}
    </span>
  );
}

/**
 * @param {object} props
 * @param {object} props.layerStatus  layerStatus shape from /map/bootstrap
 * @param {string} [props.className]
 */
export default function SourceHealthPanel({ layerStatus, className = '' }) {
  const status = layerStatus || {};
  const seen = new Set();
  const rendered = [];
  for (const key of ORDER) {
    if (key in status) {
      seen.add(key);
      rendered.push(<StatusPill key={key} statusKey={key} status={status[key]} />);
    }
  }
  // Render any extra keys the backend added that we didn't anticipate.
  for (const key of Object.keys(status)) {
    if (!seen.has(key)) {
      rendered.push(<StatusPill key={key} statusKey={key} status={status[key]} />);
    }
  }
  if (rendered.length === 0) return null;

  return (
    <div className={`source-health-panel ${className}`} role="region" aria-label="Data sources">
      {rendered}
    </div>
  );
}

// Exported for unit tests.
export const _internal = { classifyStatus, tierLabel, formatRelative, LABELS, ORDER };
