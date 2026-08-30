// frontend/src/components/WindIndicator.jsx
//
// The wind the model was actually given, drawn as an arrow.
//
// Spread is wind-dominated, so the first thing worth checking about any
// forecast is whether it runs downwind. Until now that was unanswerable from
// the screen - the wind was inside the tensor and nowhere in the UI - which
// meant a forecast pushing the wrong way looked exactly like one pushing the
// right way.
//
// The arrow points the way the wind BLOWS TOWARD, so it can be compared
// directly against the direction the spread is drawn. Meteorological reports
// conventionally give the direction wind comes FROM, the opposite, so the
// label says "toward" every time rather than relying on the reader's
// assumption.

import React from 'react';

const CALM_LABEL = 'Calm — no steering wind';

function Arrow({ towardDeg }) {
  return (
    <svg
      className="wind-indicator__arrow"
      viewBox="0 0 24 24"
      width="34"
      height="34"
      aria-hidden="true"
      // 0deg points up (north); bearing increases clockwise, same as compass.
      style={{ transform: `rotate(${towardDeg}deg)` }}
    >
      <path d="M12 2 L18 20 L12 16 L6 20 Z" fill="currentColor" />
    </svg>
  );
}

export default function WindIndicator({ wind, className = '' }) {
  if (!wind || wind.available === false) return null;

  if (wind.calm) {
    return (
      <div className={`wind-indicator wind-indicator--calm ${className}`.trim()}>
        <span className="wind-indicator__label">{CALM_LABEL}</span>
      </div>
    );
  }

  const { toward_deg: towardDeg, toward, speed_mph: speedMph, gust_ms: gustMs } = wind;

  return (
    <div
      className={`wind-indicator ${className}`.trim()}
      role="img"
      aria-label={`Wind blowing toward ${toward}, ${speedMph} miles per hour`}
    >
      <Arrow towardDeg={towardDeg} />
      <div className="wind-indicator__readout">
        <span className="wind-indicator__direction">Wind toward {toward}</span>
        <span className="wind-indicator__speed">
          {speedMph} mph
          {gustMs != null && <> · gusts {Math.round(gustMs * 2.23694)} mph</>}
        </span>
        <span className="wind-indicator__hint">
          Spread should run with this arrow.
        </span>
      </div>
    </div>
  );
}
