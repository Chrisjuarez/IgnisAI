// frontend/src/utils/viewState.js
//
// The view, expressed in the URL.
//
// Reloading used to drop you back at the default map with nothing selected,
// because none of what you were looking at lived anywhere durable. Putting it
// in the query string fixes that and buys something more useful for free: a
// URL that describes a view can be sent to someone else. "Look at this fire"
// stops being a screenshot.
//
// Only what a viewer would notice losing goes in here. Transient things -
// whether a panel is expanded, which frame an animation is on - would make the
// URL churn on every interaction and pollute browser history for no gain.

const PRECISION = 5; // ~1 m; more digits imply accuracy the map does not have.

function num(value, fallback = null) {
  // Number(null) is 0 and Number('') is 0, so an absent parameter would
  // otherwise parse as a real coordinate and drop the map at 0,0.
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Read a view out of a query string. Unknown or malformed values are dropped. */
export function parseViewState(search) {
  const params = new URLSearchParams(search || '');
  const lat = num(params.get('lat'));
  const lon = num(params.get('lon'));
  const zoom = num(params.get('z'));
  const incidentId = params.get('incident') || null;
  const day = num(params.get('day'));

  const view = {};
  // A half-specified centre is not a centre.
  if (lat != null && lon != null && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
    view.center = { lat, lon };
  }
  if (zoom != null && zoom >= 0 && zoom <= 24) view.zoom = zoom;
  if (incidentId) view.incidentId = incidentId;
  if (day != null && day >= 1) view.day = Math.round(day);
  return view;
}

/** Serialize a view to a query string, omitting anything absent. */
export function serializeViewState({ center, zoom, incidentId, day } = {}) {
  const params = new URLSearchParams();
  if (center && Number.isFinite(center.lat) && Number.isFinite(center.lon)) {
    params.set('lat', center.lat.toFixed(PRECISION));
    params.set('lon', center.lon.toFixed(PRECISION));
  }
  if (Number.isFinite(zoom)) params.set('z', Number(zoom).toFixed(2));
  if (incidentId) params.set('incident', incidentId);
  if (Number.isFinite(day)) params.set('day', String(Math.round(day)));
  const qs = params.toString();
  return qs ? `?${qs}` : '';
}

/**
 * Whether a change is worth writing to the URL.
 *
 * The map fires move events continuously, and pushing every one would make the
 * back button useless. Sub-pixel pans and hairline zooms are not view changes
 * anyone means to record.
 */
export function viewStateChanged(previous, next, { minZoomDelta = 0.25, minDegDelta = 0.0005 } = {}) {
  if (!previous) return true;
  if (previous.incidentId !== next.incidentId) return true;
  if (previous.day !== next.day) return true;

  const zoomMoved = Math.abs((previous.zoom ?? 0) - (next.zoom ?? 0)) >= minZoomDelta;
  const a = previous.center;
  const b = next.center;
  if (!a || !b) return Boolean(a) !== Boolean(b) || zoomMoved;

  const panned = Math.abs(a.lat - b.lat) >= minDegDelta || Math.abs(a.lon - b.lon) >= minDegDelta;
  return panned || zoomMoved;
}
