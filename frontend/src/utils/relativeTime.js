// frontend/src/utils/relativeTime.js
//
// Times as people read them on a fire map: how long ago, how long until, and
// the clock time - with the date only when it is not today. Every value that
// cannot be read comes back as null, so a caller decides what "unknown" says.

const MINUTE_MS = 60 * 1000;
const JUST_NOW_MS = 45 * 1000;

function toMillis(value) {
  const ms = typeof value === 'number' ? value : Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

function duration(ms) {
  const minutes = Math.round(ms / MINUTE_MS);
  if (minutes < 60) return `${Math.max(1, minutes)} min`;
  const hours = Math.round(minutes / 60);
  return hours < 36 ? `${hours} hr` : `${Math.round(hours / 24)} d`;
}

/** "just now", "25 min ago", "17 hr ago", "2 d ago". */
export function timeAgo(value, now = Date.now()) {
  const ms = toMillis(value);
  if (ms == null) return null;
  const elapsed = Math.max(0, now - ms);
  return elapsed < JUST_NOW_MS ? 'just now' : `${duration(elapsed)} ago`;
}

/** "in 35 min", "in 9 hr", and "now" once it has arrived. */
export function timeUntil(value, now = Date.now()) {
  const ms = toMillis(value);
  if (ms == null) return null;
  const remaining = ms - now;
  return remaining <= 0 ? 'now' : `in ${duration(remaining)}`;
}

/** "1:33 PM" today, "Oct 3, 9:09 PM" on any other day. */
export function clockTime(value, now = Date.now()) {
  const ms = toMillis(value);
  if (ms == null) return null;
  const date = new Date(ms);
  return date.toDateString() === new Date(now).toDateString()
    ? date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    : date.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}
