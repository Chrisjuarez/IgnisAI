// frontend/src/hooks/useRecheckAfterNextPass.js
//
// Calls `recheck` once the next satellite pass's detections have had time to
// reach FIRMS. The answer carries its own next pass, so the timer reschedules
// itself: an open incident stays current with a handful of requests a day,
// timed to when something can actually have changed, instead of polling.

import { useEffect } from 'react';

// Bouquet's 1:33 PM pass was on the map about 20 minutes after the satellite
// was overhead. Half an hour leaves room for a slower day.
export const DETECTION_LATENCY_MS = 30 * 60 * 1000;

export default function useRecheckAfterNextPass(freshness, recheck) {
  const nextPassAt = Date.parse(freshness?.nextPasses?.[0]?.at);

  useEffect(() => {
    if (!Number.isFinite(nextPassAt) || !recheck) return undefined;
    const timer = setTimeout(recheck, Math.max(0, nextPassAt + DETECTION_LATENCY_MS - Date.now()));
    return () => clearTimeout(timer);
  }, [nextPassAt, recheck]);
}
