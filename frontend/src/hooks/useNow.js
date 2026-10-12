// frontend/src/hooks/useNow.js
//
// The current time, refreshed on an interval, so labels like "5 min ago" keep
// counting while the page is open.

import { useEffect, useState } from 'react';

export default function useNow(intervalMs = 60 * 1000) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);

  return now;
}
