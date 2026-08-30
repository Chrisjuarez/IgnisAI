"""The wind the model actually saw, in a form a map can draw.

The spread forecast is dominated by wind, so the first question anyone asks of
an output is "is it going the way the wind is blowing". Today that is
unanswerable from the map: the wind is in the tensor, and the only trace of it
in the response is a normalised channel mean buried in input_summary that a
client would have to know the sign convention to interpret.

This lifts it out and names it. Convention matches the validator: the bearing
is the direction the wind blows TOWARD, which is what you want when comparing
against which way a fire is drawn spreading. Note that meteorological
convention normally reports the direction wind comes FROM - the opposite - so
the field is named `toward_deg` rather than left to be assumed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

COMPASS_POINTS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)

#: Below this the direction is noise, and drawing a confident arrow for it
#: would imply a steer the model does not have.
CALM_MS = 0.5


def compass_point(toward_deg: float) -> str:
    return COMPASS_POINTS[int((toward_deg + 11.25) % 360 // 22.5)]


def wind_summary(u_ms: Optional[float], v_ms: Optional[float],
                 gust_ms: Optional[float] = None) -> Dict[str, Any]:
    """Describe a wind vector for display.

    `u` is eastward, `v` is northward, both in m/s, as the model consumes them.
    """
    if u_ms is None or v_ms is None:
        return {"available": False, "reason": "wind_channels_missing"}

    u, v = float(u_ms), float(v_ms)
    speed = math.hypot(u, v)
    if speed < CALM_MS:
        return {"available": True, "calm": True, "speed_ms": round(speed, 2),
                "toward_deg": None, "toward": "calm",
                "gust_ms": None if gust_ms is None else round(float(gust_ms), 2)}

    toward = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0
    return {
        "available": True,
        "calm": False,
        "u_ms": round(u, 2),
        "v_ms": round(v, 2),
        "speed_ms": round(speed, 2),
        "speed_mph": round(speed * 2.23694, 1),
        "toward_deg": round(toward, 1),
        "toward": compass_point(toward),
        "gust_ms": None if gust_ms is None else round(float(gust_ms), 2),
        "note": "toward_deg is the direction the wind blows TOWARD, not the meteorological FROM.",
    }


def wind_from_channels(channels: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the mean wind out of the per-channel stats the response already builds."""
    def mean_of(name: str) -> Optional[float]:
        entry = channels.get(name)
        return entry.get("mean") if isinstance(entry, dict) else None

    return wind_summary(mean_of("u"), mean_of("v"), mean_of("gust"))
