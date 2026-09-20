"""Freshness reporting for the on-disk FIRMS and NOAA runtime caches.

Kept out of app.py so the rules can be exercised without standing up the web
framework: what counts as stale weather is domain logic, not routing.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


#: How old the weather a cache entry describes may be before it stops counting
#: as current. Hourly NOAA grids age out fast; FIRMS snapshots cover a
#: multi-day near-real-time window.
NOAA_MAX_DATA_AGE_SEC = int(os.getenv("NOAA_MAX_DATA_AGE_SEC", str(6 * 3600)))
FIRMS_MAX_DATA_AGE_SEC = int(os.getenv("FIRMS_MAX_DATA_AGE_SEC", str(48 * 3600)))

_EPOCH = dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def noaa_valid_time(filename: str) -> Optional[dt.datetime]:
    """Validity hour encoded in ``{YYYYmmddTHH}_{lat}_{lon}.npz``."""
    try:
        return dt.datetime.strptime(filename.split("_", 1)[0], "%Y%m%dT%H").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def firms_valid_time(filename: str) -> Optional[dt.datetime]:
    """Observation date encoded in ``{YYYY-mm-dd}.csv``."""
    try:
        return dt.datetime.strptime(Path(filename).stem, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def latest_file_status(
    path: str | None,
    patterns: Tuple[str, ...] = ("*",),
    *,
    valid_time_from_name: Optional[Callable[[str], Optional[dt.datetime]]] = None,
    max_data_age_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Freshness of a cache directory.

    ``age_seconds`` is how long ago the file was written, which on a container
    that downloads its cache profile at boot says nothing about the weather
    inside it. When the filename encodes a validity time, ``data_age_seconds``
    reports how old the data actually is, and drives ``stale``.
    """
    if not path:
        return {"configured": False}

    root = Path(path)
    if not root.exists():
        return {"configured": True, "ok": False, "path": str(root), "error": "missing"}

    files = [p for pattern in patterns for p in root.glob(pattern) if p.is_file()]
    if not files:
        return {"configured": True, "ok": False, "path": str(root), "error": "empty"}

    if valid_time_from_name is None:
        latest = max(files, key=lambda p: p.stat().st_mtime)
    else:
        latest = max(files, key=lambda p: valid_time_from_name(p.name) or _EPOCH)

    status: Dict[str, Any] = {
        "configured": True,
        "ok": True,
        "path": str(root),
        "latest_file": latest.name,
        "age_seconds": max(0.0, time.time() - latest.stat().st_mtime),
    }
    if valid_time_from_name is None:
        return status

    valid_time = valid_time_from_name(latest.name)
    status["valid_time"] = valid_time.isoformat() if valid_time else None
    if valid_time is None:
        status["stale"] = None
        return status

    data_age = max(0.0, (dt.datetime.now(dt.timezone.utc) - valid_time).total_seconds())
    status["data_age_seconds"] = data_age
    status["stale"] = max_data_age_sec is not None and data_age > max_data_age_sec
    return status


def firms_snapshot_status(directory: str | None) -> Dict[str, Any]:
    return latest_file_status(
        directory,
        ("*.csv", "*.CSV"),
        valid_time_from_name=firms_valid_time,
        max_data_age_sec=FIRMS_MAX_DATA_AGE_SEC,
    )


def noaa_cycle_status(directory: str | None) -> Dict[str, Any]:
    return latest_file_status(
        directory,
        ("*.npz", "*.grib2", "*.grb2"),
        valid_time_from_name=noaa_valid_time,
        max_data_age_sec=NOAA_MAX_DATA_AGE_SEC,
    )
