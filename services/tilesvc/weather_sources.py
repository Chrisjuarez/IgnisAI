from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


DAILY_MEAN_CHANNELS = ("u", "v", "gust", "tempC", "q")
DAILY_SUM_CHANNELS = ("precip",)


def aggregate_daily_weather(hourly: Mapping[str, np.ndarray], *, samples_per_day: int) -> Dict[str, np.ndarray]:
    """
    Aggregate hourly/3-hourly gridded weather to the model's daily transition.

    Input arrays are [N,H,W]. The first axis is grouped into 24-hour windows
    (24 samples for hourly HRRR/RAP, 8 samples for 3-hourly GFS).
    """
    if samples_per_day <= 0:
        raise ValueError("samples_per_day must be positive")
    result: Dict[str, np.ndarray] = {}
    for name, raw in hourly.items():
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"weather channel {name!r} must be [N,H,W], got {arr.shape}")
        days = arr.shape[0] // samples_per_day
        if days < 1:
            raise ValueError(f"weather channel {name!r} has fewer than one day of samples")
        trimmed = arr[: days * samples_per_day]
        grouped = trimmed.reshape(days, samples_per_day, arr.shape[1], arr.shape[2])
        if name in DAILY_SUM_CHANNELS:
            result[name] = grouped.sum(axis=1).astype(np.float32)
        else:
            result[name] = grouped.mean(axis=1).astype(np.float32)
    return result


def noaa_nomads_gfs_filter_url(
    *,
    date: str,
    cycle: str,
    forecast_hour: int,
    bbox: Iterable[float],
) -> str:
    """Build a NOMADS GFS 0.25deg GRIB filter URL for required surface variables."""
    west, south, east, north = [float(v) for v in bbox]
    fh = int(forecast_hour)
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?file=gfs.t{cycle}z.pgrb2.0p25.f{fh:03d}"
        "&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_surface=on"
        "&var_UGRD=on&var_VGRD=on&var_GUST=on&var_TMP=on&var_SPFH=on&var_APCP=on"
        f"&subregion=&leftlon={west}&rightlon={east}&toplat={north}&bottomlat={south}"
        f"&dir=%2Fgfs.{date}%2F{cycle}%2Fatmos"
    )
