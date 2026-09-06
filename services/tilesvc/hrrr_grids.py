# services/tilesvc/hrrr_grids.py
#
# On-demand HRRR weather grids for a prediction tile.
#
# The prebuilt npz cache only covers profiles someone backfilled ahead of time,
# so every other tile fell through to the Open-Meteo point fallback, which
# broadcasts one scalar across the whole 32 km tile. A constant wind field gives
# the model nothing to be directional about. GFS (the backfill source) is barely
# better here: at 0.25 deg a tile spans a little over one grid cell.
#
# HRRR is 3 km, so a tile covers roughly 10x10 real values and terrain-channeled
# flow survives the reprojection. Fields are pulled straight out of the public
# AWS archive using the GRIB2 byte-range index, so we transfer only the six
# records we need instead of the ~130 MB surface file.

from __future__ import annotations

import datetime as dt
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

from .grid import CRS_ALBERS, SIZE, lonlat_to_tile, tile_affine

HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# Channel -> (GRIB variable, level string as it appears in the .idx)
HRRR_BANDS: Dict[str, Tuple[str, str]] = {
    "u": ("UGRD", "10 m above ground"),
    "v": ("VGRD", "10 m above ground"),
    "gust": ("GUST", "surface"),
    "tempC": ("TMP", "2 m above ground"),
    "q": ("SPFH", "2 m above ground"),
    "precip": ("APCP", "surface"),
}

# HRRR publishes about 45-90 min after the cycle hour; give it a wide margin
# before assuming a cycle exists.
PUBLISH_LAG = dt.timedelta(hours=2)
MAX_FORECAST_HOUR = 18


@dataclass(frozen=True)
class IndexRecord:
    variable: str
    level: str
    start: int
    end: Optional[int]

    @property
    def byte_range(self) -> str:
        return f"bytes={self.start}-{'' if self.end is None else self.end - 1}"


def hrrr_sfc_url(cycle: dt.datetime, forecast_hour: int) -> str:
    """Public S3 URL for one HRRR CONUS surface file."""
    return (
        f"{HRRR_S3_BASE}/hrrr.{cycle:%Y%m%d}/conus/"
        f"hrrr.t{cycle.hour:02d}z.wrfsfcf{int(forecast_hour):02d}.grib2"
    )


def parse_grib_index(index_text: str) -> List[IndexRecord]:
    """
    Parse a wgrib2-style .idx listing into byte-addressable records.

    Each line is `msg:offset:d=YYYYMMDDHH:VAR:level:forecast:`. A record runs
    until the next message's offset; the final record runs to end of file.
    """
    offsets: List[Tuple[int, str, str]] = []
    for line in index_text.splitlines():
        fields = line.split(":")
        if len(fields) < 5:
            continue
        try:
            offset = int(fields[1])
        except ValueError:
            continue
        offsets.append((offset, fields[3], fields[4]))

    records: List[IndexRecord] = []
    for idx, (offset, variable, level) in enumerate(offsets):
        end = offsets[idx + 1][0] if idx + 1 < len(offsets) else None
        records.append(IndexRecord(variable=variable, level=level, start=offset, end=end))
    return records


def select_bands(records: Sequence[IndexRecord]) -> Dict[str, IndexRecord]:
    """Pick the first record matching each channel's variable and level."""
    wanted = {name: (variable, level) for name, (variable, level) in HRRR_BANDS.items()}
    found: Dict[str, IndexRecord] = {}
    for record in records:
        for name, (variable, level) in wanted.items():
            if name in found:
                continue
            if record.variable == variable and record.level == level:
                found[name] = record
    return found


def cycle_for_valid_time(
    valid_time: dt.datetime,
    *,
    now: Optional[dt.datetime] = None,
) -> Tuple[dt.datetime, int]:
    """
    Choose the HRRR cycle and forecast hour that best cover `valid_time`.

    Past hours use the cycle one hour earlier at f01: a one-hour lead is the
    shortest that still carries APCP, which does not exist in the f00 analysis.
    Future hours forecast forward from the newest cycle expected to be published.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    valid_time = valid_time.astimezone(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    newest_cycle = (now - PUBLISH_LAG).replace(minute=0, second=0, microsecond=0)

    cycle = valid_time - dt.timedelta(hours=1)
    if cycle <= newest_cycle:
        return cycle, 1

    lead = int((valid_time - newest_cycle).total_seconds() // 3600)
    return newest_cycle, min(max(lead, 1), MAX_FORECAST_HOUR)


def _download_bands(
    url: str,
    bands: Mapping[str, IndexRecord],
    target: Path,
    *,
    session: requests.Session,
    timeout: float,
) -> List[str]:
    """
    Concatenate the requested GRIB records into one local file.

    GRIB messages are self-describing, so a concatenation of them is itself a
    valid GRIB file and rasterio exposes each as one band. Returns the channel
    names in the order they were written: `bands` is keyed by channel but
    ordered by position in the source file, which is not the order of
    HRRR_BANDS, and reading them back positionally is how channels get swapped.
    """
    order: List[str] = []
    with target.open("wb") as handle:
        for name, record in bands.items():
            response = session.get(
                url,
                headers={"Range": record.byte_range, "User-Agent": "ignis-ai-tilesvc"},
                timeout=timeout,
            )
            response.raise_for_status()
            handle.write(response.content)
            order.append(name)
    return order


def _reproject_to_tile(src, band: int, *, lat: float, lon: float) -> np.ndarray:
    from rasterio.warp import Resampling, reproject
    import rasterio

    tile = lonlat_to_tile(lon, lat)
    dst = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(src, band),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs or "EPSG:4326",
        dst_transform=tile_affine(tile),
        dst_crs=CRS_ALBERS,
        resampling=Resampling.bilinear,
        src_nodata=src.nodata,
        dst_nodata=np.nan,
    )
    return dst.astype(np.float32)


def _read_grib_to_tile(path: Path, order: Sequence[str], *, lat: float, lon: float) -> Dict[str, np.ndarray]:
    import rasterio

    arrays: Dict[str, np.ndarray] = {}
    with rasterio.open(path) as src:
        for index, name in enumerate(order, start=1):
            if index > src.count:
                break
            arrays[name] = _reproject_to_tile(src, index, lat=lat, lon=lon)
    return arrays


def _finalize(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert units, fill gaps, and add the aliases the builder expects."""
    if "tempC" in arrays and np.nanmean(arrays["tempC"]) > 150.0:
        arrays["tempC"] = arrays["tempC"] - 273.15

    for name in HRRR_BANDS:
        arr = arrays.get(name)
        if arr is None:
            arrays[name] = np.zeros((SIZE, SIZE), dtype=np.float32)
            continue
        fill = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
        arrays[name] = np.nan_to_num(arr, nan=fill).astype(np.float32)

    arrays["temp"] = arrays["tempC"]
    arrays["prcp"] = arrays["precip"]
    arrays["rh"] = np.full((SIZE, SIZE), np.nan, np.float32)
    return arrays


def fetch_hrrr_tile_grids(
    lat: float,
    lon: float,
    valid_time: dt.datetime,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 20.0,
    now: Optional[dt.datetime] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Fetch HRRR grids for one tile-hour, or None if HRRR cannot serve it.

    Returning None rather than raising keeps this a strictly optional upgrade:
    the caller falls through to whatever it did before.
    """
    session = session or requests.Session()
    cycle, forecast_hour = cycle_for_valid_time(valid_time, now=now)
    url = hrrr_sfc_url(cycle, forecast_hour)

    index = session.get(
        url + ".idx",
        headers={"User-Agent": "ignis-ai-tilesvc"},
        timeout=timeout,
    )
    if index.status_code != 200:
        return None

    bands = select_bands(parse_grib_index(index.text))
    # Wind is the channel that actually steers spread; without it there is no
    # reason to prefer HRRR over the existing fallback.
    if "u" not in bands or "v" not in bands:
        return None

    with tempfile.TemporaryDirectory(prefix="ignis-hrrr-") as tmp:
        target = Path(tmp) / "bands.grib2"
        order = _download_bands(url, bands, target, session=session, timeout=timeout)
        arrays = _read_grib_to_tile(target, order, lat=lat, lon=lon)

    if "u" not in arrays or "v" not in arrays:
        return None
    return _finalize(arrays)


def hrrr_enabled() -> bool:
    return (os.getenv("HRRR_ON_DEMAND", "0").strip().lower() in {"1", "true", "yes", "on"})
