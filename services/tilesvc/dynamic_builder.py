# services/tilesvc/dynamic_builder.py
import os
import io
import csv
import datetime as dt
import numpy as np
import requests
from shapely.geometry import Point
from rasterio.features import rasterize

from .grid import SIZE, lonlat_to_tile, tile_affine, tile_bounds_lonlat, lonlat_to_xy_m
from ignis_ml.src.data.transforms import wind_to_uv, rh_to_q, fire_boost

NASA_KEY   = os.getenv("NASA_API_KEY")  # FIRMS “area” API key
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


# ---------------- FIRMS helpers ----------------
def _fetch_firms_csv(bbox, hours=24) -> str:
    """
    bbox=(W,S,E,N). Uses VIIRS NOAA-21 NRT.
    Returns CSV text (header if key missing).
    """
    if not NASA_KEY:
        return "latitude,longitude,acq_date,acq_time\n"
    w, s, e, n = bbox
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/"
        f"VIIRS_NOAA21_NRT/{w},{s},{e},{n}/{int(hours)}"
    )
    r = requests.get(url, headers={"User-Agent": "ignis-ai"}, timeout=20)
    r.raise_for_status()
    return r.text


def _parse_firms_points(csv_text: str):
    """Return list of (lon, lat, timestamp_utc)."""
    pts = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            ts = dt.datetime.strptime(
                row["acq_date"] + row["acq_time"].zfill(4), "%Y-%m-%d%H%M"
            ).replace(tzinfo=dt.timezone.utc)
            pts.append((lon, lat, ts))
        except Exception:
            continue
    return pts


def _rasterize_fire(points, t_start, t_end, affine):
    """
    Rasterize FIRMS detections within [t_start, t_end) into a SIZE×SIZE mask.

    IMPORTANT: Geometries are created in **EPSG:5070 meters** to match `affine`.
    We give each detection a small meter buffer (~0.6 px) so a point covers
    the cell it falls into and immediate neighbors.
    """
    pix_m = float(abs(affine.a))              # pixel size from grid (e.g., 500 m)
    buf_m = max(0.6 * pix_m, 250.0)           # at least 250 m footprint

    geoms = []
    for lon, lat, ts in points:
        if t_start <= ts < t_end:
            x, y = lonlat_to_xy_m(lon, lat)   # -> 5070 meters
            geoms.append(Point(x, y).buffer(buf_m))

    if not geoms:
        return np.zeros((SIZE, SIZE), np.float32)

    return rasterize(
        [(g, 1.0) for g in geoms],
        out_shape=(SIZE, SIZE),
        transform=affine,
        all_touched=True,
        dtype="float32",
    )


# ---------------- Weather (Open-Meteo) ----------------
def _fetch_weather(lat: float, lon: float):
    """
    Fetch current weather and expand to per-pixel constant grids.

    We request **m/s** winds and °C so they match training-time transforms.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation"
        ),
        "windspeed_unit": "ms",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    try:
        j = requests.get(OPEN_METEO, params=params, timeout=12).json()
        cur = j.get("current", {}) or {}
    except Exception:
        cur = {}

    # Fallbacks if anything missing
    T  = float(cur.get("temperature_2m",       15.0))  # °C
    RH = float(cur.get("relative_humidity_2m", 50.0))  # %
    WS = float(cur.get("wind_speed_10m",        3.0))  # m/s
    WD = float(cur.get("wind_direction_10m",    0.0))  # deg
    G  = float(cur.get("wind_gusts_10m",       WS))    # m/s
    P  = float(cur.get("precipitation",         0.0))  # mm

    # Convert to model inputs
    u, v = wind_to_uv(np.array(WS, np.float32), np.array(WD, np.float32))
    q    = rh_to_q(np.array(T,  np.float32),    np.array(RH, np.float32))

    H = W = SIZE
    return {
        "u":      np.full((H, W), u, np.float32),
        "v":      np.full((H, W), v, np.float32),
        "gust":   np.full((H, W), G, np.float32),
        "tempC":  np.full((H, W), T, np.float32),
        "q":      np.full((H, W), q, np.float32),
        "precip": np.full((H, W), P, np.float32),
    }


# ---------------- Public API ----------------
def build_dynamic_for_tile(lat: float, lon: float, T_seq: int = 3, hours_step: int = 24):
    """
    Build dynamic tensor for the tile containing (lat,lon).

    Returns `x_dyn` with shape [T, 7, SIZE, SIZE]
    channels = [fire_t, u, v, gust, tempC, q, precip]
    """
    tile = lonlat_to_tile(lon, lat)
    A    = tile_affine(tile)
    bbox = tile_bounds_lonlat(tile)

    # FIRMS points over the full [T_seq * hours_step] window
    csv_text = _fetch_firms_csv(bbox, hours=T_seq * hours_step)
    points   = _parse_firms_points(csv_text)

    # Build T slices: newest last
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    fire_stack = []
    for k in range(T_seq, 0, -1):
        t_end = now - dt.timedelta(hours=(k - 1) * hours_step)
        t_sta = t_end - dt.timedelta(hours=hours_step)
        m = _rasterize_fire(points, t_sta, t_end, A)
        # match train-time boost so sparse points carry signal
        fire_stack.append(fire_boost(m, factor=5.0))
    fire_stack = np.stack(fire_stack, axis=0).astype(np.float32)  # [T,H,W]

    # Weather (constant over the tile for now)
    wx = _fetch_weather(lat, lon)

    # Assemble dynamic tensor
    dyn = []
    for t in range(T_seq):
        dyn.append(np.stack([
            fire_stack[t],
            wx["u"], wx["v"], wx["gust"],
            wx["tempC"], wx["q"], wx["precip"]
        ], axis=0))
    x_dyn = np.stack(dyn, axis=0).astype(np.float32)              # [T,7,H,W]
    return x_dyn