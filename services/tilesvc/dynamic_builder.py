# services/tilesvc/dynamic_builder.py
import os
import io
import csv
import math
import datetime as dt
import numpy as np
import requests
from shapely.geometry import Point
from rasterio.features import rasterize

from .grid import SIZE, lonlat_to_tile, tile_affine, tile_bounds_lonlat, lonlat_to_xy_m
from ignis_ml.src.data.transforms import wind_to_uv, rh_to_q, fire_boost

NASA_KEY   = (os.getenv("NASA_API_KEY") or "").strip()  # FIRMS “area” API key
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Products to merge for better coverage (NRT = near‑real‑time)
FIRMS_PRODUCTS = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",  # Terra + Aqua combined
]

# FIRMS “area” API accepts days=1..5 (not hours). Default to 2 days (~48h) to catch late posts.
FIRMS_DAYS_DEFAULT = int(os.getenv("FIRMS_DAYS", "2"))  # clamp to 1..5 later
FIRMS_PAD_DEG = float(os.getenv("FIRMS_PAD_DEG", "0.25"))  # expand bbox to catch edge detections


# ---------------- FIRMS helpers ----------------
def _rasterize_ignition_point(lat: float, lon: float, affine):
    pix_m = float(abs(affine.a))
    buf_m = max(0.8 * pix_m, 300.0)
    x, y = lonlat_to_xy_m(lon, lat)
    geom = Point(x, y).buffer(buf_m)
    return rasterize(
        [(geom, 1.0)],
        out_shape=(SIZE, SIZE),
        transform=affine,
        all_touched=True,
        dtype="float32",
    )

def _fetch_firms_csv(bbox, days=None) -> str:
    """
    bbox=(W,S,E,N). Merge multiple FIRMS NRT products. Returns CSV text.
    FIRMS “area” API expects days in [1..5]; we clamp to that range.
    """
    if not NASA_KEY:
        return ""

    days = days or FIRMS_DAYS_DEFAULT
    days = max(1, min(int(days), 5))

    w, s, e, n = bbox
    w -= FIRMS_PAD_DEG
    s -= FIRMS_PAD_DEG
    e += FIRMS_PAD_DEG
    n += FIRMS_PAD_DEG
    header_line = None
    bodies = []
    for prod in FIRMS_PRODUCTS:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/"
            f"{prod}/{w},{s},{e},{n}/{days}"
        )
        try:
            r = requests.get(url, headers={"User-Agent": "ignis-ai"}, timeout=20)
            r.raise_for_status()
            lines = r.text.splitlines()
            if len(lines) > 1:
                if header_line is None:
                    header_line = lines[0]
                bodies.append("\n".join(lines[1:]))  # drop header for merge
        except Exception as e:
            print(f"⚠️  FIRMS fetch failed for {prod}: {e}")

    if not bodies:
        return ""
    header_line = header_line or "latitude,longitude,acq_date,acq_time"
    return header_line + "\n" + "\n".join(bodies)


def _parse_firms_points(csv_text: str):
    """Return list of (lon, lat, timestamp_utc)."""
    pts = []
    if not csv_text.strip():
        return pts
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
    # Deduplicate across products
    uniq = {}
    for lon, lat, ts in pts:
        key = (round(lon, 5), round(lat, 5), ts)
        if key not in uniq:
            uniq[key] = (lon, lat, ts)
    return list(uniq.values())


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
    q = rh_to_q(np.array(RH, np.float32), np.array(T, np.float32))


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
def build_dynamic_for_tile(lat: float, lon: float, T_seq: int = 1, hours_step: int = 24, ignition: bool = False):
    """
    Build dynamic tensor for the tile containing (lat,lon).

    Returns `x_dyn` with shape [T, 7, SIZE, SIZE]
    channels = [fire_t, u, v, gust, tempC, q, precip]
    """
    tile = lonlat_to_tile(lon, lat)
    A    = tile_affine(tile)
    bbox = tile_bounds_lonlat(tile)

    # FIRMS points over the full [T_seq * hours_step] window
    window_hours = T_seq * hours_step
    days = math.ceil(window_hours / 24.0)
    csv_text = _fetch_firms_csv(bbox, days=days)
    points   = _parse_firms_points(csv_text)
    print(f"[tilesvc] FIRMS points for tile {tile}: {len(points)} over {days}d window")

    # Build T slices: newest last
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    fire_stack = []
    for k in range(T_seq, 0, -1):
        t_end = now - dt.timedelta(hours=(k - 1) * hours_step)
        t_sta = t_end - dt.timedelta(hours=hours_step)
        m = _rasterize_fire(points, t_sta, t_end, A)
        # match train-time boost so sparse points carry signal
        fire_stack.append(fire_boost(m, scale=5.0))
    fire_stack = np.stack(fire_stack, axis=0).astype(np.float32)  # [T,H,W]
    
    if ignition:
        ign = _rasterize_ignition_point(lat, lon, A)      # [H,W] in 5070 meters
        ign = fire_boost(ign, scale=5.0)                  # match train-time boost
        # Inject into the most recent timestep (last index)
        fire_stack[-1] = np.maximum(fire_stack[-1], ign).astype(np.float32)
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
