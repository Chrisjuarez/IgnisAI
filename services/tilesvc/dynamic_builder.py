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
    """Return list of FIRMS point dicts with lon/lat/timestamp/frp metadata."""
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
            try:
                frp = float(row.get("frp") or 0.0)
            except Exception:
                frp = 0.0
            pts.append({
                "lon": lon,
                "lat": lat,
                "ts": ts,
                "frp": max(0.0, frp),
            })
        except Exception:
            continue
    # Deduplicate across products
    uniq = {}
    for pt in pts:
        key = (round(pt["lon"], 5), round(pt["lat"], 5), pt["ts"])
        if key not in uniq:
            uniq[key] = pt
    return list(uniq.values())


def _points_in_window(points, t_start, t_end):
    return [pt for pt in points if t_start <= pt["ts"] < t_end]


def _point_geoms(points, affine):
    pix_m = float(abs(affine.a))              # pixel size from grid (e.g., 500 m)
    buf_m = max(0.6 * pix_m, 250.0)           # at least 250 m footprint
    geoms = []
    for pt in points:
        x, y = lonlat_to_xy_m(pt["lon"], pt["lat"])   # -> 5070 meters
        geoms.append(Point(x, y).buffer(buf_m))
    return geoms


def _rasterize_fire(points, t_start, t_end, affine):
    """
    Rasterize FIRMS detections within [t_start, t_end) into a SIZE×SIZE mask.

    IMPORTANT: Geometries are created in **EPSG:5070 meters** to match `affine`.
    We give each detection a small meter buffer (~0.6 px) so a point covers
    the cell it falls into and immediate neighbors.
    """
    geoms = _point_geoms(_points_in_window(points, t_start, t_end), affine)

    if not geoms:
        return np.zeros((SIZE, SIZE), np.float32)

    return rasterize(
        [(g, 1.0) for g in geoms],
        out_shape=(SIZE, SIZE),
        transform=affine,
        all_touched=True,
        dtype="float32",
    )


def _rasterize_frp(points, t_start, t_end, affine):
    """Rasterize FRP values for detections in [t_start, t_end)."""
    window_points = _points_in_window(points, t_start, t_end)
    geoms = _point_geoms(window_points, affine)
    if not geoms:
        return np.zeros((SIZE, SIZE), np.float32)
    return rasterize(
        [(geom, float(pt.get("frp", 0.0))) for geom, pt in zip(geoms, window_points)],
        out_shape=(SIZE, SIZE),
        transform=affine,
        all_touched=True,
        dtype="float32",
    )


# ---------------- Weather (Open-Meteo) ----------------
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_grids(lat: float, lon: float, ref_time: dt.datetime = None):
    """
    Fetch weather and expand to per-pixel constant grids.
    If ref_time is provided and in the past (>24h ago), use the Open-Meteo Archive API.
    Otherwise use the current/forecast API.
    """
    now = dt.datetime.now(dt.timezone.utc)
    use_archive = (ref_time is not None and (now - ref_time).total_seconds() > 86400)

    cur = {}

    if use_archive:
        date_str = ref_time.strftime("%Y-%m-%d")
        target_hour = ref_time.hour
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation"
            ),
            "windspeed_unit": "ms",
            "precipitation_unit": "mm",
            "temperature_unit": "celsius",
            "timezone": "UTC",
        }
        try:
            j = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=20).json()
            hourly = j.get("hourly", {}) or {}
            times = hourly.get("time", [])
            # Find the closest hour index
            idx = min(target_hour, len(times) - 1) if times else 0
            cur = {
                "temperature_2m": hourly.get("temperature_2m", [15.0])[idx],
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [50.0])[idx],
                "wind_speed_10m": hourly.get("wind_speed_10m", [3.0])[idx],
                "wind_direction_10m": hourly.get("wind_direction_10m", [0.0])[idx],
                "wind_gusts_10m": hourly.get("wind_gusts_10m", [3.0])[idx],
                "precipitation": hourly.get("precipitation", [0.0])[idx],
            }
            print(f"[tilesvc] Archive weather for {date_str} hour {target_hour}: T={cur.get('temperature_2m')}, "
                  f"RH={cur.get('relative_humidity_2m')}, WS={cur.get('wind_speed_10m')}")
        except Exception as e:
            print(f"⚠️  Archive weather fetch failed: {e}")
            cur = {}
    else:
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
        "temp":   np.full((H, W), T, np.float32),
        "tempC":  np.full((H, W), T, np.float32),
        "rh":     np.full((H, W), RH, np.float32),
        "q":      np.full((H, W), q, np.float32),
        "prcp":   np.full((H, W), P, np.float32),
        "precip": np.full((H, W), P, np.float32),
    }


# ---------------- Public API ----------------
DEFAULT_DYNAMIC_ORDER = ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]


def build_dynamic_for_tile(
    lat: float,
    lon: float,
    T_seq: int = 1,
    hours_step: int = 24,
    ignition: bool = False,
    ref_time: dt.datetime = None,
    channel_order=None,
):
    """
    Build dynamic tensor for the tile containing (lat,lon).

    Returns `x_dyn` with shape [T, 7, SIZE, SIZE]
    default channels = [fire_t, u, v, gust, tempC, q, precip]

    If ref_time is provided, uses historical weather from that date.
    FIRMS data is only available for recent dates (1-5 days); for older dates
    the fire channel will be empty unless ignition=True.
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
    now = ref_time or dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    if ref_time:
        print(f"[tilesvc] Using historical ref_time: {ref_time.isoformat()}")
    fire_stack = []
    frp_stack = []
    for k in range(T_seq, 0, -1):
        t_end = now - dt.timedelta(hours=(k - 1) * hours_step)
        t_sta = t_end - dt.timedelta(hours=hours_step)
        m = _rasterize_fire(points, t_sta, t_end, A)
        frp_m = _rasterize_frp(points, t_sta, t_end, A)
        # match train-time boost so sparse/high-FRP points carry signal
        fire_stack.append(fire_boost(m, frp_m, scale=5.0))
        frp_stack.append(frp_m)
    fire_stack = np.stack(fire_stack, axis=0).astype(np.float32)  # [T,H,W]
    frp_stack = np.stack(frp_stack, axis=0).astype(np.float32)    # [T,H,W]

    if ignition:
        ign = _rasterize_ignition_point(lat, lon, A)      # [H,W] in 5070 meters
        ign = fire_boost(ign, scale=5.0)
        # Inject into the most recent timestep (last index)
        fire_stack[-1] = np.maximum(fire_stack[-1], ign).astype(np.float32)
    # Weather (constant over the tile for now; uses archive for historical dates)
    # Assemble dynamic tensor
    dyn = []
    order = list(channel_order or DEFAULT_DYNAMIC_ORDER)
    for t in range(T_seq):
        # Use the end of each slice as the weather timestamp for that timestep.
        step_ref_time = now - dt.timedelta(hours=(T_seq - 1 - t) * hours_step)
        wx = fetch_weather_grids(lat, lon, ref_time=step_ref_time)
        channels = {
            "fire_t": fire_stack[t],
            "frp": frp_stack[t],
            "u": wx["u"],
            "v": wx["v"],
            "gust": wx["gust"],
            "temp": wx["temp"],
            "tempC": wx["tempC"],
            "rh": wx["rh"],
            "q": wx["q"],
            "prcp": wx["prcp"],
            "precip": wx["precip"],
        }
        missing = [name for name in order if name not in channels]
        if missing:
            raise KeyError(f"Unsupported dynamic channel(s): {missing}; available={sorted(channels.keys())}")
        dyn.append(np.stack([channels[name] for name in order], axis=0))
    x_dyn = np.stack(dyn, axis=0).astype(np.float32)              # [T,7,H,W]
    return x_dyn
