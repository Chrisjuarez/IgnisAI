# services/tilesvc/dynamic_builder.py
import os
import io
import csv
import math
import time
import threading
import datetime as dt
from pathlib import Path
import numpy as np
import requests
from shapely.geometry import Point
from rasterio.features import rasterize

from .grid import SIZE, lonlat_to_tile, tile_affine, tile_bounds_lonlat, lonlat_to_xy_m
from .static_catalog import InputUnavailable
from ignis_ml.src.data.transforms import wind_to_uv, rh_to_q, fire_boost

NASA_KEY   = (os.getenv("NASA_API_KEY") or "").strip()  # FIRMS “area” API key
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# ---------------- Tiny TTL caches ----------------
# Multistep predictions call FIRMS once per request but the weather fetch runs
# T_seq times per request, and users often click the same tile repeatedly.
# Share results between calls for a short window to avoid duplicating external
# HTTP round trips (which were a root cause of tilesvc OOMs under bursty load).
_FIRMS_TTL_SEC    = int(os.getenv("FIRMS_CACHE_TTL",   "300"))  # 5 min
_WEATHER_TTL_SEC  = int(os.getenv("WEATHER_CACHE_TTL", "600"))  # 10 min
_CACHE_MAX_ENTRIES = 256

_firms_cache: dict = {}
_weather_cache: dict = {}
_cache_lock = threading.Lock()
_LAST_WEATHER_SOURCE = "unknown"
_LAST_WEATHER_REASON = "not_requested"


def _set_weather_source(source: str, reason: str = ""):
    global _LAST_WEATHER_SOURCE, _LAST_WEATHER_REASON
    _LAST_WEATHER_SOURCE = source
    _LAST_WEATHER_REASON = reason


def weather_quality_status():
    ok = _LAST_WEATHER_SOURCE == "noaa_gridded"
    return {
        "status": "ok" if ok else "degraded",
        "source": _LAST_WEATHER_SOURCE,
        "reason": _LAST_WEATHER_REASON,
    }


def _cache_get(store: dict, key, ttl: int):
    with _cache_lock:
        entry = store.get(key)
    if not entry:
        return None
    ts, val = entry
    if ttl > 0 and (time.time() - ts) > ttl:
        with _cache_lock:
            store.pop(key, None)
        return None
    return val


def _cache_set(store: dict, key, value):
    with _cache_lock:
        if len(store) >= _CACHE_MAX_ENTRIES:
            # Drop the oldest entry to keep memory bounded.
            oldest_key = min(store, key=lambda k: store[k][0])
            store.pop(oldest_key, None)
        store[key] = (time.time(), value)


def _as_grid(value, *, name: str):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (SIZE, SIZE):
        raise ValueError(f"NOAA cache channel {name!r} has shape {arr.shape}, expected {(SIZE, SIZE)}")
    return arr.astype(np.float32)


def _noaa_cache_path(lat: float, lon: float, ref_time: dt.datetime | None) -> Path | None:
    cache_dir = os.getenv("NOAA_GRID_CACHE_DIR")
    template = os.getenv("NOAA_GRID_CACHE_TEMPLATE")
    if not cache_dir and not template:
        return None
    t = ref_time or dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    values = {
        "date": t.strftime("%Y%m%d"),
        "hour": f"{t.hour:02d}",
        "iso_hour": t.strftime("%Y%m%dT%H"),
        "lat": f"{float(lat):.2f}",
        "lon": f"{float(lon):.2f}",
    }
    if template:
        return Path(template.format(**values))
    return Path(cache_dir) / f"{values['iso_hour']}.npz"


def _fetch_noaa_cached_weather_grids(lat: float, lon: float, ref_time: dt.datetime | None):
    if os.getenv("NOAA_GRIB_ENABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    path = _noaa_cache_path(lat, lon, ref_time)
    if not path or not path.exists():
        _set_weather_source("open_meteo_fallback", f"noaa_cache_missing:{path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            grids = {
                "u": _as_grid(data["u"], name="u"),
                "v": _as_grid(data["v"], name="v"),
                "gust": _as_grid(data["gust"], name="gust"),
                "temp": _as_grid(data["tempC"], name="tempC"),
                "tempC": _as_grid(data["tempC"], name="tempC"),
                "q": _as_grid(data["q"], name="q"),
                "precip": _as_grid(data["precip"], name="precip"),
            }
            grids["rh"] = np.full((SIZE, SIZE), np.nan, np.float32)
            grids["prcp"] = grids["precip"]
        _set_weather_source("noaa_gridded", str(path))
        return grids
    except Exception as exc:
        _set_weather_source("open_meteo_fallback", f"noaa_cache_invalid:{exc}")
        print(f"⚠️  NOAA weather cache failed ({path}): {exc}")
        return None

# Products to merge for better coverage (NRT = near‑real‑time)
FIRMS_PRODUCTS = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",  # Terra + Aqua combined
]

# FIRMS “area” API accepts days=1..5 (not hours). Default to 2 days (~48h) to catch late posts.
FIRMS_DAYS_DEFAULT = int(os.getenv("FIRMS_DAYS", "2"))  # clamp to 1..5 later
FIRMS_PAD_DEG = float(os.getenv("FIRMS_PAD_DEG", "0.25"))  # expand bbox to catch edge detections
FIRMS_SNAPSHOT_DIR = os.getenv("FIRMS_SNAPSHOT_DIR")
FIRMS_SNAPSHOT_REQUIRED = os.getenv("FIRMS_SNAPSHOT_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _firms_snapshot_dir() -> str:
    return os.getenv("FIRMS_SNAPSHOT_DIR") or FIRMS_SNAPSHOT_DIR or ""


def _firms_snapshot_required() -> bool:
    return (os.getenv("FIRMS_SNAPSHOT_REQUIRED") or str(FIRMS_SNAPSHOT_REQUIRED)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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

    Cached by (rounded bbox, days) for FIRMS_CACHE_TTL seconds so a storm of
    clicks on the same tile doesn't hammer the NASA API (and burn tilesvc RAM
    on redundant CSV parses).
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

    # Round to ~1 km so re-clicks on the same tile hit the cache.
    cache_key = (
        round(w, 3), round(s, 3), round(e, 3), round(n, 3), int(days),
    )
    cached = _cache_get(_firms_cache, cache_key, _FIRMS_TTL_SEC)
    if cached is not None:
        return cached

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
        result = ""
    else:
        header_line = header_line or "latitude,longitude,acq_date,acq_time"
        result = header_line + "\n" + "\n".join(bodies)

    _cache_set(_firms_cache, cache_key, result)
    return result


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


def _daily_snapshot_path(day: dt.date) -> Path | None:
    snapshot_dir = _firms_snapshot_dir()
    if not snapshot_dir:
        return None
    root = Path(snapshot_dir)
    for suffix in (".csv", ".CSV"):
        candidate = root / f"{day.isoformat()}{suffix}"
        if candidate.exists():
            return candidate
    return root / f"{day.isoformat()}.csv"


def _load_firms_points_from_snapshots(bbox, t_start: dt.datetime, t_end: dt.datetime):
    snapshot_dir = _firms_snapshot_dir()
    if not snapshot_dir:
        return None

    points = []
    missing = []
    day = t_start.date()
    while day <= t_end.date():
        path = _daily_snapshot_path(day)
        if not path or not path.exists():
            missing.append(day.isoformat())
        else:
            try:
                points.extend(_parse_firms_points(path.read_text(encoding="utf-8")))
            except Exception as exc:
                raise InputUnavailable(
                    f"Failed to read FIRMS snapshot {path}: {exc}",
                    reason="firms_snapshot_read_failed",
                    details={"path": str(path), "error": str(exc)},
                ) from exc
        day = day + dt.timedelta(days=1)

    if missing and _firms_snapshot_required():
        raise InputUnavailable(
            "Persisted FIRMS daily snapshots are missing for the requested window",
            reason="firms_snapshot_missing",
            details={"missing_dates": missing, "snapshot_dir": snapshot_dir},
        )
    if missing:
        print(f"⚠️  FIRMS snapshots missing dates {missing}; falling back to live FIRMS API")
        return None

    w, s, e, n = bbox
    filtered = [
        pt for pt in points
        if w <= pt["lon"] <= e and s <= pt["lat"] <= n and t_start <= pt["ts"] <= t_end
    ]
    return filtered


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

    Cached by (rounded lat/lon, ref hour, archive vs forecast) for
    WEATHER_CACHE_TTL seconds. Each multistep prediction calls this once per
    timestep (T=6), so this alone removes ~5 duplicate HTTP round trips per
    request.
    """
    noaa_grids = _fetch_noaa_cached_weather_grids(lat, lon, ref_time)
    if noaa_grids is not None:
        return {k: v.copy() for k, v in noaa_grids.items()}

    now = dt.datetime.now(dt.timezone.utc)
    use_archive = (ref_time is not None and (now - ref_time).total_seconds() > 86400)

    # Cache key: nearest 0.05° (~5 km) and hour bucket. Open-Meteo archive is
    # stable; forecast rotates hourly, so hour bucket gives us a natural TTL
    # without pinning to `ref_time` to the minute.
    lat_q = round(float(lat), 2)
    lon_q = round(float(lon), 2)
    if ref_time is not None:
        hour_bucket = ref_time.replace(minute=0, second=0, microsecond=0).isoformat()
    else:
        hour_bucket = now.replace(minute=0, second=0, microsecond=0).isoformat()
    cache_key = (lat_q, lon_q, hour_bucket, bool(use_archive))

    cached = _cache_get(_weather_cache, cache_key, _WEATHER_TTL_SEC)
    if cached is not None:
        _set_weather_source(
            "open_meteo_fallback",
            "archive_cache" if use_archive else "forecast_current_cache",
        )
        # Return defensive copies so callers can't mutate cached arrays.
        return {k: v.copy() for k, v in cached.items()}

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
    grids = {
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
    _set_weather_source(
        "open_meteo_fallback",
        "archive" if use_archive else "forecast_current",
    )
    _cache_set(_weather_cache, cache_key, {k: v.copy() for k, v in grids.items()})
    return grids


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
    now = ref_time or dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    if ref_time:
        print(f"[tilesvc] Using historical ref_time: {ref_time.isoformat()}")

    # FIRMS points over the full [T_seq * hours_step] window
    window_hours = T_seq * hours_step
    window_start = now - dt.timedelta(hours=window_hours)
    days = math.ceil(window_hours / 24.0)
    points = _load_firms_points_from_snapshots(bbox, window_start, now)

    # The live FIRMS NRT API only covers ~5 days. Falling through silently
    # for older windows produces an empty fire channel — the model then has
    # no idea where the fire is and predicts on weather/fuel climatology
    # only. Make the gap loud so the caller can decide whether to seed via
    # `ignition=True` or go fix the snapshot directory.
    nowish = dt.datetime.now(dt.timezone.utc)
    window_age_days = max(0, (nowish - now).days)
    FIRMS_LIVE_API_MAX_DAYS = 5

    if points is None:
        if window_age_days > FIRMS_LIVE_API_MAX_DAYS:
            print(
                f"⚠️  [tilesvc] FIRMS window is {window_age_days}d old (>{FIRMS_LIVE_API_MAX_DAYS}d); "
                f"live NRT API will return empty. snapshot_dir={_firms_snapshot_dir() or '<unset>'}; "
                "set FIRMS_SNAPSHOT_DIR + FIRMS_SNAPSHOT_REQUIRED=1 and run "
                "services/runtime_cache/__main__.py to backfill."
            )
        csv_text = _fetch_firms_csv(bbox, days=days)
        points = _parse_firms_points(csv_text)
        if not points and window_age_days > FIRMS_LIVE_API_MAX_DAYS:
            print(
                f"⚠️  [tilesvc] FIRMS empty for {window_start.isoformat()}..{now.isoformat()} "
                "(historical window, no snapshot, live API empty). The fire channel will be "
                "all-zeros unless the caller passes ignition=True or a seeded perimeter."
            )
        print(f"[tilesvc] FIRMS points for tile {tile}: {len(points)} over {days}d live/API window")
    else:
        print(f"[tilesvc] FIRMS points for tile {tile}: {len(points)} from persisted daily snapshots")

    # Build T slices: newest last
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
