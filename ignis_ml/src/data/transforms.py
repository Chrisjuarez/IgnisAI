#ignis_ml/src/data/transforms.py
import numpy as np

def wind_to_uv(speed_ms, dir_deg):
    """
    Convert meteorological wind (speed m/s, direction° FROM which it blows)
    to (u,v) components in m/s.
    """
    a = np.deg2rad(dir_deg)
    u = -np.sin(a) * speed_ms
    v = -np.cos(a) * speed_ms
    return u, v

def rh_to_q(rh_pct, temp_c, pressure_hpa=1013.25):
    """
    Convert RH% and temperature (°C) to specific humidity q (kg/kg)
    using Magnus formula for saturation vapor pressure.
    """
    T = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0, 100)
    p = float(pressure_hpa)

    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))      # hPa
    e  = (rh / 100.0) * es                               # hPa
    w  = 0.622 * e / np.maximum(p - e, 1e-6)             # mixing ratio
    q  = w / (1.0 + w)
    return q

def fire_boost(*args, **kwargs):
    """
    Safe, flexible booster. Accepts (fire) or (fire, frp, scale=...).
    If FRP provided, add a normalized FRP term; otherwise return fire.
    """
    fire = np.asarray(args[0], dtype=float) if args else None
    if fire is None:
        return None
    frp = np.asarray(args[1], dtype=float) if len(args) > 1 else None
    scale = float(kwargs.get('scale', 1.0))
    if frp is None:
        return fire
    p5, p95 = np.nanpercentile(frp, 5), np.nanpercentile(frp, 95)
    frp_n = np.clip((frp - p5) / (p95 - p5 + 1e-9), 0, 1)
    return np.clip(fire + scale * frp_n, 0, 1)
