"""Score every spread engine on the same fires - live incidents, not just Palisades.

Palisades is atypical: an ignition-time forecast, a coastal tile that is
two-thirds unburnable, and one extreme wind event. A live incident has an
irregular multi-day footprint in wildland under turning weather. An engine
tuned on the first tells you nothing about the second.
"""
import os, sys, math, datetime as dt
import numpy as np
sys.path.insert(0, "/Users/chrisjuarez/Downloads/IgnisAI")
_R = "/Users/chrisjuarez/Downloads/IgnisAI"
os.environ.setdefault("NOAA_GRIB_ENABLED", "1")

from services.tilesvc.grid import lonlat_to_tile, SIZE, PIX
from services.tilesvc.dynamic_builder import build_dynamic_for_tile
from services.tilesvc.fuel_raster import fuel_codes_for_tile
from services.tilesvc.wui_fuels import apply_wui_fuels, wui_summary
from services.tilesvc.baseline_spread import baseline_rollout
from services.tilesvc.physics_spread import physics_rollout
from services.tilesvc.pyretechnics_spread import pyretechnics_rollout
from services.tilesvc.fuel_models import lookup
from tools.audit_direction import live_incidents

ORDER = ["fire_t", "u", "v", "gust", "tempC", "q", "precip"]

def bearing(mask, origin_rc):
    r, c = np.nonzero(mask)
    if r.size == 0: return None, 0
    return ((math.degrees(math.atan2(c.mean()-origin_rc[1], -(r.mean()-origin_rc[0])))+360)%360,
            r.size)

def score(name, lat, lon, ref_time=None):
    tile = lonlat_to_tile(lon, lat)
    x = np.asarray(build_dynamic_for_tile(lat, lon, T_seq=6, hours_step=24, ignition=True,
                                          ref_time=ref_time, channel_order=ORDER), dtype=np.float32)
    obs = (x[-1, 0] > 0.5).astype(np.float32)
    if obs.sum() == 0:
        print("  %-22s no observed fire in tile" % name[:22]); return
    rr, cc = np.nonzero(obs)
    origin = (rr.mean(), cc.mean())
    # Wind per step from the last frames, so the series turns as the data does.
    series = [(float(x[i, 1].mean()), float(x[i, 2].mean())) for i in (-3, -2, -1)]
    u, v = series[-1]
    wind = (math.degrees(math.atan2(u, v)) + 360) % 360
    codes = fuel_codes_for_tile(tile)
    burn = np.array([[lookup(int(q)) is not None for q in row] for row in codes])

    runs = {
        "downwind":    baseline_rollout(obs, u_ms=u, v_ms=v, steps=3, step_hours=24,
                                        ignition_rc=(SIZE//2, SIZE//2)),
        "rothermel":   physics_rollout(obs, fuel_codes=codes, u_ms=u, v_ms=v, steps=3,
                                       step_hours=24, ignition_rc=(SIZE//2, SIZE//2)),
        "pyretechnics": pyretechnics_rollout(obs, fuel_codes=codes, wind_series=series,
                                             steps=3, step_hours=24),
    }
    print("  %-22s wind->%3.0f  %4.1f m/s  burnable %4.1f%%  obs %d cells" % (
        name[:22], wind, math.hypot(u,v), 100*burn.mean(), int(obs.sum())))
    for eng, roll in runs.items():
        last = roll[-1]["prob"]
        b, n = bearing(last >= (0.5 if eng=="pyretechnics" else 0.1), origin)
        if b is None:
            print("      %-13s no growth" % eng); continue
        off = abs((b - wind + 180) % 360 - 180)
        print("      %-13s day3 %6.1f km2  bearing %3.0f  align %+.2f" % (
            eng, n*(PIX/1000)**2, b, math.cos(math.radians(off))))

print("=== PALISADES (historical, ignition-time, coastal WUI) ===")
os.environ["FIRMS_SNAPSHOT_DIR"] = f"{_R}/.cache/runtime_cache/palisades/firms_snapshots"
os.environ["FIRMS_SNAPSHOT_REQUIRED"] = "1"
os.environ["NOAA_GRID_CACHE_DIR"] = f"{_R}/.cache/runtime_cache/palisades/noaa_grid_cache"
score("palisades", 34.0780, -118.5550, dt.datetime(2025,1,7,18,30,tzinfo=dt.timezone.utc))

print("\n=== LIVE INCIDENTS (burning now, live weather) ===")
for k in ("FIRMS_SNAPSHOT_DIR","FIRMS_SNAPSHOT_REQUIRED","NOAA_GRID_CACHE_DIR"):
    os.environ.pop(k, None)
try:
    for inc in live_incidents(4):
        try:
            score(inc.get("name","?"), inc["lat"], inc["lon"], None)
        except Exception as e:
            print("  %-22s %s: %s" % (str(inc.get('name'))[:22], type(e).__name__, str(e)[:60]))
except Exception as e:
    print("  live incident list unavailable:", type(e).__name__, str(e)[:80])
