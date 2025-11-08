# services/tilesvc/app.py
import os, io, math, base64
from typing import Tuple, Dict, Any
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from .grid import lonlat_to_tile, tile_bounds_lonlat, tile_affine, SIZE, CRS_ALBERS
from .dynamic_builder import build_dynamic_for_tile, _fetch_weather
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer

app = FastAPI(title="Ignis Tilesvc", version="1.0")

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN] if CORS_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _mask_to_png_rgba(mask: np.ndarray, color=(255, 0, 0), alpha=170) -> bytes:
    h, w = mask.shape
    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)
    a = (mask.astype(np.uint8) * alpha).astype(np.uint8)
    r[mask] = color[0]; g[mask] = color[1]; b[mask] = color[2]
    img = np.dstack([r, g, b, a])
    im = Image.fromarray(img, mode="RGBA")
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _wind_bias_dilate(base_mask: np.ndarray, u: float, v: float) -> np.ndarray:
    mask = base_mask.astype(bool)
    speed = float(math.hypot(u, v))
    steps = max(1, min(6, int(round(speed / 2.5))))
    theta = math.atan2(v, u + 1e-9)
    dy, dx = -math.sin(theta), math.cos(theta)
    base = mask.copy()
    for yy in (-1,0,1):
        for xx in (-1,0,1):
            base |= np.roll(np.roll(mask, yy, axis=0), xx, axis=1)
    out = base.copy()
    for k in range(1, steps+1):
        r, c = int(round(k*dy)), int(round(k*dx))
        out |= np.roll(np.roll(base, r, axis=0), c, axis=1)
    return out

def _mask_to_geojson(mask: np.ndarray, affine, simplify=0.0) -> Dict[str, Any]:
    mask_uint8 = (mask.astype(np.uint8) * 1)
    feats = [shape(geom) for geom, val in shapes(mask_uint8, mask=mask_uint8==1, transform=affine) if val == 1]
    if not feats:
        return {"type": "FeatureCollection", "features": []}
    merged = unary_union(feats)
    if simplify > 0:
        merged = merged.simplify(simplify, preserve_topology=True)
    to_ll = Transformer.from_crs(CRS_ALBERS, "EPSG:4326", always_xy=True)
    def _proj(x, y, z=None): return to_ll.transform(x, y)
    if merged.geom_type == "Polygon":
        geoms = [merged]
    elif merged.geom_type == "MultiPolygon":
        geoms = list(merged.geoms)
    else:
        geoms = []
    features = [{
        "type": "Feature",
        "properties": {},
        "geometry": mapping(shp_transform(_proj, g))
    } for g in geoms]
    return {"type": "FeatureCollection", "features": features}

def _naive_predict(lat: float, lon: float, Tseq: int = 3):
    dyn = build_dynamic_for_tile(lat, lon, T_seq=Tseq, hours_step=24)   # [T,7,H,W]
    latest = dyn[-1]
    fire = latest[0] > 0.05
    u = float(np.mean(latest[1])); v = float(np.mean(latest[2]))
    spread_mask = _wind_bias_dilate(fire, u, v)
    wind_speed = float(math.hypot(u, v)) * 3.6
    cover = float(np.mean(fire))
    precip = float(np.mean(latest[6]))
    prob = 0.15 + 0.02*wind_speed + 0.5*cover - 0.08*min(precip, 5.0)
    prob = float(max(0.01, min(0.95, prob)))
    env = {
        "wind_speed": wind_speed,
        "wind_direction": (math.degrees(math.atan2(v, u + 1e-9)) + 360.0) % 360.0,
        "temperature": float(np.mean(latest[4])),
        "humidity": float(np.mean(latest[5])) * 100.0,
        "precipitation": float(np.mean(latest[6])),
        "data_source": "weather_api"
    }
    return spread_mask, env, prob

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/tile-bounds")
def tile_bounds(lat: float = Query(...), lon: float = Query(...)):
    t = lonlat_to_tile(lon, lat)
    bounds = tile_bounds_lonlat(t)
    return {"bounds": list(map(float, bounds))}

@app.get("/predict")
def predict_png(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(3)):
    t = lonlat_to_tile(lon, lat)
    A = tile_affine(t)
    spread_mask, env, prob = _naive_predict(lat, lon, Tseq)
    png_bytes = _mask_to_png_rgba(spread_mask)
    return Response(content=png_bytes, media_type="image/png")

@app.get("/predict_raster")
def predict_raster(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(3)):
    t = lonlat_to_tile(lon, lat)
    bounds = tile_bounds_lonlat(t)
    spread_mask, env, prob = _naive_predict(lat, lon, Tseq)
    png_bytes = _mask_to_png_rgba(spread_mask)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return JSONResponse({"bounds": list(map(float, bounds)), "image_base64": b64})

# IMPORTANT: return a FeatureCollection *directly* for Mapbox setData(...)
@app.get("/predict_geojson")
def predict_geojson(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(3)):
    t = lonlat_to_tile(lon, lat)
    A = tile_affine(t)
    spread_mask, env, prob = _naive_predict(lat, lon, Tseq)
    gj = _mask_to_geojson(spread_mask, A, simplify=100)
    # Attach some useful meta on each feature (safe for Mapbox)
    for f in gj.get("features", []):
        f.setdefault("properties", {})
        f["properties"]["type"] = "spread"
        f["properties"]["probability"] = prob
        f["properties"]["wind_dir"] = env["wind_direction"]
    return JSONResponse(gj)

# Quick diagnostic: see points->mask behavior fast
@app.get("/diag")
def diag(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(3)):
    dyn = build_dynamic_for_tile(lat, lon, T_seq=Tseq, hours_step=24)
    latest = dyn[-1]
    info = {
        "fire_min": float(latest[0].min()),
        "fire_max": float(latest[0].max()),
        "fire_mean": float(latest[0].mean()),
        "u_mean": float(latest[1].mean()),
        "v_mean": float(latest[2].mean()),
        "precip_mean": float(latest[6].mean())
    }
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.tilesvc.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8008")))