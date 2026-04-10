import os
import io
import re
import base64
from typing import Optional, Tuple, Dict, Any

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from scipy.ndimage import gaussian_filter
from pyproj import Transformer

# Your helpers
from .grid import (
    build_grid,
    raster_to_png_bytes,
    lonlat_to_tile,
    tile_bounds_albers,
    lonlat_to_xy_m,
    PIX,
)
from .dynamic_builder import build_dynamic_for_tile
from .static_builder import build_static_for_tile, load_static_for_tile, CHANNEL_ORDER

import rasterio
from rasterio.features import shapes as rio_shapes

import torch
from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet


app = FastAPI(title="Ignis Tilesvc", version="1.0")

_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

def _get_torch_device() -> torch.device:
    d = os.getenv("TORCH_DEVICE", "cpu").lower().strip()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_torch_device()
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pt")
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.1"))

MODEL_CD = int(os.getenv("MODEL_CD", "7"))
MODEL_CS = int(os.getenv("MODEL_CS", "15"))
MODEL_HIDDEN = int(os.getenv("MODEL_HIDDEN", "128"))
MODEL_LSTM_LAYERS = int(os.getenv("MODEL_LSTM_LAYERS", "1"))
PRED_SMOOTH_SIGMA = float(os.getenv("PRED_SMOOTH_SIGMA", "1.5"))
PRED_UPSCALE = int(os.getenv("PRED_UPSCALE", "6"))

_model = None


def _try_parse_from_filename(path: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    fname = os.path.basename(path)
    m = re.search(r"_Cd(\d+)_Cs(\d+)_H(\d+)_T(\d+)_", fname)
    if not m:
        return None, None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _extract_bounds_from_grid(grid: Any, lat: float, lon: float) -> Tuple[float, float, float, float]:
    """
    Prefer bounds directly from grid if available; otherwise fall back to a small bbox around (lat,lon).

    Supports both dicts (legacy style) and the tuple returned by build_grid.
    """
    # tuple form from build_grid: (tile, affine, bounds)
    if isinstance(grid, (tuple, list)) and len(grid) == 3:
        b = grid[2]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            w, s, e, n = map(float, b)
            return (w, s, e, n)

    if isinstance(grid, dict):
        for key in ("bounds", "bounds_lonlat"):
            if key in grid and grid[key] is not None:
                b = grid[key]
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    w, s, e, n = map(float, b)
                    return (w, s, e, n)

        if all(k in grid for k in ("W", "S", "E", "N")):
            return (float(grid["W"]), float(grid["S"]), float(grid["E"]), float(grid["N"]))

    # fallback: ~0.35 deg box
    half = 0.35
    return (lon - half, lat - half, lon + half, lat + half)


def _load_model_once():
    global _model, MODEL_CD, MODEL_CS, MODEL_HIDDEN, MODEL_LSTM_LAYERS

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  MODEL_PATH does not exist: {MODEL_PATH}")
        _model = None
        return None

    cd, cs, h, t = _try_parse_from_filename(MODEL_PATH)
    if cd and os.getenv("MODEL_CD") is None:
        MODEL_CD = cd
    if cs and os.getenv("MODEL_CS") is None:
        MODEL_CS = cs
    if h and os.getenv("MODEL_HIDDEN") is None:
        MODEL_HIDDEN = h
    

    print(f"🔥 Loading model from: {MODEL_PATH}")
    print(
        f"   config: Cd={MODEL_CD}, Cs={MODEL_CS}, hidden={MODEL_HIDDEN}, "
        f"LSTM_LAYERS={MODEL_LSTM_LAYERS}, device={DEVICE}"
    )

    model = ConvLSTMUNet(
        Cd=MODEL_CD,
        Cs=MODEL_CS,
        hidden=MODEL_HIDDEN,
        lstm_layers=MODEL_LSTM_LAYERS,
    )

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(DEVICE)
    elif isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        model.to(DEVICE)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    model.eval()
    _model = model
    return _model


def _predict_probability(lat: float, lon: float, Tseq: int, ignition: bool = False) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Returns:
      prob (H,W) in [0,1]
      bounds (W,S,E,N) in lon/lat
    """
    model = _load_model_once()

    # build_grid returns (tile, affine, bounds)
    tile, _, bounds = build_grid(lat, lon)

    dyn = build_dynamic_for_tile(lat, lon, T_seq=Tseq, ignition=ignition)

    # Load per-tile static rasters and stack into channel-first array.
    stat_dict = load_static_for_tile(tile)
    try:
        stat = np.stack([np.asarray(stat_dict[k], np.float32) for k in CHANNEL_ORDER], axis=0)
    except KeyError as e:
        missing = str(e)
        raise RuntimeError(f"Static channel missing: {missing}; have keys={list(stat_dict.keys())}")

    if model is None:
        H, W = dyn.shape[-2], dyn.shape[-1]
        return np.zeros((H, W), dtype=np.float32), bounds

    if dyn.shape[1] != MODEL_CD:
        raise RuntimeError(f"Dynamic channels mismatch: got {dyn.shape[1]} expected {MODEL_CD}")
    if stat.shape[0] != MODEL_CS:
        raise RuntimeError(f"Static channels mismatch: got {stat.shape[0]} expected {MODEL_CS}")

    x_dyn = torch.from_numpy(dyn).unsqueeze(0).to(DEVICE)    # (1, T, Cd, H, W)
    x_stat = torch.from_numpy(stat).unsqueeze(0).to(DEVICE)  # (1, Cs, H, W)

    with torch.no_grad():
        logits = model(x_dyn, x_stat)  # (1,1,H,W)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    # Smooth to reduce checkerboard artifacts in overlays
    if PRED_SMOOTH_SIGMA > 0:
        try:
            prob = gaussian_filter(prob, sigma=PRED_SMOOTH_SIGMA)
        except Exception:
            pass
    prob = np.clip(prob, 0.0, 1.0)

    print(f"🔮 Prediction: min={prob.min():.6f}, max={prob.max():.6f}, "
          f"mean={prob.mean():.6f}, shape={prob.shape}, Tseq={Tseq}")

    return prob, bounds


def _prob_to_png(prob: np.ndarray, threshold: float = None) -> bytes:
    p = np.clip(prob, 0.0, 1.0)

    if threshold is not None:
        # Hard mask: anything below threshold becomes 0 (transparent-ish in your frontend)
        p = np.where(p >= float(threshold), p, 0.0)

    img = (p * 255.0).astype(np.uint8)

    # Upscale to soften grid artifacts; keep factor moderate to avoid memory spikes
    try:
        from PIL import Image
        im = Image.fromarray(img, mode="L")
        scale = max(1, int(PRED_UPSCALE))
        im = im.resize((img.shape[1] * scale, img.shape[0] * scale), resample=Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raster_to_png_bytes(img)


def _prob_to_geojson(prob: np.ndarray, bounds: Tuple[float, float, float, float], threshold: float) -> dict:
    mask = (prob >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return {"type": "FeatureCollection", "features": []}

    w, s, e, n = bounds
    H, W = mask.shape

    dx = (e - w) / float(W)
    dy = (n - s) / float(H)
    transform = rasterio.transform.from_origin(w, n, dx, dy)

    feats = []
    for geom, val in rio_shapes(mask, mask=mask.astype(bool), transform=transform):
        if val != 1:
            continue
        feats.append({"type": "Feature", "properties": {"threshold": threshold}, "geometry": geom})

    return {"type": "FeatureCollection", "features": feats}


def _crop_prob_around_point(
    prob: np.ndarray,
    bounds: Tuple[float, float, float, float],
    lat: float,
    lon: float,
    crop_frac: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Crop the prob raster to a smaller box centered on (lat, lon).
    crop_frac is the fraction of tile width/height to keep (0..1].
    """
    if crop_frac is None or crop_frac >= 1.0 or crop_frac <= 0:
        return prob, bounds

    H, W = prob.shape
    try:
        # Use the same projection as the tile grid to locate the pixel precisely.
        tile = lonlat_to_tile(lon, lat)
        minx, miny, maxx, maxy = tile_bounds_albers(tile)
        x_m, y_m = lonlat_to_xy_m(lon, lat)

        # Pixel coords of the point (origin at top-left).
        x = int((x_m - minx) / float(PIX))
        y = int((maxy - y_m) / float(PIX))
    except Exception:
        # Fallback to linear lon/lat interpolation if projection fails.
        w, s, e, n = bounds
        dx = (e - w) / float(W)
        dy = (n - s) / float(H)
        x = int((lon - w) / dx)
        y = int((n - lat) / dy)

    crop_w = max(1, int(W * crop_frac))
    crop_h = max(1, int(H * crop_frac))

    x0 = x - crop_w // 2
    y0 = y - crop_h // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # clamp to bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    cropped = prob[y0:y1, x0:x1]

    # Recompute bounds for the crop in the tile's projection, then convert to lon/lat.
    try:
        minx, miny, maxx, maxy = tile_bounds_albers(lonlat_to_tile(lon, lat))
        w_m = minx + x0 * PIX
        e_m = minx + x1 * PIX
        n_m = maxy - y0 * PIX
        s_m = maxy - y1 * PIX
        w2, s2 = _TO_WGS84.transform(w_m, s_m)
        e2, n2 = _TO_WGS84.transform(e_m, n_m)
        return cropped, (float(w2), float(s2), float(e2), float(n2))
    except Exception:
        return cropped, bounds


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "ok": True,
        "modelPath": MODEL_PATH,
        "modelExists": model_exists,
        "device": str(DEVICE),
        "Cd": MODEL_CD,
        "Cs": MODEL_CS,
        "hidden": MODEL_HIDDEN,
        "lstm_layers": MODEL_LSTM_LAYERS,
        "threshold": MODEL_THRESHOLD,
    }


@app.get("/tile-bounds")
def tile_bounds_endpoint(lat: float = Query(...), lon: float = Query(...)):
    grid = build_grid(lat, lon)
    bounds = _extract_bounds_from_grid(grid, lat=lat, lon=lon)
    return {"bounds": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]} 


@app.get("/predict")
def predict_png(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(1),
    png: bool = Query(True),
    thr: float = Query(None),
    crop_frac: float = Query(1.0),
    ignition: bool = Query(False),
):
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition)
    prob, bounds = _crop_prob_around_point(prob, bounds, lat, lon, crop_frac)
    # Optional threshold override for meta only
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD

    if not png:
        area_fraction = float((prob >= threshold).mean())
        return {
            "bounds": list(map(float, bounds)),
            "area_fraction": area_fraction,
            "threshold": threshold,
            "prob_min": float(prob.min()),
            "prob_max": float(prob.max()),
            "prob_mean": float(prob.mean()),
        }

    png_bytes = _prob_to_png(prob, threshold=threshold)
    headers = {"X-Bounds": ",".join(map(str, bounds))}
    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.get("/predict_raster")
def predict_raster_png(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(1)):
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq)
    png = _prob_to_png(prob)
    headers = {"X-Bounds": ",".join(map(str, bounds))}
    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/predict_raster_json")
def predict_raster_json(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(1),
    thr: float = Query(None),
    crop_frac: float = Query(0.5),
    ignition: bool = Query(False),
):
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition)
    prob, bounds = _crop_prob_around_point(prob, bounds, lat, lon, crop_frac)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    png = _prob_to_png(prob, threshold=threshold)

    return {
        "bounds": list(map(float, bounds)),
        "image_base64": base64.b64encode(png).decode("ascii"),
        "threshold": threshold,
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean()),
        "prob_max": float(prob.max()),
        "area_fraction": float((prob >= threshold).mean()),
    }

@app.get("/predict_geojson")
def predict_geojson(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(1),
    thr: float = Query(None),
    crop_frac: float = Query(0.5),
    ignition: bool = Query(False),
):
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq,  ignition=ignition)
    prob, bounds = _crop_prob_around_point(prob, bounds, lat, lon, crop_frac)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    gj = _prob_to_geojson(prob, bounds, threshold=threshold)
    return JSONResponse(content=gj)

@app.get("/predict_raster_json_raw")
def predict_raster_json_raw(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(1), ignition: bool = Query(True)):
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition)

    # IMPORTANT: no threshold here
    png = _prob_to_png(prob, threshold=None)

    return {
        "bounds": list(map(float, bounds)),
        "image_base64": base64.b64encode(png).decode("ascii"),
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean()),
        "prob_max": float(prob.max()),
    }
