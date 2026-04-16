import os
import io
import re
import math
import base64
import datetime as dt
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
from .dynamic_builder import build_dynamic_for_tile, fetch_weather_grids
from .static_builder import load_static_for_tile, CHANNEL_ORDER

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
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.01"))

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


def _parse_date_param(date_str: Optional[str]) -> Optional[dt.datetime]:
    if not date_str:
        return None
    try:
        d = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d
    except Exception:
        # Try simple date format YYYY-MM-DD, default to noon UTC
        try:
            d = dt.datetime.strptime(date_str, "%Y-%m-%d")
            return d.replace(hour=12, tzinfo=dt.timezone.utc)
        except Exception:
            return None


def _resolve_prediction_time(ref_time: Optional[dt.datetime]) -> dt.datetime:
    if ref_time is not None:
        return ref_time
    return dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)


def _load_static_tensor(tile: Any) -> np.ndarray:
    stat_dict = load_static_for_tile(tile)
    try:
        return np.stack([np.asarray(stat_dict[k], np.float32) for k in CHANNEL_ORDER], axis=0)
    except KeyError as e:
        missing = str(e)
        raise RuntimeError(f"Static channel missing: {missing}; have keys={list(stat_dict.keys())}")


def _postprocess_probability(prob: np.ndarray) -> np.ndarray:
    if PRED_SMOOTH_SIGMA > 0:
        try:
            prob = gaussian_filter(prob, sigma=PRED_SMOOTH_SIGMA)
        except Exception:
            pass
    return np.clip(prob, 0.0, 1.0)


def _resize_prob_to_shape(prob: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = map(int, shape)
    if prob.shape == (target_h, target_w):
        return np.asarray(prob, dtype=np.float32)

    tensor = torch.from_numpy(np.asarray(prob, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy().astype(np.float32)


def _rectangle_coordinates_from_bounds(bounds: Tuple[float, float, float, float]) -> list:
    w, s, e, n = bounds
    return [
        [float(w), float(n)],
        [float(e), float(n)],
        [float(e), float(s)],
        [float(w), float(s)],
    ]


def _bbox_from_coordinates(coordinates: list) -> Tuple[float, float, float, float]:
    lons = [float(coord[0]) for coord in coordinates]
    lats = [float(coord[1]) for coord in coordinates]
    return (min(lons), min(lats), max(lons), max(lats))


def _window_coordinates(
    shape: Tuple[int, int],
    lat: float,
    lon: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fallback_bounds: Tuple[float, float, float, float],
) -> Tuple[list, Tuple[float, float, float, float]]:
    H, W = shape
    try:
        minx, miny, maxx, maxy = tile_bounds_albers(lonlat_to_tile(lon, lat))
        # Use PIX directly — same grid constant the model was trained with
        west_m = minx + x0 * PIX
        east_m = minx + x1 * PIX
        north_m = maxy - y0 * PIX
        south_m = maxy - y1 * PIX
        corners_m = [
            (west_m, north_m),
            (east_m, north_m),
            (east_m, south_m),
            (west_m, south_m),
        ]
        coordinates = []
        for x_m, y_m in corners_m:
            lon2, lat2 = _TO_WGS84.transform(x_m, y_m)
            if not (math.isfinite(lon2) and math.isfinite(lat2)):
                raise ValueError("non-finite transformed crop corner")
            coordinates.append([float(lon2), float(lat2)])
        return coordinates, _bbox_from_coordinates(coordinates)
    except Exception:
        rectangle = _rectangle_coordinates_from_bounds(fallback_bounds)
        return rectangle, fallback_bounds


def _predict_probability_from_inputs(dyn: np.ndarray, stat: np.ndarray, *, log_label: str = "Prediction") -> np.ndarray:
    model = _load_model_once()

    if dyn.shape[1] != MODEL_CD:
        raise RuntimeError(f"Dynamic channels mismatch: got {dyn.shape[1]} expected {MODEL_CD}")
    if stat.shape[0] != MODEL_CS:
        raise RuntimeError(f"Static channels mismatch: got {stat.shape[0]} expected {MODEL_CS}")

    if model is None:
        H, W = dyn.shape[-2], dyn.shape[-1]
        prob = np.zeros((H, W), dtype=np.float32)
    else:
        x_dyn = torch.from_numpy(dyn).unsqueeze(0).to(DEVICE)    # (1, T, Cd, H, W)
        x_stat = torch.from_numpy(stat).unsqueeze(0).to(DEVICE)  # (1, Cs, H, W)
        with torch.no_grad():
            logits = model(x_dyn, x_stat)  # (1,1,H,W)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    prob = _postprocess_probability(prob)
    print(
        f"🔮 {log_label}: min={prob.min():.6f}, max={prob.max():.6f}, "
        f"mean={prob.mean():.6f}, shape={prob.shape}, Tseq={dyn.shape[0]}"
    )
    return prob


def _prepare_prediction_inputs(
    lat: float,
    lon: float,
    Tseq: int,
    *,
    ignition: bool = False,
    ref_time: Optional[dt.datetime] = None,
    hours_step: int = 24,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], dt.datetime]:
    tile, _, bounds = build_grid(lat, lon)
    dyn = build_dynamic_for_tile(
        lat,
        lon,
        T_seq=Tseq,
        hours_step=hours_step,
        ignition=ignition,
        ref_time=ref_time,
    )
    stat = _load_static_tensor(tile)
    return dyn, stat, bounds, _resolve_prediction_time(ref_time)


def _predict_probability(
    lat: float,
    lon: float,
    Tseq: int,
    ignition: bool = False,
    ref_time: Optional[dt.datetime] = None,
    *,
    hours_step: int = 24,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Returns:
      prob (H,W) in [0,1]
      bounds (W,S,E,N) in lon/lat
    """
    dyn, stat, bounds, _ = _prepare_prediction_inputs(
        lat,
        lon,
        Tseq,
        ignition=ignition,
        ref_time=ref_time,
        hours_step=hours_step,
    )
    return _predict_probability_from_inputs(dyn, stat), bounds


def _prob_to_grayscale(prob: np.ndarray, threshold: float = None) -> np.ndarray:
    p = np.clip(prob, 0.0, 1.0)

    if threshold is not None:
        p = np.where(p >= float(threshold), p, 0.0)

    pmin = float(p[p > 0].min()) if (p > 0).any() else 0.0
    pmax = float(p.max())
    if pmax > pmin:
        normalized = np.where(p > 0, (p - pmin) / (pmax - pmin), 0.0)
        return (normalized * 255.0).astype(np.uint8)
    return (p * 255.0).astype(np.uint8)


def _encode_png_from_channels(gray: np.ndarray, alpha: Optional[np.ndarray] = None) -> bytes:
    try:
        from PIL import Image

        scale = max(1, int(PRED_UPSCALE))
        if alpha is None:
            im = Image.fromarray(gray, mode="L")
        else:
            rgba = np.stack([gray, gray, gray, alpha], axis=-1)
            im = Image.fromarray(rgba, mode="RGBA")
        im = im.resize((gray.shape[1] * scale, gray.shape[0] * scale), resample=Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raster_to_png_bytes(gray)


def _prob_to_png(prob: np.ndarray, threshold: float = None) -> bytes:
    return _encode_png_from_channels(_prob_to_grayscale(prob, threshold=threshold))


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


def _point_pixel_for_raster(
    shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    lat: float,
    lon: float,
) -> Tuple[int, int]:
    H, W = shape
    try:
        tile = lonlat_to_tile(lon, lat)
        minx, miny, maxx, maxy = tile_bounds_albers(tile)
        x_m, y_m = lonlat_to_xy_m(lon, lat)
        # Use the grid constant PIX directly — avoids coupling to image dimensions
        x = int(round((x_m - minx) / PIX))
        y = int(round((maxy - y_m) / PIX))
    except Exception:
        w, s, e, n = bounds
        dx = (e - w) / float(W)
        dy = (n - s) / float(H)
        x = int(round((lon - w) / dx))
        y = int(round((n - lat) / dy))
    return max(0, min(W - 1, x)), max(0, min(H - 1, y))


def _build_crop_window(
    shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    lat: float,
    lon: float,
    crop_frac: float,
) -> Dict[str, Any]:
    """
    Build a reusable crop window centered on the clicked point.
    Returns crop indices, cropped bounds, and the fire point in crop-local pixels.
    """
    H, W = shape
    x, y = _point_pixel_for_raster(shape, bounds, lat, lon)

    if crop_frac is None or crop_frac >= 1.0 or crop_frac <= 0:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
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

    cropped_coordinates, cropped_bounds = _window_coordinates(
        shape,
        lat,
        lon,
        x0,
        y0,
        x1,
        y1,
        bounds,
    )

    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "bounds": cropped_bounds,
        "coordinates": cropped_coordinates,
        "anchor_px": (max(0, min(x1 - x0 - 1, x - x0)), max(0, min(y1 - y0 - 1, y - y0))),
    }


def _apply_crop_window(prob: np.ndarray, crop_window: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[float, float, float, float], Tuple[int, int]]:
    x0 = crop_window["x0"]
    y0 = crop_window["y0"]
    x1 = crop_window["x1"]
    y1 = crop_window["y1"]
    return prob[y0:y1, x0:x1], crop_window["bounds"], crop_window["anchor_px"]


def _radial_alpha_mask(shape: Tuple[int, int], anchor_px: Tuple[int, int]) -> np.ndarray:
    H, W = shape
    cx, cy = anchor_px
    yy, xx = np.indices((H, W), dtype=np.float32)
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    corner_dists = [
        math.hypot(float(cx), float(cy)),
        math.hypot(float(W - 1 - cx), float(cy)),
        math.hypot(float(cx), float(H - 1 - cy)),
        math.hypot(float(W - 1 - cx), float(H - 1 - cy)),
    ]
    max_dist = max(max(corner_dists), 1.0)
    norm = np.clip(dist / max_dist, 0.0, 1.0)
    fade = np.clip(1.0 - np.power(norm, 1.7), 0.0, 1.0)
    return fade.astype(np.float32)


def _prob_to_display_png(prob: np.ndarray, threshold: float, anchor_px: Tuple[int, int]) -> bytes:
    gray = _prob_to_grayscale(prob, threshold=threshold)
    active = (gray > 0).astype(np.float32)
    fade = _radial_alpha_mask(gray.shape, anchor_px)
    alpha = np.round(active * fade * 255.0).astype(np.uint8)
    return _encode_png_from_channels(gray, alpha=alpha)


def _step_label(lead_hours: int) -> str:
    if lead_hours % 24 == 0:
        days = lead_hours // 24
        return f"{days} day" if days == 1 else f"{days} days"
    return f"{lead_hours} hours"


def _rollout_multistep_predictions(
    lat: float,
    lon: float,
    *,
    Tseq: int,
    steps: int,
    step_hours: int,
    crop_frac: float,
    ignition: bool,
    ref_time: Optional[dt.datetime],
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], list]:
    dyn, stat, bounds, base_time = _prepare_prediction_inputs(
        lat,
        lon,
        Tseq,
        ignition=ignition,
        ref_time=ref_time,
        hours_step=step_hours,
    )
    normalized_steps = max(1, int(steps))
    crop_window = None
    rollout = []
    current_dyn = dyn.copy()

    for index in range(normalized_steps):
        lead_hours = (index + 1) * int(step_hours)
        prob = _predict_probability_from_inputs(current_dyn, stat, log_label=f"Forecast step {index + 1}")
        if crop_window is None:
            crop_window = _build_crop_window(prob.shape, bounds, lat, lon, crop_frac)
        rollout.append({
            "index": index,
            "lead_hours": lead_hours,
            "label": _step_label(lead_hours),
            "prob": prob,
        })

        if index == normalized_steps - 1:
            continue

        next_time = base_time + dt.timedelta(hours=lead_hours)
        wx = fetch_weather_grids(lat, lon, ref_time=next_time)
        next_fire = _resize_prob_to_shape(prob, current_dyn.shape[-2:])
        next_slice = np.stack([
            np.clip(next_fire, 0.0, 1.0),
            wx["u"], wx["v"], wx["gust"],
            wx["tempC"], wx["q"], wx["precip"],
        ], axis=0).astype(np.float32)
        if current_dyn.shape[0] == 1:
            current_dyn = next_slice[None, ...]
        else:
            current_dyn = np.concatenate([current_dyn[1:], next_slice[None, ...]], axis=0)

    if crop_window is None:
        crop_window = _build_crop_window(current_dyn.shape[-2:], bounds, lat, lon, crop_frac)

    return bounds, crop_window, rollout


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
    date: str = Query(None),
):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition, ref_time=ref_time)
    crop_window = _build_crop_window(prob.shape, bounds, lat, lon, crop_frac)
    prob, bounds, _ = _apply_crop_window(prob, crop_window)
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
def predict_raster_png(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(1), date: str = Query(None)):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ref_time=ref_time)
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
    date: str = Query(None),
):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition, ref_time=ref_time)
    crop_window = _build_crop_window(prob.shape, bounds, lat, lon, crop_frac)
    prob, bounds, anchor_px = _apply_crop_window(prob, crop_window)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    png = _prob_to_display_png(prob, threshold=threshold, anchor_px=anchor_px)

    return {
        "bounds": list(map(float, bounds)),
        "coordinates": crop_window["coordinates"],
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
    date: str = Query(None),
):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition, ref_time=ref_time)
    crop_window = _build_crop_window(prob.shape, bounds, lat, lon, crop_frac)
    prob, bounds, _ = _apply_crop_window(prob, crop_window)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    gj = _prob_to_geojson(prob, bounds, threshold=threshold)
    return JSONResponse(content=gj)

@app.get("/predict_raster_json_raw")
def predict_raster_json_raw(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(1), ignition: bool = Query(True), date: str = Query(None)):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ignition=ignition, ref_time=ref_time)

    # IMPORTANT: no threshold here
    png = _prob_to_png(prob, threshold=None)

    return {
        "bounds": list(map(float, bounds)),
        "image_base64": base64.b64encode(png).decode("ascii"),
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean()),
        "prob_max": float(prob.max()),
    }


@app.get("/predict_multistep")
def predict_multistep(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(1),
    steps: int = Query(6),
    step_hours: int = Query(6),
    thr: float = Query(None),
    crop_frac: float = Query(0.5),
    ignition: bool = Query(False),
    date: str = Query(None),
):
    ref_time = _parse_date_param(date)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    bounds, crop_window, rollout = _rollout_multistep_predictions(
        lat,
        lon,
        Tseq=Tseq,
        steps=steps,
        step_hours=step_hours,
        crop_frac=crop_frac,
        ignition=ignition,
        ref_time=ref_time,
    )

    payload_steps = []
    cropped_bounds = crop_window["bounds"]
    for step in rollout:
        cropped, _, anchor_px = _apply_crop_window(step["prob"], crop_window)
        png = _prob_to_display_png(cropped, threshold=threshold, anchor_px=anchor_px)
        payload_steps.append({
            "index": step["index"],
            "lead_hours": step["lead_hours"],
            "label": step["label"],
            "image_base64": base64.b64encode(png).decode("ascii"),
            "prob_min": float(cropped.min()),
            "prob_mean": float(cropped.mean()),
            "prob_max": float(cropped.max()),
            "area_fraction": float((cropped >= threshold).mean()),
        })

    return {
        "bounds": list(map(float, cropped_bounds)),
        "coordinates": crop_window["coordinates"],
        "threshold": threshold,
        "step_hours": int(step_hours),
        "steps": payload_steps,
    }
