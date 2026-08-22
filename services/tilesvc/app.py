import os
import io
import re
import json
import math
import base64
import datetime as dt
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from fastapi import FastAPI, Query, Request
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
from .dynamic_builder import DEFAULT_DYNAMIC_ORDER, build_dynamic_for_tile, fetch_weather_grids, weather_quality_status
from .static_builder import CHANNEL_ORDER
from .cache_health import firms_snapshot_status, noaa_cycle_status
from .calibration import calibrate_probability, calibration_status
from .ml_runtime import file_sha256, runtime_imports, source_version_info
from .prediction_contract import next_fire_from_delta, risk_class_summary
from .static_catalog import InputUnavailable, load_static_tensor_for_model
from .display_quality import display_mask_from_static, is_static_placeholder_or_missing

import rasterio
from rasterio.features import shapes as rio_shapes

import torch

ConvLSTMUNet, RUNTIME_ARCH_VERSION, MODEL_MODULE, append_derived_features, DERIVED_FEATURE_NAMES, FEATURE_MODULE = runtime_imports()


app = FastAPI(title="Ignis Tilesvc", version="1.0")


_METRIC_COUNTERS: Dict[str, float] = {}
_METRIC_SUMS: Dict[str, float] = {}
_METRIC_COUNTS: Dict[str, float] = {}


def _metric_inc(name: str, value: float = 1.0) -> None:
    _METRIC_COUNTERS[name] = _METRIC_COUNTERS.get(name, 0.0) + float(value)


def _metric_observe(name: str, value: float) -> None:
    _METRIC_SUMS[name] = _METRIC_SUMS.get(name, 0.0) + float(value)
    _METRIC_COUNTS[name] = _METRIC_COUNTS.get(name, 0.0) + 1.0


def _metrics_text() -> str:
    lines: List[str] = []
    for name in sorted(_METRIC_COUNTERS):
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {_METRIC_COUNTERS[name]:.6f}")
    for name in sorted(_METRIC_SUMS):
        lines.append(f"# TYPE {name} summary")
        lines.append(f"{name}_sum {_METRIC_SUMS[name]:.6f}")
        lines.append(f"{name}_count {_METRIC_COUNTS.get(name, 0.0):.0f}")
    return "\n".join(lines) + "\n"


def _noaa_cache_health_dir() -> Optional[str]:
    cache_dir = os.getenv("NOAA_GRID_CACHE_DIR")
    if cache_dir:
        return cache_dir
    template = os.getenv("NOAA_GRID_CACHE_TEMPLATE")
    if template:
        return str(Path(template).parent)
    return None


@app.middleware("http")
async def prediction_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if request.url.path.startswith("/predict") or request.url.path.startswith("/input_audit"):
            _metric_observe("ignis_prediction_request_latency_ms", elapsed_ms)


@app.exception_handler(InputUnavailable)
async def input_unavailable_handler(_request: Request, exc: InputUnavailable):
    _metric_inc("ignis_unavailable_predictions_total")
    if exc.reason.startswith("static_"):
        _metric_inc("ignis_missing_static_total")
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": "input_unavailable",
            "reason": exc.reason,
            "detail": str(exc),
            "quality": {
                "status": "unavailable",
                "degraded": False,
                "reasons": [exc.reason],
            },
            "details": exc.details,
        },
    )

_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

def _get_torch_device() -> torch.device:
    d = os.getenv("TORCH_DEVICE", "cpu").lower().strip()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_torch_device()
DEFAULT_MODEL_PATH = "/app/models/convlstm_unet_v3_delta_Cd13_Cs15_H64_T6_nautilus.pt"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.01"))

# ---------------------------------------------------------------------------
# Autoregressive (AR) feedback configuration.
#
# `MODEL_THRESHOLD` is the *classification* threshold the trainer chose to
# maximize F1 — it answers "is this cell labelled as burned?". For an
# autoregressive multistep rollout we additionally need to decide what cells
# count as "fire" when feeding the next step's input. Reusing the
# classification threshold (typically 0.5–0.95) silently kills the rollout:
# if the per-step probability rarely exceeds 0.5+, no new cells are fed back
# and the fire footprint freezes.
#
# We expose two knobs:
#
#   MODEL_AR_FEEDBACK_MODE   "soft"      (default) propagate the soft
#                                        probability mass into next-step
#                                        fire_t. Best when the model is
#                                        well-calibrated.
#                            "threshold" hard threshold; cells above
#                                        MODEL_AR_FEEDBACK_THRESHOLD become 1
#                                        in the next-step fire_t.
#
#   MODEL_AR_FEEDBACK_THRESHOLD  Float; only used when mode == "threshold".
#                                Defaults to 0.10 — much lower than the
#                                classification threshold by design.
#
# Don't change MODEL_THRESHOLD to fix rollout behavior — that breaks the
# user-visible "decision threshold" semantics. Tune these instead.
# ---------------------------------------------------------------------------
MODEL_AR_FEEDBACK_MODE = os.getenv("MODEL_AR_FEEDBACK_MODE", "soft").strip().lower()
if MODEL_AR_FEEDBACK_MODE not in {"soft", "threshold"}:
    print(
        f"[tilesvc] MODEL_AR_FEEDBACK_MODE={MODEL_AR_FEEDBACK_MODE!r} not recognized; "
        "falling back to 'soft'."
    )
    MODEL_AR_FEEDBACK_MODE = "soft"
MODEL_AR_FEEDBACK_THRESHOLD = float(os.getenv("MODEL_AR_FEEDBACK_THRESHOLD", "0.10"))

PREDICTIONS_ENABLED = os.getenv("PREDICTIONS_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
REQUIRED_ARCH_VERSION = os.getenv("REQUIRED_ARCH_VERSION", "v3")
REQUIRED_TARGET_MODE = os.getenv("REQUIRED_TARGET_MODE", "delta")

MODEL_CD = int(os.getenv("MODEL_CD", "7"))
MODEL_CS = int(os.getenv("MODEL_CS", "15"))
MODEL_HIDDEN = int(os.getenv("MODEL_HIDDEN", "128"))
MODEL_LSTM_LAYERS = int(os.getenv("MODEL_LSTM_LAYERS", "2"))
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH") or str(Path(__file__).with_name("model_config.default.json"))
PRED_SMOOTH_SIGMA = float(os.getenv("PRED_SMOOTH_SIGMA", "1.5"))
PRED_UPSCALE = int(os.getenv("PRED_UPSCALE", "1"))
PRED_DISPLAY_FLOOR = float(os.getenv("PRED_DISPLAY_FLOOR", "0.02"))
DISPLAY_MASK_WATER_THRESHOLD = float(os.getenv("DISPLAY_MASK_WATER_THRESHOLD", "0.5"))
DISPLAY_MASK_IMPERVIOUS_THRESHOLD = float(os.getenv("DISPLAY_MASK_IMPERVIOUS_THRESHOLD", "0.8"))

_model = None
_model_meta = None
_model_sha256 = None


def _csv_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    return parsed or list(default)


MODEL_TSEQ = int(os.getenv("MODEL_TSEQ", "1"))
MODEL_STEP_HOURS = int(os.getenv("MODEL_STEP_HOURS", "24"))
MODEL_DYNAMIC_ORDER = _csv_env("MODEL_DYNAMIC_ORDER", DEFAULT_DYNAMIC_ORDER)
MODEL_STATIC_ORDER = _csv_env("MODEL_STATIC_ORDER", CHANNEL_ORDER)
MODEL_TARGET_MODE = os.getenv("MODEL_TARGET_MODE", "mask")
MODEL_DERIVED_FEATURES_ENABLE = os.getenv("MODEL_DERIVED_FEATURES_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
MODEL_DERIVED_FEATURES_INCLUDE = None
MODEL_DAYS_SINCE_FIRE_CAP = None
if os.getenv("MODEL_DERIVED_FEATURES_INCLUDE"):
    MODEL_DERIVED_FEATURES_INCLUDE = [
        part.strip()
        for part in os.getenv("MODEL_DERIVED_FEATURES_INCLUDE", "").split(",")
        if part.strip()
    ]
if os.getenv("MODEL_DAYS_SINCE_FIRE_CAP"):
    MODEL_DAYS_SINCE_FIRE_CAP = int(os.getenv("MODEL_DAYS_SINCE_FIRE_CAP"))


def _try_parse_from_filename(path: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    fname = os.path.basename(path)
    m = re.search(r"_Cd(\d+)_Cs(\d+)_H(\d+)_T(\d+)_", fname)
    if not m:
        return None, None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


_filename_cd, _filename_cs, _filename_h, _filename_t = _try_parse_from_filename(MODEL_PATH)
if _filename_cd and os.getenv("MODEL_CD") is None:
    MODEL_CD = _filename_cd
if _filename_cs and os.getenv("MODEL_CS") is None:
    MODEL_CS = _filename_cs
if _filename_h and os.getenv("MODEL_HIDDEN") is None:
    MODEL_HIDDEN = _filename_h
if _filename_t and os.getenv("MODEL_TSEQ") is None:
    MODEL_TSEQ = _filename_t


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


def _load_model_config_file() -> Dict[str, Any]:
    if not MODEL_CONFIG_PATH:
        return {}
    try:
        with open(MODEL_CONFIG_PATH, "r") as f:
            if MODEL_CONFIG_PATH.endswith((".yml", ".yaml")):
                # Avoid adding a YAML dependency to the service; JSON is the supported config format.
                raise RuntimeError("MODEL_CONFIG_PATH must be JSON unless PyYAML is added")
            return json.load(f)
    except Exception as exc:
        print(f"⚠️  Failed to load MODEL_CONFIG_PATH={MODEL_CONFIG_PATH}: {exc}")
        return {}


def _list_from_meta(meta: Dict[str, Any], *keys: str) -> Optional[List[str]]:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str):
            parsed = [part.strip() for part in value.split(",") if part.strip()]
            if parsed:
                return parsed
        if isinstance(value, (list, tuple)) and value:
            return [str(part) for part in value]
    return None


def _apply_model_metadata(meta: Dict[str, Any], source: str) -> None:
    global MODEL_CD, MODEL_CS, MODEL_HIDDEN, MODEL_LSTM_LAYERS, MODEL_THRESHOLD
    global MODEL_TSEQ, MODEL_STEP_HOURS, MODEL_DYNAMIC_ORDER, MODEL_STATIC_ORDER
    global MODEL_TARGET_MODE, MODEL_DERIVED_FEATURES_ENABLE, MODEL_DERIVED_FEATURES_INCLUDE
    global MODEL_DAYS_SINCE_FIRE_CAP

    if not isinstance(meta, dict) or not meta:
        return

    source_str = str(source or "")
    checkpoint_wins = source_str == MODEL_PATH or source_str.endswith((".pt", ".pth"))

    if meta.get("cd") is not None and (checkpoint_wins or os.getenv("MODEL_CD") is None):
        MODEL_CD = int(meta["cd"])
    if meta.get("cs") is not None and (checkpoint_wins or os.getenv("MODEL_CS") is None):
        MODEL_CS = int(meta["cs"])
    if meta.get("hidden") is not None and (checkpoint_wins or os.getenv("MODEL_HIDDEN") is None):
        MODEL_HIDDEN = int(meta["hidden"])
    if meta.get("lstm_layers") is not None and (checkpoint_wins or os.getenv("MODEL_LSTM_LAYERS") is None):
        MODEL_LSTM_LAYERS = int(meta["lstm_layers"])
    if meta.get("best_threshold") is not None and (checkpoint_wins or os.getenv("MODEL_THRESHOLD") is None):
        MODEL_THRESHOLD = float(meta["best_threshold"])
    if meta.get("threshold") is not None and (checkpoint_wins or os.getenv("MODEL_THRESHOLD") is None):
        MODEL_THRESHOLD = float(meta["threshold"])
    seq_len = meta.get("seq_len", meta.get("Tseq", meta.get("tseq", meta.get("T"))))
    if seq_len is not None and (checkpoint_wins or os.getenv("MODEL_TSEQ") is None):
        MODEL_TSEQ = int(seq_len)
    step_hours = meta.get("step_hours", meta.get("hours_step"))
    if step_hours is not None and (checkpoint_wins or os.getenv("MODEL_STEP_HOURS") is None):
        MODEL_STEP_HOURS = int(step_hours)

    dyn_order = _list_from_meta(meta, "dyn_order", "dynamic_order", "dynamic_channels")
    if dyn_order and (checkpoint_wins or os.getenv("MODEL_DYNAMIC_ORDER") is None):
        MODEL_DYNAMIC_ORDER = dyn_order

    stat_order = _list_from_meta(meta, "stat_order", "static_order", "static_channels")
    if stat_order and (checkpoint_wins or os.getenv("MODEL_STATIC_ORDER") is None):
        MODEL_STATIC_ORDER = stat_order

    target_mode = meta.get("target_mode")
    if target_mode is not None and (checkpoint_wins or os.getenv("MODEL_TARGET_MODE") is None):
        MODEL_TARGET_MODE = str(target_mode)

    if meta.get("derived_features_enable") is not None and (
        checkpoint_wins or os.getenv("MODEL_DERIVED_FEATURES_ENABLE") is None
    ):
        MODEL_DERIVED_FEATURES_ENABLE = bool(meta.get("derived_features_enable"))

    if "derived_features_include" in meta and (
        checkpoint_wins or os.getenv("MODEL_DERIVED_FEATURES_INCLUDE") is None
    ):
        value = meta.get("derived_features_include")
        MODEL_DERIVED_FEATURES_INCLUDE = list(value) if value is not None else None

    if meta.get("days_since_fire_cap") is not None and (
        checkpoint_wins or os.getenv("MODEL_DAYS_SINCE_FIRE_CAP") is None
    ):
        MODEL_DAYS_SINCE_FIRE_CAP = int(meta["days_since_fire_cap"])

    print(f"[tilesvc] Applied model metadata from {source}")


def _checkpoint_metadata_from_obj(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        return {
            k: v
            for k, v in ckpt.items()
            if k not in {"state_dict", "model_state_dict", "optimizer", "scheduler"}
            and not hasattr(v, "shape")
        }
    return {}


def _model_metadata(config_status: str = None) -> Dict[str, Any]:
    extra = _model_meta or {}
    meta = {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_sha256": _model_sha256 or file_sha256(MODEL_PATH),
        "device": str(DEVICE),
        "Cd": MODEL_CD,
        "Cs": MODEL_CS,
        "hidden": MODEL_HIDDEN,
        "lstm_layers": MODEL_LSTM_LAYERS,
        "threshold": MODEL_THRESHOLD,
        "display_floor": PRED_DISPLAY_FLOOR,
        "Tseq": MODEL_TSEQ,
        "step_hours": MODEL_STEP_HOURS,
        # Surfaced in /healthz, /predict_multistep, etc. so a deploy can be
        # verified end-to-end: if these keys are missing or stale, the new
        # AR-feedback code didn't actually ship.
        "ar_feedback_mode": MODEL_AR_FEEDBACK_MODE,
        "ar_feedback_threshold": MODEL_AR_FEEDBACK_THRESHOLD,
        "arch_version": extra.get("arch_version"),
        "runtime_arch_version": RUNTIME_ARCH_VERSION,
        "target_mode": MODEL_TARGET_MODE,
        "derived_features_enable": MODEL_DERIVED_FEATURES_ENABLE,
        "derived_features_include": MODEL_DERIVED_FEATURES_INCLUDE,
        "days_since_fire_cap": MODEL_DAYS_SINCE_FIRE_CAP,
        "dynamic_order": list(MODEL_DYNAMIC_ORDER),
        "static_order": list(MODEL_STATIC_ORDER),
        "derived_feature_names": list(DERIVED_FEATURE_NAMES),
        "model_module": MODEL_MODULE,
        "feature_module": FEATURE_MODULE,
        "ml_source": source_version_info(),
        "metadata_source": extra.get("config_source"),
        "config_status": config_status or (
            "explicit_or_checkpoint" if (_model_meta or MODEL_CONFIG_PATH) else "environment_or_filename_defaults"
        ),
        "predictions_enabled": PREDICTIONS_ENABLED,
    }
    for key in ("model_name", "production_valid", "static_placeholders", "normalization", "metadata_notes"):
        if key in extra and extra[key] is not None:
            meta[key] = extra[key]
    return meta


def _ensure_model_metadata_loaded(load_checkpoint: bool = True) -> None:
    global _model_meta
    if _model_meta is not None:
        return

    config_meta = _load_model_config_file()
    if config_meta:
        _apply_model_metadata(config_meta, MODEL_CONFIG_PATH or "MODEL_CONFIG_PATH")

    checkpoint_meta = {}
    if load_checkpoint and os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            checkpoint_meta = _checkpoint_metadata_from_obj(ckpt)
            if checkpoint_meta:
                _apply_model_metadata(checkpoint_meta, MODEL_PATH)
        except Exception as exc:
            print(f"⚠️  Failed to read checkpoint metadata from {MODEL_PATH}: {exc}")

    status = "checkpoint" if checkpoint_meta else ("config_file" if config_meta else "environment_or_filename_defaults")
    merged_meta = {**config_meta, **checkpoint_meta}
    _model_meta = {
        "config_source": status,
        "checkpoint_meta_keys": sorted(checkpoint_meta.keys()),
        "config_meta_keys": sorted(config_meta.keys()),
        "arch_version": merged_meta.get("arch_version"),
        "model_name": merged_meta.get("model_name") or merged_meta.get("name"),
        "production_valid": merged_meta.get("production_valid"),
        "static_placeholders": _list_from_meta(
            merged_meta,
            "static_placeholders",
            "placeholder_static_channels",
            "unavailable_static_channels",
        ) or [],
        "normalization": merged_meta.get("normalization"),
        "metadata_notes": merged_meta.get("notes"),
    }


def _expected_dynamic_cd() -> int:
    if not MODEL_DERIVED_FEATURES_ENABLE:
        return len(MODEL_DYNAMIC_ORDER)
    include = MODEL_DERIVED_FEATURES_INCLUDE
    return len(MODEL_DYNAMIC_ORDER) + (len(DERIVED_FEATURE_NAMES) if include is None else len(include))


def _validate_model_contract(checkpoint_meta: Dict[str, Any]) -> None:
    arch = checkpoint_meta.get("arch_version")
    if REQUIRED_ARCH_VERSION and arch != REQUIRED_ARCH_VERSION:
        raise RuntimeError(f"checkpoint arch_version={arch!r} does not match required {REQUIRED_ARCH_VERSION!r}")
    target_mode = str(checkpoint_meta.get("target_mode", MODEL_TARGET_MODE))
    if REQUIRED_TARGET_MODE and target_mode != REQUIRED_TARGET_MODE:
        raise RuntimeError(f"checkpoint target_mode={target_mode!r} does not match required {REQUIRED_TARGET_MODE!r}")
    expected_cd = _expected_dynamic_cd()
    if int(MODEL_CD) != expected_cd:
        raise RuntimeError(
            f"model dynamic channel count mismatch: checkpoint Cd={MODEL_CD}, "
            f"but base+derived order implies {expected_cd}"
        )
    if int(MODEL_CS) != len(MODEL_STATIC_ORDER):
        raise RuntimeError(
            f"model static channel count mismatch: checkpoint Cs={MODEL_CS}, "
            f"static_order has {len(MODEL_STATIC_ORDER)} channels"
        )
    if RUNTIME_ARCH_VERSION != "unknown" and arch and RUNTIME_ARCH_VERSION != arch:
        raise RuntimeError(
            f"runtime model code arch={RUNTIME_ARCH_VERSION!r} does not match checkpoint arch={arch!r}; "
            "mount/install the matching ignis_ml_nautilus package"
        )


_initial_config_meta = _load_model_config_file()
if _initial_config_meta:
    _apply_model_metadata(_initial_config_meta, MODEL_CONFIG_PATH or "MODEL_CONFIG_PATH")


def _load_model_once():
    global _model, _model_sha256, MODEL_CD, MODEL_CS, MODEL_HIDDEN, MODEL_LSTM_LAYERS

    if _model is not None:
        return _model

    _ensure_model_metadata_loaded(load_checkpoint=True)

    if not os.path.exists(MODEL_PATH):
        raise InputUnavailable(
            f"MODEL_PATH does not exist: {MODEL_PATH}",
            reason="model_missing",
            details={"model_path": MODEL_PATH},
        )
    _model_sha256 = file_sha256(MODEL_PATH)

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
    checkpoint_meta = _checkpoint_metadata_from_obj(ckpt)
    if checkpoint_meta:
        _apply_model_metadata(checkpoint_meta, MODEL_PATH)
        _validate_model_contract(checkpoint_meta)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(DEVICE)
    elif isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            cleaned[nk] = v
        strict_load = os.getenv("ALLOW_PARTIAL_MODEL_LOAD", "0").strip().lower() not in {"1", "true", "yes", "on"}
        incompat = model.load_state_dict(cleaned, strict=strict_load)
        if (getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None)):
            print(
                "⚠️  Model state_dict loaded with mismatches: "
                f"missing={getattr(incompat, 'missing_keys', [])}, "
                f"unexpected={getattr(incompat, 'unexpected_keys', [])}"
            )
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
    stat, _summary = load_static_tensor_for_model(tile, MODEL_STATIC_ORDER)
    return stat


def _load_static_tensor_with_summary(tile: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    return load_static_tensor_for_model(tile, MODEL_STATIC_ORDER)


def _channel_stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return {"min": None, "mean": None, "max": None, "pct_zero": None, "finite": False}
    vals = a[finite]
    return {
        "min": float(vals.min()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "pct_zero": float((vals == 0).mean()),
        "finite": True,
    }


_DYN_RANGES: Dict[str, Tuple[float, float]] = {
    "fire_t": (0.0, 1.0),
    "u": (-15.0, 15.0),
    "v": (-15.0, 15.0),
    "gust": (0.0, 25.0),
    "tempC": (-10.0, 45.0),
    "q": (0.0, 0.02),
    "precip": (0.0, 50.0),
}


def _to01_dynamic(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _prepare_dynamic_for_model(dyn_phys: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    base_order = list(MODEL_DYNAMIC_ORDER)
    if dyn_phys.shape[1] != len(base_order):
        raise RuntimeError(
            f"base dynamic tensor has {dyn_phys.shape[1]} channels but dynamic_order has {len(base_order)}"
        )
    x = np.empty_like(dyn_phys, dtype=np.float32)
    for idx, name in enumerate(base_order):
        lo, hi = _DYN_RANGES.get(name, (0.0, 1.0))
        x[:, idx] = _to01_dynamic(dyn_phys[:, idx], lo, hi)
    if base_order and base_order[0] == "fire_t":
        x[:, 0] = np.clip(x[:, 0] * 5.0, 0.0, 1.0)

    out_order = list(base_order)
    if MODEL_DERIVED_FEATURES_ENABLE:
        x, out_order = append_derived_features(
            x,
            dyn_order=base_order,
            include=MODEL_DERIVED_FEATURES_INCLUDE,
            days_since_fire_cap=MODEL_DAYS_SINCE_FIRE_CAP,
        )
    if x.shape[1] != MODEL_CD:
        raise RuntimeError(f"model dynamic tensor has Cd={x.shape[1]} but checkpoint expects Cd={MODEL_CD}")
    return x.astype(np.float32), {
        "base_dynamic_order": base_order,
        "model_dynamic_order": out_order,
        "base_dyn_shape": list(dyn_phys.shape),
        "model_dyn_shape": list(x.shape),
        "derived_features_enable": MODEL_DERIVED_FEATURES_ENABLE,
        "derived_features_include": MODEL_DERIVED_FEATURES_INCLUDE,
    }


def _tensor_input_summary(
    dyn: np.ndarray,
    stat: np.ndarray,
    *,
    bounds: Tuple[float, float, float, float],
    base_time: dt.datetime,
    dyn_model: Optional[np.ndarray] = None,
    dynamic_preparation: Optional[Dict[str, Any]] = None,
    static_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dynamic_channels = {
        name: _channel_stats(dyn[:, idx, :, :])
        for idx, name in enumerate(MODEL_DYNAMIC_ORDER)
    }
    static_channels = {
        name: {
            **_channel_stats(stat[idx]),
            "placeholder_or_missing": is_static_placeholder_or_missing(name, stat[idx]),
        }
        for idx, name in enumerate(MODEL_STATIC_ORDER)
    }
    return {
        "dyn_shape": list(dyn.shape),
        "model_dyn_shape": list(dyn_model.shape) if dyn_model is not None else None,
        "stat_shape": list(stat.shape),
        "bounds": [float(v) for v in bounds],
        "base_time": base_time.isoformat(),
        "dynamic_order": list(MODEL_DYNAMIC_ORDER),
        "model_dynamic_order": dynamic_preparation.get("model_dynamic_order") if dynamic_preparation else None,
        "static_order": list(MODEL_STATIC_ORDER),
        "dynamic_channels": dynamic_channels,
        "static_channels": static_channels,
        "static_catalog": static_summary.get("catalog") if static_summary else None,
        "missing_or_placeholder_static": [
            name for name, stats in static_channels.items()
            if stats.get("placeholder_or_missing")
        ],
        "dynamic_preparation": dynamic_preparation or {},
    }


def _postprocess_probability(prob: np.ndarray) -> np.ndarray:
    if PRED_SMOOTH_SIGMA > 0:
        try:
            prob = gaussian_filter(prob, sigma=PRED_SMOOTH_SIGMA)
        except Exception:
            pass
    return np.clip(prob, 0.0, 1.0)


def _resolve_display_floor(value: Optional[float]) -> float:
    if value is None:
        return max(0.0, min(1.0, float(PRED_DISPLAY_FLOOR)))
    return max(0.0, min(1.0, float(value)))


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
    if not PREDICTIONS_ENABLED:
        raise InputUnavailable(
            "Predictions are disabled by PREDICTIONS_ENABLED=false",
            reason="predictions_disabled",
        )
    model = _load_model_once()

    dyn_model, _prep = _prepare_dynamic_for_model(dyn)
    if dyn_model.shape[1] != MODEL_CD:
        raise RuntimeError(f"Dynamic channels mismatch: got {dyn_model.shape[1]} expected {MODEL_CD}")
    if stat.shape[0] != MODEL_CS:
        raise RuntimeError(f"Static channels mismatch: got {stat.shape[0]} expected {MODEL_CS}")

    x_dyn = torch.from_numpy(dyn_model).unsqueeze(0).to(DEVICE)    # (1, T, Cd, H, W)
    x_stat = torch.from_numpy(stat).unsqueeze(0).to(DEVICE)  # (1, Cs, H, W)
    infer_start = time.perf_counter()
    with torch.no_grad():
        logits = model(x_dyn, x_stat)  # (1,1,H,W)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    _metric_observe("ignis_model_inference_ms", (time.perf_counter() - infer_start) * 1000.0)

    prob = _postprocess_probability(prob)
    print(
        f"🔮 {log_label}: min={prob.min():.6f}, max={prob.max():.6f}, "
        f"mean={prob.mean():.6f}, shape={prob.shape}, Tseq={dyn.shape[0]}, target_mode={MODEL_TARGET_MODE}"
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
    _ensure_model_metadata_loaded(load_checkpoint=True)
    tile, _, bounds = build_grid(lat, lon)
    dyn = build_dynamic_for_tile(
        lat,
        lon,
        T_seq=Tseq,
        hours_step=hours_step,
        ignition=ignition,
        ref_time=ref_time,
        channel_order=MODEL_DYNAMIC_ORDER,
    )
    stat = _load_static_tensor(tile)
    return dyn, stat, bounds, _resolve_prediction_time(ref_time)


def _prepare_prediction_inputs_with_summary(
    lat: float,
    lon: float,
    Tseq: int,
    *,
    ignition: bool = False,
    ref_time: Optional[dt.datetime] = None,
    hours_step: int = 24,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], dt.datetime, Dict[str, Any]]:
    _ensure_model_metadata_loaded(load_checkpoint=True)
    tile, _, bounds = build_grid(lat, lon)
    dyn = build_dynamic_for_tile(
        lat,
        lon,
        T_seq=Tseq,
        hours_step=hours_step,
        ignition=ignition,
        ref_time=ref_time,
        channel_order=MODEL_DYNAMIC_ORDER,
    )
    stat, static_summary = _load_static_tensor_with_summary(tile)
    return dyn, stat, bounds, _resolve_prediction_time(ref_time), static_summary


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

        # Slide the window inward when it extends past a tile boundary so
        # the crop stays full-size and the fire doesn't end up at the edge.
        if x1 > W:
            x0 -= (x1 - W)
            x1 = W
        if x0 < 0:
            x1 += (-x0)
            x0 = 0
        if y1 > H:
            y0 -= (y1 - H)
            y1 = H
        if y0 < 0:
            y1 += (-y0)
            y0 = 0

        # Final safety clamp (handles tile smaller than crop_w)
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

    # Radial fade from the anchor (fire source) outward
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    corner_dists = [
        math.hypot(float(cx), float(cy)),
        math.hypot(float(W - 1 - cx), float(cy)),
        math.hypot(float(cx), float(H - 1 - cy)),
        math.hypot(float(W - 1 - cx), float(H - 1 - cy)),
    ]
    max_dist = max(max(corner_dists), 1.0)
    norm = np.clip(dist / max_dist, 0.0, 1.0)
    radial_fade = np.clip(1.0 - np.power(norm, 1.7), 0.0, 1.0)

    # Edge vignette: fade toward all 4 borders to remove the hard rectangular
    # frame that becomes visible when the overlay is viewed zoomed out.
    # The fade is suppressed on any edge that the anchor is near, so the fire
    # source itself is never darkened when it sits close to a tile boundary.
    margin = max(3, min(W, H) // 6)

    # Normalised distance from each edge (0 = at edge, 1 = >= margin inside)
    d_left   = np.clip(xx / margin, 0.0, 1.0)
    d_right  = np.clip((W - 1 - xx) / margin, 0.0, 1.0)
    d_top    = np.clip(yy / margin, 0.0, 1.0)
    d_bottom = np.clip((H - 1 - yy) / margin, 0.0, 1.0)

    # How far the anchor itself is from each edge (0 = anchor at edge, 1 = far)
    a_left   = float(np.clip(cx / margin, 0.0, 1.0))
    a_right  = float(np.clip((W - 1 - cx) / margin, 0.0, 1.0))
    a_top    = float(np.clip(cy / margin, 0.0, 1.0))
    a_bottom = float(np.clip((H - 1 - cy) / margin, 0.0, 1.0))

    # edge_weight = d_edge + (1 - a_edge): when anchor is AT the edge (a=0),
    # weight is always 1 (no fade there); when anchor is far (a=1), weight
    # equals d_edge (full fade at that border).
    vignette = (
        np.clip(d_left   + (1.0 - a_left),   0.0, 1.0) *
        np.clip(d_right  + (1.0 - a_right),  0.0, 1.0) *
        np.clip(d_top    + (1.0 - a_top),    0.0, 1.0) *
        np.clip(d_bottom + (1.0 - a_bottom), 0.0, 1.0)
    )

    return (radial_fade * vignette).astype(np.float32)


def _prob_to_display_png(prob: np.ndarray, threshold: float, anchor_px: Tuple[int, int]) -> bytes:
    gray = _prob_to_grayscale(prob, threshold=threshold)
    active = (gray > 0).astype(np.float32)
    fade = _radial_alpha_mask(gray.shape, anchor_px)
    alpha = np.round(active * fade * 255.0).astype(np.uint8)
    return _encode_png_from_channels(gray, alpha=alpha)


def _layer_image_base64(
    layer: np.ndarray,
    *,
    threshold: float,
    anchor_px: Tuple[int, int],
) -> str:
    png = _prob_to_display_png(layer.astype(np.float32), threshold=threshold, anchor_px=anchor_px)
    return base64.b64encode(png).decode("ascii")


def _observed_fire_from_dyn(dyn: np.ndarray) -> np.ndarray:
    try:
        fire_idx = list(MODEL_DYNAMIC_ORDER).index("fire_t")
    except ValueError:
        fire_idx = 0
    return (dyn[-1, fire_idx] >= 0.5).astype(np.float32)


def _next_fire_from_delta(prob_new_burn: np.ndarray, observed_fire: np.ndarray, threshold: float) -> np.ndarray:
    return next_fire_from_delta(
        prob_new_burn,
        observed_fire,
        threshold,
        target_mode=MODEL_TARGET_MODE,
    )


def _risk_class_summary(risk: np.ndarray) -> Dict[str, float]:
    return risk_class_summary(risk)


def _probability_contract(
    *,
    prob_new_burn: np.ndarray,
    observed_fire: np.ndarray,
    threshold: float,
    display_score: np.ndarray,
    risk_class: np.ndarray,
    calibration_meta: Dict[str, Any],
) -> Dict[str, Any]:
    p_next_fire = _next_fire_from_delta(prob_new_burn, observed_fire, threshold)
    weather_quality = weather_quality_status()
    quality_status = weather_quality.get("status", "degraded")
    quality_reasons = [] if quality_status == "ok" else [weather_quality.get("reason") or "open_meteo_weather_fallback"]
    if quality_status == "ok":
        _metric_inc("ignis_noaa_weather_predictions_total")
    else:
        _metric_inc("ignis_open_meteo_fallback_total")
        _metric_inc("ignis_degraded_predictions_total")
    return {
        "p_new_burn": {
            "description": "Raw v3 delta probability for new burn/spread pixels",
            "prob_min": float(prob_new_burn.min()),
            "prob_mean": float(prob_new_burn.mean()),
            "prob_max": float(prob_new_burn.max()),
            "area_fraction": float((prob_new_burn >= threshold).mean()),
        },
        "p_next_fire": {
            "description": "Reconstructed next-fire mask/probability proxy: observed_or_persisted_fire OR thresholded new burn",
            "area_fraction": float(p_next_fire.mean()),
        },
        "observed_fire": {
            "description": "Current observed/persisted FIRMS fire channel used as model input",
            "area_fraction": float((observed_fire >= 0.5).mean()),
        },
        "display_score": {
            "description": "Calibrated display score used for heatmap rendering",
            "min": float(display_score.min()),
            "mean": float(display_score.mean()),
            "max": float(display_score.max()),
            "calibration": calibration_meta,
        },
        "risk_class": {
            "description": "Calibrated advisory risk classes for display",
            "fractions": _risk_class_summary(risk_class),
        },
        "quality": {
            "status": quality_status,
            "degraded": quality_status != "ok",
            "reasons": quality_reasons,
        },
        "data_sources": {
            "fire": "NASA FIRMS live/archive window plus optional click ignition",
            "weather": "NOAA gridded" if quality_status == "ok" else "Open-Meteo fallback",
            "weather_source": weather_quality.get("source"),
            "static": "STATIC_CATALOG_PATH COG/S3 manifest",
        },
    }


def _step_label(lead_hours: int) -> str:
    if lead_hours % 24 == 0:
        days = lead_hours // 24
        return f"{days} day" if days == 1 else f"{days} days"
    return f"{lead_hours} hours"


def _encode_solid_png(shape: Tuple[int, int], value: float = 0.5) -> bytes:
    """
    Encode a solid-fill PNG for the bounds/coordinate plumbing sanity test.

    Every pixel gets the same gray value and full alpha — no thresholding,
    no radial alpha mask, no per-pixel processing. If rendering this in the
    frontend does not produce a cleanly filled translucent rectangle over
    the advertised crop bounds, the bug is in the bounds/coordinates
    pipeline rather than in the probability field itself.
    """
    H, W = int(shape[0]), int(shape[1])
    v = int(round(max(0.0, min(1.0, float(value))) * 255.0))
    gray = np.full((H, W), v, dtype=np.uint8)
    alpha = np.full((H, W), 255, dtype=np.uint8)
    return _encode_png_from_channels(gray, alpha=alpha)


def _debug_dump_dir(tag: str) -> Path:
    base = Path(os.getenv("IGNIS_DEBUG_DIR", "/tmp/ignis_debug"))
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = base / f"multistep-{stamp}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dump_multistep_artifacts(
    out_dir: Path,
    *,
    dyn: np.ndarray,
    stat: np.ndarray,
    rollout: List[Dict[str, Any]],
    crop_window: Dict[str, Any],
    bounds: Tuple[float, float, float, float],
    ref_time: Optional[dt.datetime],
    lat: float,
    lon: float,
    step_hours: int,
    Tseq: int,
    threshold: float,
    ignition: bool,
) -> None:
    """
    Save raw model inputs and per-step probability arrays for offline
    inspection in the notebook (Model_Eval.ipynb). No masking, no cropping,
    no PNG encoding — we want the tensors the model actually saw / produced.
    """
    try:
        np.save(out_dir / "dyn_input.npy", np.asarray(dyn, dtype=np.float32))
        np.save(out_dir / "stat_input.npy", np.asarray(stat, dtype=np.float32))
        for step in rollout:
            idx = int(step["index"])
            lead = int(step["lead_hours"])
            prob = np.asarray(step["prob"], dtype=np.float32)
            np.save(out_dir / f"prob_step_{idx:02d}_lead{lead:03d}h.npy", prob)

        meta: Dict[str, Any] = {
            "lat": float(lat),
            "lon": float(lon),
            "Tseq": int(Tseq),
            "step_hours": int(step_hours),
            "threshold": float(threshold),
            "ignition": bool(ignition),
            "ref_time": ref_time.isoformat() if ref_time else None,
            "bounds": [float(x) for x in bounds],
            "crop_window": {
                "x0": int(crop_window["x0"]),
                "y0": int(crop_window["y0"]),
                "x1": int(crop_window["x1"]),
                "y1": int(crop_window["y1"]),
                "bounds": [float(x) for x in crop_window["bounds"]],
                "coordinates": crop_window.get("coordinates"),
                "anchor_px": [int(v) for v in crop_window["anchor_px"]],
            },
            "dyn_shape": list(dyn.shape),
            "stat_shape": list(stat.shape),
            "steps": [
                {
                    "index": int(s["index"]),
                    "lead_hours": int(s["lead_hours"]),
                    "label": s["label"],
                    "prob_shape": list(np.asarray(s["prob"]).shape),
                    "prob_min": float(np.asarray(s["prob"]).min()),
                    "prob_mean": float(np.asarray(s["prob"]).mean()),
                    "prob_max": float(np.asarray(s["prob"]).max()),
                }
                for s in rollout
            ],
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as ex:
        # Debug sink is best-effort — never let it take down a request.
        try:
            with open(out_dir / "dump_error.txt", "w") as f:
                f.write(repr(ex))
        except Exception:
            pass


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
    threshold: float,
    debug_sink: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any], list]:
    dyn, stat, bounds, base_time, static_summary = _prepare_prediction_inputs_with_summary(
        lat,
        lon,
        Tseq,
        ignition=ignition,
        ref_time=ref_time,
        hours_step=step_hours,
    )
    if debug_sink is not None:
        # Capture the original (pre-rollout) inputs so callers in dump mode can
        # save the exact tensors the model started from. These are references,
        # not copies — callers should treat them as read-only.
        debug_sink["dyn"] = dyn
        debug_sink["stat"] = stat
        debug_sink["base_time"] = base_time
        debug_sink["static_summary"] = static_summary
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
        try:
            fire_idx = list(MODEL_DYNAMIC_ORDER).index("fire_t")
        except ValueError:
            fire_idx = 0

        # Keep prev_fire as the soft probability mass from the prior step
        # rather than collapsing it through a 0.5 cutoff. This preserves the
        # gradient information the model was trained on (`fire_boost` produces
        # values in (0, 1] for sparse FIRMS pixels — clamping at 0.5 throws
        # most of that signal away).
        prev_fire = current_dyn[-1, fire_idx].astype(np.float32)

        if MODEL_TARGET_MODE == "delta":
            # AR feedback. See MODEL_AR_FEEDBACK_MODE comment above for why
            # we don't reuse `threshold` (the classification threshold) here.
            if MODEL_AR_FEEDBACK_MODE == "soft":
                # Carry probability mass forward; the model itself smooths
                # any over-confident pixels on the next forward pass.
                next_fire = np.maximum(prev_fire, prob.astype(np.float32))
            else:
                growth = (prob >= MODEL_AR_FEEDBACK_THRESHOLD).astype(np.float32)
                next_fire = np.maximum(prev_fire, growth)
        else:
            next_fire = _resize_prob_to_shape(prob, current_dyn.shape[-2:])
        zero = np.zeros_like(next_fire, dtype=np.float32)
        channels = {
            "fire_t": np.clip(next_fire, 0.0, 1.0),
            "frp": zero,
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
        next_slice = np.stack([channels[name] for name in MODEL_DYNAMIC_ORDER], axis=0).astype(np.float32)
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


@app.get("/metrics")
def metrics():
    return Response(content=_metrics_text(), media_type="text/plain; version=0.0.4")


@app.get("/healthz")
def healthz():
    _ensure_model_metadata_loaded(load_checkpoint=False)
    model_exists = os.path.exists(MODEL_PATH)
    model_sha256 = file_sha256(MODEL_PATH)
    try:
        from .static_catalog import load_catalog
        static_catalog_ok = True
        static_catalog_error = None
        static_catalog_meta = load_catalog()
    except Exception as exc:
        static_catalog_ok = False
        static_catalog_error = str(exc)
        static_catalog_meta = {}
    return {
        "ok": True,
        "predictionsEnabled": PREDICTIONS_ENABLED,
        "modelPath": MODEL_PATH,
        "modelExists": model_exists,
        "modelSha256": model_sha256,
        "device": str(DEVICE),
        "Cd": MODEL_CD,
        "Cs": MODEL_CS,
        "hidden": MODEL_HIDDEN,
        "lstm_layers": MODEL_LSTM_LAYERS,
        "threshold": MODEL_THRESHOLD,
        "display_floor": PRED_DISPLAY_FLOOR,
        "Tseq": MODEL_TSEQ,
        "step_hours": MODEL_STEP_HOURS,
        "target_mode": MODEL_TARGET_MODE,
        "runtime_arch_version": RUNTIME_ARCH_VERSION,
        "dynamic_order": MODEL_DYNAMIC_ORDER,
        "static_order": MODEL_STATIC_ORDER,
        "staticCatalog": {
            "ok": static_catalog_ok,
            "error": static_catalog_error,
            "version": static_catalog_meta.get("version"),
            "path": static_catalog_meta.get("_path"),
            "extent": static_catalog_meta.get("extent"),
            "crs": static_catalog_meta.get("crs"),
            "resolution_m": static_catalog_meta.get("resolution_m"),
            "shape": static_catalog_meta.get("shape"),
            "storage": static_catalog_meta.get("storage"),
            "channels": sorted((static_catalog_meta.get("channels") or {}).keys()),
            "fuel_channels": static_catalog_meta.get("fuel_channels"),
        },
        "calibration": calibration_status(model_sha256=model_sha256),
        "firmsSnapshot": firms_snapshot_status(os.getenv("FIRMS_SNAPSHOT_DIR")),
        "noaaCycle": noaa_cycle_status(_noaa_cache_health_dir()),
        "weatherQuality": weather_quality_status(),
        "mlSource": source_version_info(),
    }


@app.get("/tile-bounds")
def tile_bounds_endpoint(lat: float = Query(...), lon: float = Query(...)):
    grid = build_grid(lat, lon)
    bounds = _extract_bounds_from_grid(grid, lat=lat, lon=lon)
    return {"bounds": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]} 


@app.get("/input_audit")
def input_audit(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(MODEL_TSEQ),
    step_hours: int = Query(MODEL_STEP_HOURS),
    ignition: bool = Query(False),
    date: str = Query(None),
    include_npz: bool = Query(False),
):
    ref_time = _parse_date_param(date)
    dyn, stat, bounds, base_time, static_summary = _prepare_prediction_inputs_with_summary(
        lat,
        lon,
        Tseq,
        ignition=ignition,
        ref_time=ref_time,
        hours_step=step_hours,
    )
    dyn_model, dyn_prep = _prepare_dynamic_for_model(dyn)
    payload = {
        "ok": True,
        "model_meta": _model_metadata(),
        "calibration": calibration_status(model_sha256=_model_sha256 or file_sha256(MODEL_PATH)),
        "input_summary": _tensor_input_summary(
            dyn,
            stat,
            bounds=bounds,
            base_time=base_time,
            dyn_model=dyn_model,
            dynamic_preparation=dyn_prep,
            static_summary=static_summary,
        ),
    }
    if include_npz:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            x_dyn=dyn_model.astype(np.float32),
            x_stat=stat.astype(np.float32),
            bounds=np.asarray(bounds, dtype=np.float32),
            dynamic_order=np.asarray(dyn_prep.get("model_dynamic_order") or [], dtype=object),
            static_order=np.asarray(MODEL_STATIC_ORDER, dtype=object),
            model_meta=np.asarray(json.dumps(payload["model_meta"], default=str), dtype=object),
        )
        payload["audit_npz_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        payload["audit_npz_filename"] = "ignis_input_audit.npz"
    return payload


@app.get("/predict")
def predict_png(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(MODEL_TSEQ),
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
def predict_raster_png(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(MODEL_TSEQ), date: str = Query(None)):
    ref_time = _parse_date_param(date)
    prob, bounds = _predict_probability(lat, lon, Tseq=Tseq, ref_time=ref_time)
    png = _prob_to_png(prob)
    headers = {"X-Bounds": ",".join(map(str, bounds))}
    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/predict_raster_json")
def predict_raster_json(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(MODEL_TSEQ),
    thr: float = Query(None),
    display_floor: float = Query(None),
    crop_frac: float = Query(0.5),
    ignition: bool = Query(False),
    date: str = Query(None),
):
    ref_time = _parse_date_param(date)
    dyn, stat, bounds, base_time, static_summary = _prepare_prediction_inputs_with_summary(
        lat, lon, Tseq, ignition=ignition, ref_time=ref_time
    )
    dyn_model, dyn_prep = _prepare_dynamic_for_model(dyn)
    input_summary = _tensor_input_summary(
        dyn,
        stat,
        bounds=bounds,
        base_time=base_time,
        dyn_model=dyn_model,
        dynamic_preparation=dyn_prep,
        static_summary=static_summary,
    )
    prob = _predict_probability_from_inputs(dyn, stat)
    crop_window = _build_crop_window(prob.shape, bounds, lat, lon, crop_frac)
    prob, bounds, anchor_px = _apply_crop_window(prob, crop_window)
    observed_fire_full = _observed_fire_from_dyn(dyn)
    observed_fire, _, _ = _apply_crop_window(observed_fire_full, crop_window)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    resolved_display_floor = _resolve_display_floor(display_floor)
    display_score, risk_class, calibration_meta = calibrate_probability(prob, model_sha256=_model_sha256 or file_sha256(MODEL_PATH))
    render_start = time.perf_counter()
    png = _prob_to_display_png(display_score, threshold=resolved_display_floor, anchor_px=anchor_px)
    _metric_observe("ignis_png_render_ms", (time.perf_counter() - render_start) * 1000.0)
    p_next_fire = _next_fire_from_delta(prob, observed_fire, threshold)
    layer_images = {
        "new_burn": base64.b64encode(png).decode("ascii"),
        "p_new_burn": base64.b64encode(png).decode("ascii"),
        "next_fire": _layer_image_base64(p_next_fire, threshold=0.5, anchor_px=anchor_px),
        "p_next_fire": _layer_image_base64(p_next_fire, threshold=0.5, anchor_px=anchor_px),
        "observed_fire": _layer_image_base64(observed_fire, threshold=0.5, anchor_px=anchor_px),
    }
    contract = _probability_contract(
        prob_new_burn=prob,
        observed_fire=observed_fire,
        threshold=threshold,
        display_score=display_score,
        risk_class=risk_class,
        calibration_meta=calibration_meta,
    )

    return {
        "bounds": list(map(float, bounds)),
        "coordinates": crop_window["coordinates"],
        "image_base64": layer_images["new_burn"],
        "layer_images": layer_images,
        "threshold": threshold,
        "display_floor": resolved_display_floor,
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean()),
        "prob_max": float(prob.max()),
        "area_fraction": float((prob >= threshold).mean()),
        "display_area_fraction": float((display_score >= resolved_display_floor).mean()),
        "probability_scale": {"mode": "absolute", "min": 0.0, "max": 1.0, "display_floor": resolved_display_floor},
        "model_meta": _model_metadata(),
        "input_summary": input_summary,
        **contract,
    }

@app.get("/predict_geojson")
def predict_geojson(
    lat: float = Query(...),
    lon: float = Query(...),
    Tseq: int = Query(MODEL_TSEQ),
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
def predict_raster_json_raw(lat: float = Query(...), lon: float = Query(...), Tseq: int = Query(MODEL_TSEQ), ignition: bool = Query(True), date: str = Query(None)):
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
    Tseq: int = Query(MODEL_TSEQ),
    steps: int = Query(6),
    step_hours: int = Query(MODEL_STEP_HOURS),
    thr: float = Query(None),
    display_floor: float = Query(None),
    crop_frac: float = Query(0.5),
    ignition: bool = Query(False),
    date: str = Query(None),
    debug: str = Query(
        None,
        description=(
            "Diagnostic modes, comma-separated. "
            "'solid' skips the model and returns a constant 0.5 probability in "
            "every cell (plumbing test for bounds/coordinates). "
            "'dump' runs the model normally and saves raw dyn/stat/prob arrays "
            "to IGNIS_DEBUG_DIR (default /tmp/ignis_debug) for offline analysis. "
            "Multiple modes may be combined, e.g. debug=solid,dump."
        ),
    ),
):
    ref_time = _parse_date_param(date)
    threshold = float(thr) if thr is not None else MODEL_THRESHOLD
    resolved_display_floor = _resolve_display_floor(display_floor)
    debug_modes = {tok.strip().lower() for tok in (debug or "").split(",") if tok.strip()}
    debug_solid = "solid" in debug_modes
    debug_dump = "dump" in debug_modes

    normalized_steps = max(1, int(steps))
    debug_sink: Dict[str, Any] = {}

    if debug_solid:
        # Step-1 plumbing sanity test: bypass the model entirely and produce a
        # constant probability field on the same input shape / bounds the real
        # pipeline would use. If the overlay renders as a fully-filled
        # translucent rectangle aligned with the crop bounds, the bounds and
        # coordinates pipeline is correct and the bug lives in the probability
        # field (model / inputs / AR feedback). If the overlay is not a clean
        # rectangle (e.g. a vertical strip), the bug lives in the
        # bounds/coordinates math.
        dyn, stat, bounds, base_time, static_summary = _prepare_prediction_inputs_with_summary(
            lat, lon, Tseq, ignition=ignition, ref_time=ref_time, hours_step=step_hours,
        )
        prob_shape = tuple(int(v) for v in dyn.shape[-2:])
        crop_window = _build_crop_window(prob_shape, bounds, lat, lon, crop_frac)
        rollout = [
            {
                "index": i,
                "lead_hours": (i + 1) * int(step_hours),
                "label": _step_label((i + 1) * int(step_hours)),
                "prob": np.full(prob_shape, 0.5, dtype=np.float32),
            }
            for i in range(normalized_steps)
        ]
        debug_sink["dyn"] = dyn
        debug_sink["stat"] = stat
        debug_sink["base_time"] = base_time
        debug_sink["static_summary"] = static_summary
    else:
        bounds, crop_window, rollout = _rollout_multistep_predictions(
            lat,
            lon,
            Tseq=Tseq,
            steps=normalized_steps,
            step_hours=step_hours,
            crop_frac=crop_frac,
            ignition=ignition,
            ref_time=ref_time,
            threshold=threshold,
            debug_sink=debug_sink,
        )

    payload_steps = []
    cropped_bounds = crop_window["bounds"]
    observed_fire_full = _observed_fire_from_dyn(debug_sink["dyn"]) if debug_sink.get("dyn") is not None else None
    display_mask_full, display_mask_summary = display_mask_from_static(
        debug_sink.get("stat"),
        list(MODEL_STATIC_ORDER),
        water_threshold=DISPLAY_MASK_WATER_THRESHOLD,
        impervious_threshold=DISPLAY_MASK_IMPERVIOUS_THRESHOLD,
    )
    for step in rollout:
        cropped, _, anchor_px = _apply_crop_window(step["prob"], crop_window)
        observed_crop = (
            _apply_crop_window(observed_fire_full, crop_window)[0]
            if observed_fire_full is not None
            else np.zeros_like(cropped, dtype=np.float32)
        )
        display_mask_crop = (
            _apply_crop_window(display_mask_full, crop_window)[0]
            if display_mask_full is not None
            else np.ones_like(cropped, dtype=np.float32)
        )
        step_display_mask = {
            **display_mask_summary,
            "masked_fraction": float((display_mask_crop <= 0.0).mean()),
        }
        if debug_solid:
            # Solid-fill PNG bypasses the radial alpha mask and thresholding
            # so the plumbing test isn't confounded by per-pixel masking.
            png = _encode_solid_png(cropped.shape, value=0.5)
            display_score = np.full(cropped.shape, 0.5, dtype=np.float32)
            risk_class = np.full(cropped.shape, "medium", dtype=object)
            calibration_meta = {"method": "debug_solid"}
        else:
            display_score, risk_class, calibration_meta = calibrate_probability(
                cropped,
                model_sha256=_model_sha256 or file_sha256(MODEL_PATH),
            )
            display_score = np.asarray(display_score, dtype=np.float32) * display_mask_crop
            render_start = time.perf_counter()
            png = _prob_to_display_png(display_score, threshold=resolved_display_floor, anchor_px=anchor_px)
            _metric_observe("ignis_png_render_ms", (time.perf_counter() - render_start) * 1000.0)
        p_next_fire = _next_fire_from_delta(cropped, observed_crop, threshold)
        p_next_fire_display = np.maximum(observed_crop, p_next_fire * display_mask_crop).astype(np.float32)
        image_base64 = base64.b64encode(png).decode("ascii")
        layer_images = {
            "new_burn": image_base64,
            "p_new_burn": image_base64,
            "next_fire": _layer_image_base64(p_next_fire_display, threshold=0.5, anchor_px=anchor_px),
            "p_next_fire": _layer_image_base64(p_next_fire_display, threshold=0.5, anchor_px=anchor_px),
            "observed_fire": _layer_image_base64(observed_crop, threshold=0.5, anchor_px=anchor_px),
        }
        contract = _probability_contract(
            prob_new_burn=cropped,
            observed_fire=observed_crop,
            threshold=threshold,
            display_score=display_score,
            risk_class=risk_class,
            calibration_meta=calibration_meta,
        )
        payload_steps.append({
            "index": step["index"],
            "lead_hours": step["lead_hours"],
            "label": step["label"],
            "image_base64": image_base64,
            "layer_images": layer_images,
            "prob_min": float(cropped.min()),
            "prob_mean": float(cropped.mean()),
            "prob_max": float(cropped.max()),
            "area_fraction": float((cropped >= threshold).mean()),
            "display_area_fraction": float((display_score >= resolved_display_floor).mean()),
            "display_floor": resolved_display_floor,
            "display_mask": step_display_mask,
            "threshold_override": thr is not None,
            "contour": _prob_to_geojson(cropped, cropped_bounds, threshold=threshold),
            "contour_50": _prob_to_geojson(cropped, cropped_bounds, threshold=0.5),
            **contract,
        })

    debug_payload: Dict[str, Any] = {}
    if debug_dump and debug_sink is not None:
        tag_parts = []
        if debug_solid:
            tag_parts.append("solid")
        tag_parts.append(f"lat{lat:.4f}")
        tag_parts.append(f"lon{lon:.4f}")
        tag = "_".join(tag_parts).replace("-", "m")
        out_dir = _debug_dump_dir(tag)
        _dump_multistep_artifacts(
            out_dir,
            dyn=debug_sink.get("dyn"),
            stat=debug_sink.get("stat"),
            rollout=rollout,
            crop_window=crop_window,
            bounds=bounds,
            ref_time=ref_time,
            lat=lat,
            lon=lon,
            step_hours=int(step_hours),
            Tseq=int(Tseq),
            threshold=threshold,
            ignition=bool(ignition),
        )
        debug_payload["dump_dir"] = str(out_dir)

    dyn_model, dyn_prep = _prepare_dynamic_for_model(debug_sink["dyn"])
    response: Dict[str, Any] = {
        "bounds": list(map(float, cropped_bounds)),
        "coordinates": crop_window["coordinates"],
        "threshold": threshold,
        "display_floor": resolved_display_floor,
        "step_hours": int(step_hours),
        "steps": payload_steps,
        "display_mask": display_mask_summary,
        "threshold_override": thr is not None,
        "probability_scale": {"mode": "absolute", "min": 0.0, "max": 1.0, "display_floor": resolved_display_floor},
        "model_meta": _model_metadata(),
        "input_summary": _tensor_input_summary(
            debug_sink["dyn"],
            debug_sink["stat"],
            bounds=bounds,
            base_time=debug_sink["base_time"],
            dyn_model=dyn_model,
            dynamic_preparation=dyn_prep,
            static_summary=debug_sink.get("static_summary"),
        ),
    }
    if debug_modes:
        debug_payload["modes"] = sorted(debug_modes)
        response["debug"] = debug_payload
    return response
