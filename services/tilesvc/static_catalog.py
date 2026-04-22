from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .grid import CRS_ALBERS, PIX, SIZE, tile_affine


REQUIRED_BASE_STATIC = (
    "elev",
    "ndvi",
    "bi",
    "erc",
    "pdsi",
    "chili",
    "impervious",
    "water",
    "population",
    "fuel1",
    "fuel2",
    "fuel3",
)

DERIVED_STATIC = ("slope", "aspect_cos", "aspect_sin")

STATIC_NORMALIZATION: Dict[str, Tuple[str, Tuple[float, ...] | None]] = {
    "elev": ("minmax", (0, 4000)),
    "slope": ("minmax", (0, 45)),
    "aspect_cos": ("pm1_to_01", None),
    "aspect_sin": ("pm1_to_01", None),
    "ndvi": ("minmax", (0, 1)),
    "bi": ("minmax", (0, 200)),
    "erc": ("minmax", (0, 100)),
    "pdsi": ("shift_scale", (-10, 10)),
    "chili": ("minmax", (0, 255)),
    "impervious": ("minmax", (0, 100)),
    "water": ("minmax", (0, 100)),
    "population": ("log1p_div", (1000,)),
    "fuel1": ("clip_minmax", (-3, 3)),
    "fuel2": ("clip_minmax", (-3, 3)),
    "fuel3": ("clip_minmax", (-8, 3)),
}

DEFAULT_STATIC_PLACEHOLDERS: Dict[str, float] = {
    "elev": 0.25,
    "slope": 0.0,
    "aspect_cos": 0.5,
    "aspect_sin": 0.5,
    "ndvi": 0.5,
    "bi": 0.25,
    "erc": 0.5,
    "pdsi": 0.5,
    "chili": 0.5,
    "impervious": 0.0,
    "water": 0.0,
    "population": 0.0,
    "fuel1": 0.5,
    "fuel2": 0.5,
    "fuel3": 0.5,
}


class InputUnavailable(RuntimeError):
    def __init__(self, message: str, *, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


@dataclass(frozen=True)
class StaticChannel:
    name: str
    uri: str
    units: Optional[str] = None
    nodata: Any = None
    valid_range: Optional[Tuple[float, float]] = None
    crs: Optional[str] = None
    resampling: str = "bilinear"
    metadata: Optional[Dict[str, Any]] = None


def _resolve_catalog_path() -> Optional[Path]:
    raw = os.getenv("STATIC_CATALOG_PATH")
    if not raw:
        return None
    return Path(raw)


def _static_catalog_required() -> bool:
    return os.getenv("STATIC_CATALOG_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _placeholder_static_tensor(
    expected_order: Iterable[str],
    *,
    reason: str,
    detail: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    normalized = []
    channels: Dict[str, Any] = {}
    for name in expected_order:
        value = float(DEFAULT_STATIC_PLACEHOLDERS.get(name, 0.5))
        normalized.append(np.full((SIZE, SIZE), value, dtype=np.float32))
        channels[name] = {
            "min": value,
            "mean": value,
            "max": value,
            "finite_ratio": 1.0,
            "pct_zero": 1.0 if value == 0.0 else 0.0,
            "placeholder_or_missing": True,
            "reason": reason,
        }
    tensor = np.stack(normalized, axis=0).astype(np.float32)
    return tensor, {
        "catalog": {
            "path": str(_resolve_catalog_path()) if _resolve_catalog_path() else None,
            "version": "placeholder",
            "generated_at": None,
            "placeholder": True,
            "reason": reason,
            "detail": detail,
        },
        "channels": channels,
        "derived": list(DERIVED_STATIC),
        "stat_shape": list(tensor.shape),
        "placeholder_or_missing": True,
    }


def load_catalog(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else _resolve_catalog_path()
    if not p:
        raise InputUnavailable(
            "STATIC_CATALOG_PATH is required for production predictions",
            reason="static_catalog_missing",
        )
    if not p.exists():
        raise InputUnavailable(
            f"Static catalog does not exist: {p}",
            reason="static_catalog_missing",
            details={"path": str(p)},
        )
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    channels = data.get("channels")
    if not isinstance(channels, Mapping):
        raise InputUnavailable(
            "Static catalog must contain a channels object",
            reason="static_catalog_invalid",
            details={"path": str(p)},
        )
    missing = [name for name in REQUIRED_BASE_STATIC if name not in channels]
    if missing:
        raise InputUnavailable(
            f"Static catalog missing required channels: {missing}",
            reason="static_catalog_missing_channels",
            details={"missing": missing, "path": str(p)},
        )
    data["_path"] = str(p)
    return data


def catalog_version(catalog: Mapping[str, Any]) -> Dict[str, Any]:
    channels = catalog.get("channels") if isinstance(catalog.get("channels"), Mapping) else {}
    fuel_channels = {
        name: raw
        for name, raw in channels.items()
        if name in {"fuel1", "fuel2", "fuel3"} and isinstance(raw, Mapping)
    }
    return {
        "path": catalog.get("_path"),
        "version": catalog.get("version"),
        "generated_at": catalog.get("generated_at"),
        "extent": catalog.get("extent"),
        "crs": catalog.get("crs"),
        "resolution_m": catalog.get("resolution_m"),
        "shape": catalog.get("shape"),
        "storage": catalog.get("storage"),
        "fuel_channels": catalog.get("fuel_channels"),
        "fuel_channel_status": {
            name: {
                "quality": raw.get("quality"),
                "parity_status": raw.get("parity_status"),
                "candidate": raw.get("candidate"),
            }
            for name, raw in fuel_channels.items()
        },
    }


def _parse_channel(name: str, raw: Mapping[str, Any]) -> StaticChannel:
    uri = raw.get("uri") or raw.get("url") or raw.get("path")
    if not uri:
        raise InputUnavailable(
            f"Static catalog channel {name!r} is missing uri",
            reason="static_catalog_invalid_channel",
            details={"channel": name},
        )
    valid_range = raw.get("valid_range")
    parsed_range = None
    if isinstance(valid_range, list) and len(valid_range) == 2:
        parsed_range = (float(valid_range[0]), float(valid_range[1]))
    return StaticChannel(
        name=name,
        uri=str(uri),
        units=raw.get("units"),
        nodata=raw.get("nodata"),
        valid_range=parsed_range,
        crs=raw.get("crs"),
        resampling=str(raw.get("resampling") or "bilinear").lower(),
        metadata={
            key: raw.get(key)
            for key in ("source", "quality", "parity_status", "candidate", "notes")
            if key in raw
        },
    )


def _resampling(name: str) -> Resampling:
    if name in {"nearest", "mode"}:
        return Resampling.nearest
    if name in {"cubic"}:
        return Resampling.cubic
    return Resampling.bilinear


def _read_channel(channel: StaticChannel, tile) -> np.ndarray:
    dst = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    try:
        with rasterio.open(channel.uri) as src:
            if channel.crs and src.crs and str(src.crs) != str(channel.crs):
                raise InputUnavailable(
                    f"Static channel {channel.name!r} CRS mismatch",
                    reason="static_channel_crs_mismatch",
                    details={"channel": channel.name, "expected": channel.crs, "actual": str(src.crs)},
                )
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tile_affine(tile),
                dst_crs=CRS_ALBERS,
                dst_nodata=np.nan,
                resampling=_resampling(channel.resampling),
            )
    except InputUnavailable:
        raise
    except Exception as exc:
        raise InputUnavailable(
            f"Failed to read static channel {channel.name!r}: {exc}",
            reason="static_channel_read_failed",
            details={"channel": channel.name, "uri": channel.uri, "error": str(exc)},
        ) from exc

    if channel.nodata is not None:
        try:
            dst = np.where(dst == float(channel.nodata), np.nan, dst)
        except Exception:
            pass
    return dst.astype(np.float32)


def _validate_channel(name: str, arr: np.ndarray, channel: StaticChannel) -> Dict[str, Any]:
    finite = np.isfinite(arr)
    finite_ratio = float(finite.mean()) if arr.size else 0.0
    if finite_ratio < float(os.getenv("STATIC_MIN_FINITE_RATIO", "0.90")):
        raise InputUnavailable(
            f"Static channel {name!r} has insufficient finite coverage",
            reason="static_channel_nodata_heavy",
            details={"channel": name, "finite_ratio": finite_ratio},
        )
    vals = arr[finite]
    if vals.size == 0:
        raise InputUnavailable(
            f"Static channel {name!r} has no finite values",
            reason="static_channel_empty",
            details={"channel": name},
        )
    pct_zero = float((vals == 0).mean())
    if name != "water" and pct_zero > float(os.getenv("STATIC_MAX_ZERO_RATIO", "0.999")):
        raise InputUnavailable(
            f"Static channel {name!r} appears to be an all-zero placeholder",
            reason="static_channel_placeholder",
            details={"channel": name, "pct_zero": pct_zero},
        )
    if channel.valid_range:
        lo, hi = channel.valid_range
        outside = float(((vals < lo) | (vals > hi)).mean())
        if outside > float(os.getenv("STATIC_MAX_RANGE_OUTSIDE_RATIO", "0.05")):
            raise InputUnavailable(
                f"Static channel {name!r} has too many values outside valid_range",
                reason="static_channel_range_invalid",
                details={"channel": name, "outside_ratio": outside, "valid_range": [lo, hi]},
            )
    return {
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
        "finite_ratio": finite_ratio,
        "pct_zero": pct_zero,
        "units": channel.units,
        "uri": channel.uri,
        "source": (channel.metadata or {}).get("source"),
        "quality": (channel.metadata or {}).get("quality"),
        "parity_status": (channel.metadata or {}).get("parity_status"),
        "candidate": (channel.metadata or {}).get("candidate"),
        "notes": (channel.metadata or {}).get("notes"),
    }


def _slope_aspect(elev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dem = np.nan_to_num(elev.astype(np.float32), nan=float(np.nanmean(elev)))
    dy, dx = np.gradient(dem, float(PIX), float(PIX))
    slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)
    aspect_rad = np.arctan2(-dx, -dy)
    return slope_deg, np.cos(aspect_rad).astype(np.float32), np.sin(aspect_rad).astype(np.float32)


def _to01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def normalize_static(name: str, arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mode, spec = STATIC_NORMALIZATION.get(name, ("minmax", (0.0, 1.0)))
    if mode == "minmax":
        lo, hi = spec  # type: ignore[misc]
        return _to01(a, float(lo), float(hi))
    if mode == "clip_minmax":
        lo, hi = spec  # type: ignore[misc]
        return _to01(np.clip(a, float(lo), float(hi)), float(lo), float(hi))
    if mode == "pm1_to_01":
        return np.clip((a + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    if mode == "shift_scale":
        lo, hi = spec  # type: ignore[misc]
        return np.clip((a - float(lo)) / (float(hi) - float(lo) + 1e-8), 0.0, 1.0).astype(np.float32)
    if mode == "log1p_div":
        (denom,) = spec  # type: ignore[misc]
        return np.clip(np.log1p(np.maximum(a, 0.0)) / np.log1p(float(denom)), 0.0, 1.0).astype(np.float32)
    return _to01(a, 0.0, 1.0)


def load_static_tensor_for_model(tile, expected_order: Iterable[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    expected_order = list(expected_order)
    try:
        catalog = load_catalog()
    except InputUnavailable as exc:
        if _static_catalog_required():
            raise
        return _placeholder_static_tensor(expected_order, reason=exc.reason, detail=str(exc))
    raw_channels = catalog["channels"]
    arrays: Dict[str, np.ndarray] = {}
    summary: Dict[str, Any] = {
        "catalog": catalog_version(catalog),
        "channels": {},
        "derived": list(DERIVED_STATIC),
    }

    for name in REQUIRED_BASE_STATIC:
        channel = _parse_channel(name, raw_channels[name])
        arr = _read_channel(channel, tile)
        summary["channels"][name] = _validate_channel(name, arr, channel)
        arrays[name] = arr

    slope, ac, asin = _slope_aspect(arrays["elev"])
    arrays["slope"] = slope
    arrays["aspect_cos"] = ac
    arrays["aspect_sin"] = asin

    normalized = []
    missing = []
    for name in expected_order:
        arr = arrays.get(name)
        if arr is None:
            missing.append(name)
            continue
        normalized.append(normalize_static(name, arr))
    if missing:
        raise InputUnavailable(
            f"Static tensor missing expected channels: {missing}",
            reason="static_tensor_missing_channels",
            details={"missing": missing},
        )
    tensor = np.stack(normalized, axis=0).astype(np.float32)
    summary["stat_shape"] = list(tensor.shape)
    return tensor, summary
