from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import rasterio
import requests
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.errors import DriverRegistrationError
from rasterio.warp import reproject

from services.tilesvc.grid import CRS_ALBERS, CRS_WGS84, PIX, TileID, lonlat_to_tile, tile_bounds_albers
from services.tilesvc.static_catalog import REQUIRED_BASE_STATIC


PIPELINE_VERSION = "v1"
DEFAULT_STATIC_PREFIX = "ignis/static"
DEFAULT_NODATA = -9999.0

_TO_ALBERS = Transformer.from_crs(CRS_WGS84, CRS_ALBERS, always_xy=True)
_TO_WGS84 = Transformer.from_crs(CRS_ALBERS, CRS_WGS84, always_xy=True)


@dataclass(frozen=True)
class ExtentSpec:
    name: str
    west: float
    south: float
    east: float
    north: float
    description: str

    def lonlat_bounds(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


@dataclass(frozen=True)
class TargetGrid:
    extent: ExtentSpec
    transform: Affine
    width: int
    height: int
    bounds_albers: Tuple[float, float, float, float]
    resolution_m: int = PIX
    crs: str = CRS_ALBERS


@dataclass(frozen=True)
class BuildOptions:
    extent: str = "western_conus"
    version: str = ""
    bucket_uri: Optional[str] = None
    source_config: Optional[Path] = None
    work_dir: Path = Path(".cache/static_pipeline")
    catalog_out: Optional[Path] = None
    upload: bool = True
    dry_run: bool = False
    catalog_uri_mode: str = "s3"
    allow_template_sources: bool = False
    custom_extent: Optional[ExtentSpec] = None


def default_version(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%d")


def default_bucket_uri() -> Optional[str]:
    bucket = os.getenv("IGNIS_STATIC_BUCKET")
    if not bucket:
        return None
    prefix = os.getenv("IGNIS_STATIC_PREFIX", DEFAULT_STATIC_PREFIX).strip("/")
    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def extent_for_name(name: str) -> ExtentSpec:
    normalized = name.strip().lower()
    if normalized == "western_conus":
        return ExtentSpec(
            name="western_conus",
            west=-125.1,
            south=31.0,
            east=-101.8,
            north=49.5,
            description="Western CONUS bbox covering CA, OR, WA, NV, AZ, UT, ID, MT, WY, CO, NM.",
        )
    if normalized == "california_mvp":
        return ExtentSpec(
            name="california_mvp",
            west=-125.0,
            south=32.0,
            east=-113.5,
            north=42.2,
            description="California and immediate border area for smoke tests.",
        )
    raise ValueError(f"Unsupported static extent {name!r}; expected western_conus or california_mvp")


def extent_for_tile(tile: TileID, *, name: str = "tile_fixture") -> ExtentSpec:
    minx, miny, maxx, maxy = tile_bounds_albers(tile)
    corners = [
        _TO_WGS84.transform(minx, miny),
        _TO_WGS84.transform(minx, maxy),
        _TO_WGS84.transform(maxx, miny),
        _TO_WGS84.transform(maxx, maxy),
    ]
    lons = [float(lon) for lon, _lat in corners]
    lats = [float(lat) for _lon, lat in corners]
    return ExtentSpec(
        name=name,
        west=min(lons),
        south=min(lats),
        east=max(lons),
        north=max(lats),
        description=f"Single Ignis tile fixture ix={tile.ix}, iy={tile.iy}.",
    )


def _densified_lonlat_edges(extent: ExtentSpec, samples: int = 48) -> Iterable[Tuple[float, float]]:
    lons = np.linspace(extent.west, extent.east, samples)
    lats = np.linspace(extent.south, extent.north, samples)
    for lon in lons:
        yield float(lon), float(extent.south)
        yield float(lon), float(extent.north)
    for lat in lats:
        yield float(extent.west), float(lat)
        yield float(extent.east), float(lat)


def target_grid_for_extent(extent: ExtentSpec, resolution_m: int = PIX) -> TargetGrid:
    xs = []
    ys = []
    for lon, lat in _densified_lonlat_edges(extent):
        x, y = _TO_ALBERS.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))
    minx = math.floor(min(xs) / resolution_m) * resolution_m
    miny = math.floor(min(ys) / resolution_m) * resolution_m
    maxx = math.ceil(max(xs) / resolution_m) * resolution_m
    maxy = math.ceil(max(ys) / resolution_m) * resolution_m
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target grid for extent {extent.name}: {width}x{height}")
    return TargetGrid(
        extent=extent,
        transform=Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy),
        width=width,
        height=height,
        bounds_albers=(minx, miny, maxx, maxy),
        resolution_m=resolution_m,
    )


def load_source_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Static source config does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    channels = data.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("Static source config must contain a channels object")
    missing = [name for name in REQUIRED_BASE_STATIC if name not in channels]
    if missing:
        raise ValueError(f"Static source config missing required channels: {missing}")
    return data


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix URI, got {uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def _target_s3_uri(bucket_uri: str, extent: str, version: str, filename: str) -> Tuple[str, str, str]:
    bucket, prefix = _parse_s3_uri(bucket_uri)
    key_parts = [part for part in (prefix, extent, PIPELINE_VERSION, version, filename) if part]
    key = "/".join(key_parts)
    return bucket, key, f"s3://{bucket}/{key}"


def _resampling(name: str) -> Resampling:
    n = str(name or "bilinear").lower()
    if n in {"nearest", "mode"}:
        return Resampling.nearest
    if n == "cubic":
        return Resampling.cubic
    return Resampling.bilinear


def _source_nodata(raw: Mapping[str, Any], src: rasterio.DatasetReader) -> Optional[float]:
    if raw.get("nodata") is not None:
        return float(raw["nodata"])
    if src.nodata is not None:
        return float(src.nodata)
    return None


def _read_raster_to_grid(raw: Mapping[str, Any], grid: TargetGrid, *, resampling: str) -> np.ndarray:
    uri = raw.get("uri") or raw.get("url") or raw.get("path")
    if not uri:
        raise ValueError("Raster source requires uri/url/path")
    band = int(raw.get("band", 1))
    dst = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    with rasterio.open(str(uri)) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=_source_nodata(raw, src),
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=_resampling(resampling),
        )
    scale = float(raw.get("scale", 1.0))
    offset = float(raw.get("offset", 0.0))
    if scale != 1.0 or offset != 0.0:
        dst = dst * scale + offset
    return dst.astype(np.float32)


def _constant_to_grid(raw: Mapping[str, Any], grid: TargetGrid) -> np.ndarray:
    return np.full((grid.height, grid.width), float(raw.get("value", 0.0)), dtype=np.float32)


def _categorical_mask_to_grid(raw: Mapping[str, Any], grid: TargetGrid) -> np.ndarray:
    values = set(int(v) for v in raw.get("classes", [11]))
    true_value = float(raw.get("true_value", 100.0))
    false_value = float(raw.get("false_value", 0.0))
    src_arr = _read_raster_to_grid(raw, grid, resampling="nearest")
    rounded = np.rint(src_arr).astype(np.int32)
    mask = np.isfinite(src_arr) & np.isin(rounded, list(values))
    out = np.where(mask, true_value, false_value).astype(np.float32)
    out[~np.isfinite(src_arr)] = np.nan
    return out


def _fetch_opentopo_dem(raw: Mapping[str, Any], grid: TargetGrid) -> np.ndarray:
    api_key = raw.get("api_key") or os.getenv("OPENTOPO_API_KEY")
    demtype = str(raw.get("demtype") or os.getenv("OT_DEM") or "SRTMGL1")
    params = {
        "demtype": demtype,
        "south": grid.extent.south,
        "north": grid.extent.north,
        "west": grid.extent.west,
        "east": grid.extent.east,
        "outputFormat": "GTiff",
    }
    if api_key:
        params["API_Key"] = api_key
    timeout = int(raw.get("timeout_seconds", os.getenv("OT_TIMEOUT", "300")))
    response = requests.get("https://portal.opentopography.org/API/globaldem", params=params, timeout=timeout)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        tmp.write(response.content)
        tmp.flush()
        return _read_raster_to_grid({"uri": tmp.name, "nodata": raw.get("nodata")}, grid, resampling="bilinear")


def _landfire_component(raw: Mapping[str, Any], grid: TargetGrid) -> np.ndarray:
    component = str(raw.get("component", "")).lower()
    if component not in {"fuel1", "fuel2", "fuel3"}:
        raise ValueError("landfire_fuel_component source requires component=fuel1|fuel2|fuel3")
    src_arr = _read_raster_to_grid(raw, grid, resampling="nearest")
    codes = np.rint(src_arr).astype(np.int32)
    out = np.full(src_arr.shape, np.nan, dtype=np.float32)

    nonburnable = np.isin(codes, [91, 92, 93, 98, 99])
    grass = (codes >= 101) & (codes <= 109)
    grass_shrub = (codes >= 121) & (codes <= 149)
    timber = ((codes >= 161) & (codes <= 189))
    slash = (codes >= 201) & (codes <= 204)

    if component == "fuel1":
        out[nonburnable] = -3.0
        out[grass] = 1.5
        out[grass_shrub] = 1.0
        out[timber] = -0.5
        out[slash] = 0.5
    elif component == "fuel2":
        out[nonburnable] = -3.0
        out[grass] = -0.5
        out[grass_shrub] = 1.2
        out[timber] = 0.5
        out[slash] = 1.5
    else:
        out[nonburnable] = -8.0
        out[grass] = -1.0
        out[grass_shrub] = -0.5
        out[timber] = 1.5
        out[slash] = 1.0

    out[np.isfinite(src_arr) & np.isnan(out)] = 0.0
    return out.astype(np.float32)


def build_channel_array(name: str, raw: Mapping[str, Any], grid: TargetGrid) -> np.ndarray:
    source_type = str(raw.get("type") or raw.get("source_type") or "raster").lower()
    if source_type == "constant":
        return _constant_to_grid(raw, grid)
    if source_type == "raster":
        return _read_raster_to_grid(raw, grid, resampling=str(raw.get("resampling", "bilinear")))
    if source_type in {"categorical_mask", "nlcd_water"}:
        return _categorical_mask_to_grid(raw, grid)
    if source_type in {"opentopography_globaldem", "opentopo_dem"}:
        return _fetch_opentopo_dem(raw, grid)
    if source_type in {"landfire_fuel_component", "landfire_candidate"}:
        merged = {**raw, "component": raw.get("component") or name}
        return _landfire_component(merged, grid)
    raise ValueError(f"Unsupported source type for channel {name!r}: {source_type!r}")


def _block_size(width: int, height: int) -> int:
    side = min(width, height, 512)
    if side >= 512:
        return 512
    if side >= 256:
        return 256
    if side >= 128:
        return 128
    if side >= 64:
        return 64
    return 16


def write_cog(
    path: Path,
    arr: np.ndarray,
    grid: TargetGrid,
    *,
    nodata: float = DEFAULT_NODATA,
    tags: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float32)
    data = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    block = _block_size(grid.width, grid.height)
    common: Dict[str, Any] = {
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
    }
    tag_values = {str(k): json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in (tags or {}).items()}
    try:
        with rasterio.open(
            path,
            "w",
            driver="COG",
            compress="DEFLATE",
            blocksize=block,
            overview_resampling="nearest",
            **common,
        ) as dst:
            dst.write(data, 1)
            if tag_values:
                dst.update_tags(**tag_values)
    except (DriverRegistrationError, rasterio.errors.RasterioIOError):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            tiled=True,
            blockxsize=block,
            blockysize=block,
            compress="deflate",
            BIGTIFF="IF_SAFER",
            **common,
        ) as dst:
            dst.write(data, 1)
            if min(grid.width, grid.height) >= 64:
                dst.build_overviews([2, 4, 8], Resampling.nearest)
            if tag_values:
                dst.update_tags(**tag_values)


def _finite_summary(arr: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"finite_ratio": 0.0, "min": None, "mean": None, "max": None, "pct_zero": None}
    vals = arr[finite]
    return {
        "finite_ratio": float(finite.mean()),
        "min": float(vals.min()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "pct_zero": float((vals == 0).mean()),
    }


def validate_channel_array(name: str, arr: np.ndarray, raw: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _finite_summary(arr)
    if summary["finite_ratio"] < float(raw.get("min_finite_ratio", 0.90)):
        raise ValueError(f"Channel {name!r} has insufficient finite coverage: {summary['finite_ratio']:.3f}")
    if name != "water" and summary["pct_zero"] is not None and summary["pct_zero"] > float(raw.get("max_zero_ratio", 0.999)):
        raise ValueError(f"Channel {name!r} looks like an all-zero placeholder")
    valid_range = raw.get("valid_range")
    if isinstance(valid_range, list) and len(valid_range) == 2 and summary["finite_ratio"] > 0:
        finite = np.isfinite(arr)
        vals = arr[finite]
        lo, hi = float(valid_range[0]), float(valid_range[1])
        outside = float(((vals < lo) | (vals > hi)).mean())
        if outside > float(raw.get("max_range_outside_ratio", 0.05)):
            raise ValueError(f"Channel {name!r} has too many values outside valid_range {valid_range}: {outside:.3f}")
        summary["outside_valid_range_ratio"] = outside
    return summary


def _catalog_channel_entry(uri: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "uri": uri,
        "units": raw.get("units"),
        "nodata": raw.get("output_nodata", DEFAULT_NODATA),
        "valid_range": raw.get("valid_range"),
        "crs": CRS_ALBERS,
        "resampling": raw.get("runtime_resampling", raw.get("resampling", "bilinear")),
        "source": raw.get("source", {}),
    }
    for key in ("quality", "parity_status", "candidate", "notes"):
        if key in raw:
            entry[key] = raw[key]
    return entry


def _validate_source_config_for_build(config: Mapping[str, Any], allow_template_sources: bool) -> None:
    if allow_template_sources:
        return
    for name in REQUIRED_BASE_STATIC:
        raw = config["channels"][name]
        marker = str(raw.get("uri") or raw.get("url") or raw.get("path") or "")
        if "REPLACE_" in marker or marker.startswith("s3://REPLACE"):
            raise ValueError(
                f"Channel {name!r} still contains a template URI. "
                "Copy config/static_sources.western_conus.example.json to a real config and fill source URIs first."
            )


def _upload_file(path: Path, bucket: str, key: str) -> None:
    import boto3

    client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    extra_args = {
        "ContentType": "image/tiff" if path.suffix.lower() in {".tif", ".tiff"} else "application/json",
    }
    client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)


def build_static_pipeline(options: BuildOptions) -> Dict[str, Any]:
    version = options.version or default_version()
    bucket_uri = options.bucket_uri or default_bucket_uri()
    if options.catalog_uri_mode == "s3" and not bucket_uri:
        raise ValueError("bucket_uri is required for catalog_uri_mode='s3'")

    extent = options.custom_extent or extent_for_name(options.extent)
    grid = target_grid_for_extent(extent)
    config_path = options.source_config or Path(f"config/static_sources.{extent.name}.json")
    if not config_path.exists():
        example = config_path.with_name(f"{config_path.stem}.example{config_path.suffix}")
        if example.exists():
            config_path = example
    config = load_source_config(config_path)
    _validate_source_config_for_build(config, options.allow_template_sources or options.dry_run)

    local_dir = options.work_dir / extent.name / PIPELINE_VERSION / version
    local_dir.mkdir(parents=True, exist_ok=True)
    catalog_channels: Dict[str, Any] = {}
    channel_summaries: Dict[str, Any] = {}

    if bucket_uri:
        _bucket, _prefix = _parse_s3_uri(bucket_uri)

    for name in REQUIRED_BASE_STATIC:
        raw = dict(config["channels"][name])
        filename = f"{name}.tif"
        local_path = local_dir / filename
        if bucket_uri:
            bucket, key, uri = _target_s3_uri(bucket_uri, extent.name, version, filename)
        else:
            bucket, key, uri = "", "", str(local_path.resolve())

        if options.catalog_uri_mode == "local":
            uri = str(local_path.resolve())

        if options.dry_run:
            channel_summaries[name] = {"planned_uri": uri, "source_type": raw.get("type", raw.get("source_type", "raster"))}
            catalog_channels[name] = _catalog_channel_entry(uri, raw)
            continue

        arr = build_channel_array(name, raw, grid)
        channel_summaries[name] = validate_channel_array(name, arr, raw)
        write_cog(
            local_path,
            arr,
            grid,
            nodata=float(raw.get("output_nodata", DEFAULT_NODATA)),
            tags={
                "ignis_channel": name,
                "ignis_static_version": version,
                "ignis_extent": extent.name,
                "source": raw.get("source", {}),
                "quality": raw.get("quality", "production"),
                "parity_status": raw.get("parity_status", ""),
            },
        )
        if options.upload:
            if not bucket_uri:
                raise ValueError("Cannot upload without bucket_uri")
            _upload_file(local_path, bucket, key)
        catalog_channels[name] = _catalog_channel_entry(uri, raw)

    catalog = {
        "version": f"{extent.name}-{PIPELINE_VERSION}-{version}",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "extent": {
            "name": extent.name,
            "description": extent.description,
            "lonlat_bounds": list(extent.lonlat_bounds()),
            "albers_bounds": list(grid.bounds_albers),
        },
        "crs": grid.crs,
        "resolution_m": grid.resolution_m,
        "shape": {"height": grid.height, "width": grid.width},
        "storage": {"bucket_uri": bucket_uri, "uploaded": bool(options.upload and not options.dry_run)},
        "channels": catalog_channels,
        "fuel_channels": {
            "status": "candidate",
            "strict_production_blocked_until": "fuel/static parity audit passes against training tiles",
        },
    }

    catalog_path = options.catalog_out or (local_dir / "static_catalog.production.json")
    if not options.dry_run:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if options.upload and bucket_uri:
            bucket, key, _uri = _target_s3_uri(bucket_uri, extent.name, version, "static_catalog.production.json")
            _upload_file(catalog_path, bucket, key)

    return {
        "ok": True,
        "dry_run": options.dry_run,
        "extent": extent.name,
        "version": version,
        "grid": {"height": grid.height, "width": grid.width, "bounds_albers": list(grid.bounds_albers)},
        "source_config": str(config_path),
        "local_dir": str(local_dir),
        "catalog_path": str(catalog_path),
        "catalog": catalog,
        "channel_summaries": channel_summaries,
    }


def tile_build_options(
    *,
    lon: float,
    lat: float,
    source_config: Path,
    work_dir: Path,
    version: str = "test",
    catalog_out: Optional[Path] = None,
) -> Tuple[BuildOptions, ExtentSpec]:
    tile = lonlat_to_tile(lon, lat)
    extent = extent_for_tile(tile)
    options = BuildOptions(
        extent=extent.name,
        version=version,
        bucket_uri=None,
        source_config=source_config,
        work_dir=work_dir,
        catalog_out=catalog_out,
        upload=False,
        catalog_uri_mode="local",
        custom_extent=extent,
    )
    return options, extent
