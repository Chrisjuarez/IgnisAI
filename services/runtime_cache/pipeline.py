from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from services.tilesvc.grid import CRS_ALBERS, SIZE, lonlat_to_tile, tile_affine, tile_bounds_lonlat


DEFAULT_RUNTIME_BUCKET = "s3://ignisai-static-chrisjuarez-2026/runtime"
DEFAULT_PROFILE = "palisades"
PALISADES_LAT = 34.05
PALISADES_LON = -118.55
PALISADES_REF_TIME = "2025-01-07T18:30:00Z"
DEFAULT_TSEQ = 6
DEFAULT_STEPS = 6
DEFAULT_STEP_HOURS = 24
REQUIRED_NOAA_CHANNELS = ("u", "v", "gust", "tempC", "q", "precip")
FIRMS_COLUMNS = ("latitude", "longitude", "acq_date", "acq_time", "frp")
DEFAULT_FIRMS_PRODUCTS = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP")
GFS_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


@dataclass(frozen=True)
class RuntimeBuildResult:
    profile: str
    firms_files: List[str]
    noaa_files: List[str]
    uploaded: bool


def parse_ref_time(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        out = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        out = dt.datetime.fromisoformat(text)
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def default_runtime_bucket_uri() -> str:
    return os.getenv("IGNIS_RUNTIME_CACHE_BUCKET", DEFAULT_RUNTIME_BUCKET).rstrip("/")


def parse_s3_uri(uri: str) -> S3Uri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix URI, got {uri!r}")
    return S3Uri(parsed.netloc, parsed.path.lstrip("/").rstrip("/"))


def s3_join(base_uri: str, *parts: str) -> str:
    parsed = parse_s3_uri(base_uri)
    key_parts = [p.strip("/") for p in (parsed.key, *parts) if p and p.strip("/")]
    return f"s3://{parsed.bucket}/{'/'.join(key_parts)}"


def cache_hours_for_multistep(
    ref_time: dt.datetime,
    *,
    t_seq: int = DEFAULT_TSEQ,
    steps: int = DEFAULT_STEPS,
    step_hours: int = DEFAULT_STEP_HOURS,
) -> List[dt.datetime]:
    """Return weather timestamps read by build_dynamic_for_tile plus rollout."""
    ref_time = parse_ref_time(ref_time)
    hours: List[dt.datetime] = []
    for idx in range(int(t_seq)):
        hours.append(ref_time - dt.timedelta(hours=(int(t_seq) - 1 - idx) * int(step_hours)))
    for idx in range(max(0, int(steps) - 1)):
        hours.append(ref_time + dt.timedelta(hours=(idx + 1) * int(step_hours)))
    deduped = {h.replace(minute=0, second=0, microsecond=0): None for h in hours}
    return sorted(deduped.keys())


def firms_snapshot_dates(ref_time: dt.datetime, *, t_seq: int = DEFAULT_TSEQ, step_hours: int = DEFAULT_STEP_HOURS) -> List[dt.date]:
    ref_time = parse_ref_time(ref_time)
    window_start = ref_time - dt.timedelta(hours=int(t_seq) * int(step_hours))
    day = window_start.date()
    end = ref_time.date()
    out: List[dt.date] = []
    while day <= end:
        out.append(day)
        day += dt.timedelta(days=1)
    return out


def noaa_cache_filename(hour: dt.datetime, lat: float, lon: float) -> str:
    hour = parse_ref_time(hour).replace(minute=0, second=0, microsecond=0)
    return f"{hour.strftime('%Y%m%dT%H')}_{float(lat):.2f}_{float(lon):.2f}.npz"


def runtime_s3_key(kind: str, profile: str, filename: str) -> str:
    return f"{kind.strip('/')}/{profile.strip('/')}/{filename}"


def _tile_bbox(lat: float, lon: float, *, pad_deg: float = 0.1) -> Tuple[float, float, float, float]:
    w, s, e, n = tile_bounds_lonlat(lonlat_to_tile(float(lon), float(lat)))
    return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)


def _firm_map_key(explicit: Optional[str] = None) -> str:
    key = explicit or os.getenv("NASA_API_KEY") or os.getenv("FIRMS_API_KEY") or os.getenv("NASA_FIRMS_MAP_KEY")
    return (key or "").strip()


def _parse_firms_csv(text: str) -> List[Dict[str, str]]:
    if not text.strip():
        return []
    first = text.splitlines()[0].lower()
    if "latitude" not in first or "longitude" not in first:
        return []
    rows: List[Dict[str, str]] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            acq_date = str(row["acq_date"]).strip()
            acq_time = str(row["acq_time"]).strip().zfill(4)
            frp = float(row.get("frp") or 0.0)
        except Exception:
            continue
        rows.append(
            {
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "acq_date": acq_date,
                "acq_time": acq_time,
                "frp": f"{max(0.0, frp):.6g}",
            }
        )
    return rows


def fetch_firms_rows_for_day(
    *,
    day: dt.date,
    bbox: Tuple[float, float, float, float],
    map_key: Optional[str] = None,
    products: Sequence[str] = DEFAULT_FIRMS_PRODUCTS,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, str]]:
    key = _firm_map_key(map_key)
    if not key:
        raise RuntimeError("NASA_API_KEY/FIRMS_API_KEY is required to build FIRMS runtime snapshots")
    session = session or requests.Session()
    area = ",".join(f"{float(v):.5f}" for v in bbox)
    rows: List[Dict[str, str]] = []
    for product in products:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
            f"{product}/{area}/1/{day.isoformat()}"
        )
        response = session.get(url, headers={"User-Agent": "ignis-ai-runtime-cache"}, timeout=45)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        rows.extend(_parse_firms_csv(response.text))

    # Deduplicate overlapping products.
    deduped: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        key_tuple = (row["latitude"], row["longitude"], row["acq_date"], row["acq_time"])
        deduped.setdefault(key_tuple, row)
    return sorted(deduped.values(), key=lambda r: (r["acq_date"], r["acq_time"], r["latitude"], r["longitude"]))


def write_firms_snapshot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIRMS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIRMS_COLUMNS})


def build_firms_snapshots(
    *,
    lat: float = PALISADES_LAT,
    lon: float = PALISADES_LON,
    ref_time: str | dt.datetime = PALISADES_REF_TIME,
    out_dir: Path,
    t_seq: int = DEFAULT_TSEQ,
    step_hours: int = DEFAULT_STEP_HOURS,
    map_key: Optional[str] = None,
    products: Sequence[str] = DEFAULT_FIRMS_PRODUCTS,
    overwrite: bool = False,
) -> List[Path]:
    ref = parse_ref_time(ref_time)
    bbox = _tile_bbox(lat, lon)
    session = requests.Session()
    written: List[Path] = []
    for day in firms_snapshot_dates(ref, t_seq=t_seq, step_hours=step_hours):
        path = out_dir / f"{day.isoformat()}.csv"
        if path.exists() and not overwrite:
            written.append(path)
            continue
        rows = fetch_firms_rows_for_day(day=day, bbox=bbox, map_key=map_key, products=products, session=session)
        write_firms_snapshot(path, rows)
        written.append(path)
    return written


@dataclass(frozen=True)
class GfsRecord:
    number: int
    start: int
    end: Optional[int]
    variable: str
    level: str
    forecast: str
    raw: str


def gfs_pgrb2_url(hour: dt.datetime, *, forecast_hour: int = 0) -> str:
    hour = parse_ref_time(hour)
    date = hour.strftime("%Y%m%d")
    cycle = f"{hour.hour:02d}"
    return (
        f"{GFS_AWS_BASE}/gfs.{date}/{cycle}/atmos/"
        f"gfs.t{cycle}z.pgrb2.0p25.f{int(forecast_hour):03d}"
    )


def parse_gfs_index(index_text: str) -> List[GfsRecord]:
    raw_records = []
    for line in index_text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        try:
            raw_records.append(
                {
                    "number": int(parts[0]),
                    "start": int(parts[1]),
                    "variable": parts[3],
                    "level": parts[4],
                    "forecast": parts[5],
                    "raw": line,
                }
            )
        except ValueError:
            continue
    out: List[GfsRecord] = []
    for idx, record in enumerate(raw_records):
        end = raw_records[idx + 1]["start"] if idx + 1 < len(raw_records) else None
        out.append(GfsRecord(end=end, **record))
    return out


def _matches_level(record: GfsRecord, expected: str) -> bool:
    return expected.lower() in record.level.lower()


def _select_gfs_records(records: Sequence[GfsRecord]) -> Dict[str, GfsRecord]:
    specs = {
        "u": ("UGRD", "10 m above ground"),
        "v": ("VGRD", "10 m above ground"),
        "gust": ("GUST", "surface"),
        "tempC": ("TMP", "2 m above ground"),
        "q": ("SPFH", "2 m above ground"),
        "precip": ("APCP", "surface"),
    }
    selected: Dict[str, GfsRecord] = {}
    for name, (variable, level) in specs.items():
        for record in records:
            if record.variable == variable and _matches_level(record, level):
                selected[name] = record
                break
    required = {"u", "v", "gust", "tempC", "q"}
    missing = sorted(required.difference(selected))
    if missing:
        raise RuntimeError(f"GFS index missing required weather records: {missing}")
    return selected


def _download_gfs_subset(hour: dt.datetime, work_dir: Path, *, session: Optional[requests.Session] = None) -> Path:
    session = session or requests.Session()
    base_url = gfs_pgrb2_url(hour)
    index_response = session.get(base_url + ".idx", headers={"User-Agent": "ignis-ai-runtime-cache"}, timeout=45)
    index_response.raise_for_status()
    selected = _select_gfs_records(parse_gfs_index(index_response.text))
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{parse_ref_time(hour).strftime('%Y%m%dT%H')}_gfs_subset.grib2"
    with out.open("wb") as f:
        for record in selected.values():
            end = "" if record.end is None else str(record.end - 1)
            headers = {"Range": f"bytes={record.start}-{end}", "User-Agent": "ignis-ai-runtime-cache"}
            response = session.get(base_url, headers=headers, timeout=120)
            response.raise_for_status()
            f.write(response.content)
    return out


def _band_tags(src: rasterio.io.DatasetReader, band: int) -> str:
    tags = src.tags(band)
    pieces = [src.descriptions[band - 1] or ""]
    for key in ("GRIB_ELEMENT", "GRIB_SHORT_NAME", "GRIB_COMMENT", "GRIB_REF_TIME", "GRIB_VALID_TIME"):
        pieces.append(str(tags.get(key, "")))
    return " ".join(pieces).upper()


def _find_band(src: rasterio.io.DatasetReader, *, variable: str, level: str) -> Optional[int]:
    var = variable.upper()
    lvl = level.upper()
    for band in range(1, src.count + 1):
        text = _band_tags(src, band)
        if var in text and lvl in text:
            return band
    for band in range(1, src.count + 1):
        if var in _band_tags(src, band):
            return band
    return None


def _reproject_band(src: rasterio.io.DatasetReader, band: int, *, lat: float, lon: float) -> np.ndarray:
    tile = lonlat_to_tile(lon, lat)
    dst = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(src, band),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs or "EPSG:4326",
        dst_transform=tile_affine(tile),
        dst_crs=CRS_ALBERS,
        resampling=Resampling.bilinear,
        src_nodata=src.nodata,
        dst_nodata=np.nan,
    )
    return dst.astype(np.float32)


def _read_gfs_subset_to_arrays(grib_path: Path, *, lat: float, lon: float) -> Dict[str, np.ndarray]:
    specs = {
        "u": ("UGRD", "10 m"),
        "v": ("VGRD", "10 m"),
        "gust": ("GUST", "surface"),
        "tempC": ("TMP", "2 m"),
        "q": ("SPFH", "2 m"),
        "precip": ("APCP", "surface"),
    }
    arrays: Dict[str, np.ndarray] = {}
    with rasterio.open(grib_path) as src:
        for name, (variable, level) in specs.items():
            band = _find_band(src, variable=variable, level=level)
            if band is None:
                if name == "precip":
                    arrays[name] = np.zeros((SIZE, SIZE), dtype=np.float32)
                    continue
                raise RuntimeError(f"GRIB subset did not expose {variable} {level}")
            arrays[name] = _reproject_band(src, band, lat=lat, lon=lon)
    if np.nanmean(arrays["tempC"]) > 150.0:
        arrays["tempC"] = arrays["tempC"] - 273.15
    for name in REQUIRED_NOAA_CHANNELS:
        arr = np.nan_to_num(arrays[name], nan=float(np.nanmean(arrays[name])) if np.isfinite(arrays[name]).any() else 0.0)
        arrays[name] = arr.astype(np.float32)
    return arrays


def validate_noaa_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        missing = [name for name in REQUIRED_NOAA_CHANNELS if name not in data]
        if missing:
            raise ValueError(f"{path} is missing NOAA cache channels: {missing}")
        stats: Dict[str, Any] = {"path": str(path), "keys": keys, "channels": {}}
        for name in REQUIRED_NOAA_CHANNELS:
            arr = np.asarray(data[name])
            if arr.shape != (SIZE, SIZE):
                raise ValueError(f"{path}:{name} has shape {arr.shape}, expected {(SIZE, SIZE)}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path}:{name} contains non-finite values")
            stats["channels"][name] = {
                "min": float(np.min(arr)),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
            }
    return stats


def write_noaa_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(arrays[name], dtype=np.float32) for name in REQUIRED_NOAA_CHANNELS}
    np.savez_compressed(path, **payload)
    validate_noaa_npz(path)


def build_gfs_npz_for_hour(
    *,
    hour: dt.datetime,
    lat: float = PALISADES_LAT,
    lon: float = PALISADES_LON,
    out_dir: Path,
    work_dir: Path,
    overwrite: bool = False,
    session: Optional[requests.Session] = None,
) -> Path:
    filename = noaa_cache_filename(hour, lat, lon)
    out = out_dir / filename
    if out.exists() and not overwrite:
        validate_noaa_npz(out)
        return out
    grib = _download_gfs_subset(hour, work_dir, session=session)
    arrays = _read_gfs_subset_to_arrays(grib, lat=lat, lon=lon)
    write_noaa_npz(out, arrays)
    return out


def build_noaa_cache(
    *,
    lat: float = PALISADES_LAT,
    lon: float = PALISADES_LON,
    ref_time: str | dt.datetime = PALISADES_REF_TIME,
    out_dir: Path,
    work_dir: Path,
    t_seq: int = DEFAULT_TSEQ,
    steps: int = DEFAULT_STEPS,
    step_hours: int = DEFAULT_STEP_HOURS,
    overwrite: bool = False,
) -> List[Path]:
    session = requests.Session()
    written: List[Path] = []
    for hour in cache_hours_for_multistep(parse_ref_time(ref_time), t_seq=t_seq, steps=steps, step_hours=step_hours):
        written.append(
            build_gfs_npz_for_hour(
                hour=hour,
                lat=lat,
                lon=lon,
                out_dir=out_dir,
                work_dir=work_dir,
                overwrite=overwrite,
                session=session,
            )
        )
    return written


def _s3_client():
    try:
        import boto3
    except ModuleNotFoundError:
        return None
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))


def _aws_cli_json(args: Sequence[str]) -> Any:
    proc = subprocess.run(["aws", *args, "--output", "json"], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout or "null")


def _upload_s3_file(path: Path, *, bucket: str, key: str, content_type: str) -> None:
    client = _s3_client()
    if client is not None:
        client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
        return
    subprocess.run(
        ["aws", "s3", "cp", str(path), f"s3://{bucket}/{key}", "--content-type", content_type],
        check=True,
    )


def _download_s3_file(*, bucket: str, key: str, target: Path) -> None:
    client = _s3_client()
    if client is not None:
        client.download_file(bucket, key, str(target))
        return
    subprocess.run(["aws", "s3", "cp", f"s3://{bucket}/{key}", str(target)], check=True)


def _list_s3_keys_with_fallback(*, bucket: str, prefix: str) -> List[str]:
    client = _s3_client()
    if client is not None:
        return _list_s3_keys(client, bucket=bucket, prefix=prefix)
    payload = _aws_cli_json(["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix])
    return [item["Key"] for item in payload.get("Contents", []) if not item["Key"].endswith("/")]


def upload_files(files: Iterable[Path], *, bucket_uri: str, kind: str, profile: str) -> List[str]:
    parsed = parse_s3_uri(bucket_uri)
    uris: List[str] = []
    for path in files:
        key_parts = [p for p in (parsed.key, runtime_s3_key(kind, profile, path.name)) if p]
        key = "/".join(key_parts)
        content_type = "application/octet-stream"
        if path.suffix.lower() == ".csv":
            content_type = "text/csv"
        _upload_s3_file(path, bucket=parsed.bucket, key=key, content_type=content_type)
        uris.append(f"s3://{parsed.bucket}/{key}")
    return uris


def build_palisades_runtime_cache(
    *,
    bucket_uri: str = DEFAULT_RUNTIME_BUCKET,
    profile: str = DEFAULT_PROFILE,
    lat: float = PALISADES_LAT,
    lon: float = PALISADES_LON,
    ref_time: str | dt.datetime = PALISADES_REF_TIME,
    out_dir: Path = Path(".cache/runtime_cache/palisades"),
    work_dir: Path = Path(".cache/runtime_cache/work"),
    upload: bool = True,
    build_firms: bool = True,
    build_noaa: bool = True,
    overwrite: bool = False,
    map_key: Optional[str] = None,
) -> RuntimeBuildResult:
    firms_dir = out_dir / "firms_snapshots"
    noaa_dir = out_dir / "noaa_grid_cache"
    firms_files: List[Path] = []
    noaa_files: List[Path] = []
    if build_firms:
        firms_files = build_firms_snapshots(lat=lat, lon=lon, ref_time=ref_time, out_dir=firms_dir, map_key=map_key, overwrite=overwrite)
    if build_noaa:
        noaa_files = build_noaa_cache(lat=lat, lon=lon, ref_time=ref_time, out_dir=noaa_dir, work_dir=work_dir, overwrite=overwrite)
    if upload:
        if firms_files:
            upload_files(firms_files, bucket_uri=bucket_uri, kind="firms_snapshots", profile=profile)
        if noaa_files:
            upload_files(noaa_files, bucket_uri=bucket_uri, kind="noaa_grid_cache", profile=profile)
    return RuntimeBuildResult(
        profile=profile,
        firms_files=[str(p) for p in firms_files],
        noaa_files=[str(p) for p in noaa_files],
        uploaded=upload,
    )


def _list_s3_keys(client: Any, *, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        keys.extend([item["Key"] for item in response.get("Contents", []) if not item["Key"].endswith("/")])
        if not response.get("IsTruncated"):
            return keys
        token = response.get("NextContinuationToken")


def sync_runtime_cache(
    *,
    bucket_uri: str = DEFAULT_RUNTIME_BUCKET,
    profile: str = DEFAULT_PROFILE,
    firms_dir: Path = Path("/data/firms_snapshots"),
    noaa_dir: Path = Path("/data/noaa_grid_cache"),
    required: bool = False,
) -> Dict[str, Any]:
    parsed = parse_s3_uri(bucket_uri)
    firms_dir.mkdir(parents=True, exist_ok=True)
    noaa_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "bucket": parsed.bucket,
        "profile": profile,
        "firms": {"downloaded": 0, "dir": str(firms_dir)},
        "noaa": {"downloaded": 0, "dir": str(noaa_dir)},
    }
    for kind, target_dir, suffix in (
        ("firms_snapshots", firms_dir, ".csv"),
        ("noaa_grid_cache", noaa_dir, ".npz"),
    ):
        prefix = "/".join(p for p in (parsed.key, kind, profile) if p).rstrip("/") + "/"
        keys = [key for key in _list_s3_keys_with_fallback(bucket=parsed.bucket, prefix=prefix) if key.lower().endswith(suffix)]
        for key in keys:
            target = target_dir / Path(key).name
            _download_s3_file(bucket=parsed.bucket, key=key, target=target)
            if suffix == ".npz":
                validate_noaa_npz(target)
        result["firms" if kind == "firms_snapshots" else "noaa"]["downloaded"] = len(keys)
    if required and (result["firms"]["downloaded"] == 0 or result["noaa"]["downloaded"] == 0):
        raise RuntimeError(f"Runtime cache sync did not download both FIRMS and NOAA files: {json.dumps(result)}")
    return result


def ensure_runtime_dirs(
    *,
    firms_dir: Path = Path("/data/firms_snapshots"),
    noaa_dir: Path = Path("/data/noaa_grid_cache"),
) -> Dict[str, str]:
    firms_dir.mkdir(parents=True, exist_ok=True)
    noaa_dir.mkdir(parents=True, exist_ok=True)
    return {"firms_dir": str(firms_dir), "noaa_dir": str(noaa_dir)}


def summarize_result(result: RuntimeBuildResult | Mapping[str, Any]) -> str:
    if isinstance(result, RuntimeBuildResult):
        payload: Mapping[str, Any] = {
            "profile": result.profile,
            "firms_files": result.firms_files,
            "noaa_files": result.noaa_files,
            "uploaded": result.uploaded,
        }
    else:
        payload = result
    return json.dumps(payload, indent=2, sort_keys=True)


def validate_runtime_dirs(*, firms_dir: Path, noaa_dir: Path) -> Dict[str, Any]:
    firms = sorted(firms_dir.glob("*.csv")) if firms_dir.exists() else []
    noaa = sorted(noaa_dir.glob("*.npz")) if noaa_dir.exists() else []
    noaa_stats = [validate_noaa_npz(path) for path in noaa]
    return {
        "firms_dir": str(firms_dir),
        "firms_count": len(firms),
        "firms_files": [path.name for path in firms],
        "noaa_dir": str(noaa_dir),
        "noaa_count": len(noaa),
        "noaa_files": [path.name for path in noaa],
        "noaa_stats": noaa_stats,
    }
