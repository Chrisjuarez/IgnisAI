from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import (
    DEFAULT_PROFILE,
    DEFAULT_RUNTIME_BUCKET,
    DEFAULT_STEP_HOURS,
    DEFAULT_STEPS,
    DEFAULT_TSEQ,
    DEFAULT_WEATHER_SOURCE_PRIORITY,
    PALISADES_LAT,
    PALISADES_LON,
    PALISADES_REF_TIME,
    build_event_runtime_cache,
    build_palisades_runtime_cache,
    default_runtime_bucket_uri,
    ensure_runtime_dirs,
    summarize_result,
    sync_runtime_cache,
    validate_runtime_dirs,
)


def _path_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _cache_root() -> Path:
    """Where runtime caches and grib intermediates live.

    Overridable because these do not belong on a small internal disk. A
    backfill downloads one grib per hour per fire at roughly 130 MB, and while
    those are now discarded after extraction, the npz caches still accumulate.
    Set IGNIS_CACHE_ROOT to an external volume to keep both off the boot drive.
    """
    return Path(os.getenv("IGNIS_CACHE_ROOT", ".cache/runtime_cache"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m services.runtime_cache",
        description="Build and sync S3-backed runtime FIRMS/NOAA caches for IgnisAI tilesvc.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    default_priority = ",".join(DEFAULT_WEATHER_SOURCE_PRIORITY)

    build = sub.add_parser("build-palisades", help="Build the Palisades FIRMS/NOAA runtime cache.")
    build.add_argument("--bucket", default=default_runtime_bucket_uri(), help=f"S3 runtime prefix, default {DEFAULT_RUNTIME_BUCKET}.")
    build.add_argument("--profile", default=DEFAULT_PROFILE)
    build.add_argument("--lat", type=float, default=PALISADES_LAT)
    build.add_argument("--lon", type=float, default=PALISADES_LON)
    build.add_argument("--ref-time", default=PALISADES_REF_TIME)
    build.add_argument("--out-dir", type=Path, default=_cache_root() / "palisades")
    build.add_argument("--work-dir", type=Path, default=_cache_root() / "work")
    build.add_argument("--no-upload", action="store_true")
    build.add_argument("--skip-firms", action="store_true")
    build.add_argument("--skip-noaa", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--map-key", default=None, help="NASA FIRMS MAP_KEY. Defaults to NASA_API_KEY/FIRMS_API_KEY.")
    build.add_argument(
        "--source-priority",
        default=default_priority,
        help=(
            "Comma-separated weather source priority. Supported: hrrr,gfs. "
            f"Default {default_priority} (try HRRR 3km first, fall back to GFS 0.25 deg)."
        ),
    )

    event = sub.add_parser(
        "build-event",
        help="Build the FIRMS/NOAA runtime cache for an arbitrary historical fire (Eaton, Camp, Dixie, ...).",
    )
    event.add_argument("--profile", required=True, help="Profile name; doubles as the S3 prefix and out-dir folder.")
    event.add_argument("--lat", type=float, required=True)
    event.add_argument("--lon", type=float, required=True)
    event.add_argument("--ref-time", required=True, help="ISO timestamp, e.g. 2025-01-07T18:30:00Z.")
    event.add_argument("--bucket", default=default_runtime_bucket_uri())
    event.add_argument("--out-dir", type=Path, default=None,
                       help="Defaults to $IGNIS_CACHE_ROOT/{profile}/, or "
                            ".cache/runtime_cache/{profile}/ when unset.")
    event.add_argument("--work-dir", type=Path, default=_cache_root() / "work")
    event.add_argument("--no-upload", action="store_true")
    event.add_argument("--skip-firms", action="store_true")
    event.add_argument("--skip-noaa", action="store_true")
    event.add_argument("--overwrite", action="store_true")
    event.add_argument("--map-key", default=None)
    event.add_argument("--t-seq", type=int, default=DEFAULT_TSEQ, help=f"History length used by tilesvc (default {DEFAULT_TSEQ}).")
    event.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Forecast horizon (default {DEFAULT_STEPS}).")
    event.add_argument("--step-hours", type=int, default=DEFAULT_STEP_HOURS)
    event.add_argument("--source-priority", default=default_priority)

    sync = sub.add_parser("sync", help="Sync runtime cache files from S3 into local /data directories.")
    sync.add_argument("--bucket", default=default_runtime_bucket_uri())
    sync.add_argument("--profile", default=os.getenv("IGNIS_RUNTIME_CACHE_PROFILE", DEFAULT_PROFILE))
    sync.add_argument("--firms-dir", type=Path, default=_path_env("FIRMS_SNAPSHOT_DIR", "/data/firms_snapshots"))
    sync.add_argument("--noaa-dir", type=Path, default=_path_env("NOAA_GRID_CACHE_DIR", "/data/noaa_grid_cache"))
    sync.add_argument("--required", action="store_true", default=os.getenv("IGNIS_RUNTIME_CACHE_REQUIRED", "0").lower() in {"1", "true", "yes", "on"})

    dirs = sub.add_parser("ensure-dirs", help="Create runtime cache directories without syncing S3.")
    dirs.add_argument("--firms-dir", type=Path, default=_path_env("FIRMS_SNAPSHOT_DIR", "/data/firms_snapshots"))
    dirs.add_argument("--noaa-dir", type=Path, default=_path_env("NOAA_GRID_CACHE_DIR", "/data/noaa_grid_cache"))

    validate = sub.add_parser("validate", help="Validate local runtime cache directories.")
    validate.add_argument("--firms-dir", type=Path, default=_path_env("FIRMS_SNAPSHOT_DIR", "data/firms_snapshots"))
    validate.add_argument("--noaa-dir", type=Path, default=_path_env("NOAA_GRID_CACHE_DIR", "data/noaa_grid_cache"))
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "build-palisades":
        result = build_palisades_runtime_cache(
            bucket_uri=args.bucket,
            profile=args.profile,
            lat=args.lat,
            lon=args.lon,
            ref_time=args.ref_time,
            out_dir=args.out_dir,
            work_dir=args.work_dir,
            upload=not args.no_upload,
            build_firms=not args.skip_firms,
            build_noaa=not args.skip_noaa,
            overwrite=args.overwrite,
            map_key=args.map_key,
            source_priority=args.source_priority,
        )
        print(summarize_result(result))
        return
    if args.command == "build-event":
        # Resolve the default here rather than leaving it None. The callee
        # falls back to a hard-coded .cache/runtime_cache/{profile}, which
        # ignores IGNIS_CACHE_ROOT and quietly writes to the boot drive even
        # when the cache root points at an external volume.
        out_dir = args.out_dir or (_cache_root() / args.profile)
        result = build_event_runtime_cache(
            profile=args.profile,
            lat=args.lat,
            lon=args.lon,
            ref_time=args.ref_time,
            bucket_uri=args.bucket,
            out_dir=out_dir,
            work_dir=args.work_dir,
            upload=not args.no_upload,
            build_firms=not args.skip_firms,
            build_noaa=not args.skip_noaa,
            overwrite=args.overwrite,
            map_key=args.map_key,
            t_seq=args.t_seq,
            steps=args.steps,
            step_hours=args.step_hours,
            source_priority=args.source_priority,
        )
        print(summarize_result(result))
        return
    if args.command == "sync":
        result = sync_runtime_cache(
            bucket_uri=args.bucket,
            profile=args.profile,
            firms_dir=args.firms_dir,
            noaa_dir=args.noaa_dir,
            required=bool(args.required),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "ensure-dirs":
        print(json.dumps(ensure_runtime_dirs(firms_dir=args.firms_dir, noaa_dir=args.noaa_dir), indent=2, sort_keys=True))
        return
    if args.command == "validate":
        print(json.dumps(validate_runtime_dirs(firms_dir=args.firms_dir, noaa_dir=args.noaa_dir), indent=2, sort_keys=True))
        return
    parser.error(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()

