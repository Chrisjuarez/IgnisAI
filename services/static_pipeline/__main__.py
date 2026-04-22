from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import BuildOptions, build_static_pipeline, default_bucket_uri, default_version


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m services.static_pipeline",
        description="Build IgnisAI static COGs and static_catalog.production.json.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build static COGs, upload to S3, and write a catalog.")
    build.add_argument("--extent", default="western_conus", choices=["western_conus", "california_mvp"])
    build.add_argument("--version", default=default_version(), help="Version label, usually YYYYMMDD.")
    build.add_argument(
        "--bucket",
        default=default_bucket_uri(),
        help="S3 prefix, e.g. s3://my-bucket/ignis/static. Defaults to IGNIS_STATIC_BUCKET/PREFIX.",
    )
    build.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="JSON source config. Defaults to config/static_sources.<extent>.json.",
    )
    build.add_argument("--work-dir", type=Path, default=Path(".cache/static_pipeline"))
    build.add_argument(
        "--catalog-out",
        type=Path,
        default=None,
        help="Where to write the runtime catalog locally. Use config/static_catalog.production.json after upload.",
    )
    build.add_argument("--no-upload", action="store_true", help="Build local COGs and catalog only.")
    build.add_argument("--dry-run", action="store_true", help="Validate config and print planned outputs; no downloads/uploads.")
    build.add_argument(
        "--catalog-uri-mode",
        choices=["s3", "local"],
        default="s3",
        help="Use S3 URIs in the catalog for production or local paths for tests/dev.",
    )
    build.add_argument(
        "--allow-template-sources",
        action="store_true",
        help="Allow REPLACE_* source URIs. Useful only for dry-run/template validation.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "build":
        result = build_static_pipeline(
            BuildOptions(
                extent=args.extent,
                version=args.version,
                bucket_uri=args.bucket,
                source_config=args.source_config,
                work_dir=args.work_dir,
                catalog_out=args.catalog_out,
                upload=not args.no_upload,
                dry_run=args.dry_run,
                catalog_uri_mode=args.catalog_uri_mode,
                allow_template_sources=args.allow_template_sources,
            )
        )
        print(json.dumps({k: v for k, v in result.items() if k != "catalog"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
