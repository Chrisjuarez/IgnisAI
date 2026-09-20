"""Durable record of checkpoint validation runs.

Validation runs as a one-off job inside this service, because that is the only
place holding credentials for the static rasters. Render does not expose job
stdout through its API, so a run that printed its answer and exited left no
retrievable trace - including when it failed, which is exactly when the output
matters most.

Runs therefore write a JSON report to the mounted disk, and the service serves
them back. Successes and failures both, so a broken run is diagnosable without
dashboard access.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Lives on the mounted disk so reports outlive the container that wrote them.
DEFAULT_REPORT_DIR = os.getenv("VALIDATION_REPORT_DIR", "/data/validation")

#: Reports are small, but a job loop should not be able to fill the disk.
MAX_REPORTS = int(os.getenv("VALIDATION_REPORT_MAX", "50"))

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    return _UNSAFE.sub("-", str(value or "run")).strip("-") or "run"


def report_dir(directory: str | Path | None = None) -> Path:
    return Path(directory or DEFAULT_REPORT_DIR)


def write_report(payload: Dict[str, Any], *, directory: str | Path | None = None) -> Optional[Path]:
    """Persist one run. Never raises: a reporting failure must not fail the run."""
    try:
        target = report_dir(directory)
        target.mkdir(parents=True, exist_ok=True)

        recorded_at = payload.get("recorded_at") or dt.datetime.now(dt.timezone.utc).isoformat()
        payload = {**payload, "recorded_at": recorded_at}
        stamp = _slug(recorded_at.replace(":", "").replace("-", ""))
        name = f"{stamp}-{_slug(payload.get('event'))}-{_slug(payload.get('checkpoint'))}.json"

        path = target / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _prune(target)
        return path
    except OSError as exc:
        print(f"⚠️  Could not write validation report: {exc}")
        return None


def _prune(target: Path) -> None:
    reports = sorted(target.glob("*.json"))
    for stale in reports[:-MAX_REPORTS] if len(reports) > MAX_REPORTS else []:
        try:
            stale.unlink()
        except OSError:
            pass


def list_reports(directory: str | Path | None = None, *, limit: int = 20) -> List[Dict[str, Any]]:
    """Recorded runs, newest first. Unreadable files are surfaced, not hidden."""
    target = report_dir(directory)
    if not target.is_dir():
        return []

    reports: List[Dict[str, Any]] = []
    for path in sorted(target.glob("*.json"), reverse=True)[:max(0, limit)]:
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            reports.append({"file": path.name, "status": "unreadable", "error": str(exc)})
    return reports
