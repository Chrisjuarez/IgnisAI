"""Tests for tools/audit_alignment.py.

We deliberately avoid hitting the network in these tests (no FIRMS, no
Open-Meteo, no NOAA grib downloads). The targets here are the pure
helpers and the canonical-grid + FIRMS-rasterization checks, which run
entirely against in-memory numpy / pyproj.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from tools import audit_alignment as audit


@pytest.fixture(autouse=True)
def _clear_grid_caches(monkeypatch):
    # Each test gets a fresh env so the dynamic_builder caches don't
    # leak NOAA paths across tests.
    monkeypatch.delenv("NOAA_GRID_CACHE_DIR", raising=False)
    monkeypatch.delenv("NOAA_GRID_CACHE_TEMPLATE", raising=False)
    monkeypatch.delenv("STATIC_CATALOG_PATH", raising=False)


def test_resolve_target_uses_preset_when_provided():
    args = audit._parse_args(["--preset", "palisades", "--json"])
    lat, lon, ref_time = audit._resolve_target(args)

    assert pytest.approx(lat, abs=1e-3) == 34.078
    assert pytest.approx(lon, abs=1e-3) == -118.555
    assert ref_time is not None
    assert ref_time.tzinfo is dt.timezone.utc
    assert ref_time.year == 2025 and ref_time.month == 1 and ref_time.day == 7


def test_resolve_target_accepts_explicit_lat_lon_and_iso_time():
    args = audit._parse_args(
        ["--lat", "37.7", "--lon", "-122.4", "--ref-time", "2024-09-01T12:00:00Z", "--json"]
    )
    lat, lon, ref_time = audit._resolve_target(args)

    assert lat == 37.7 and lon == -122.4
    assert ref_time == dt.datetime(2024, 9, 1, 12, tzinfo=dt.timezone.utc)


def test_check_canonical_tile_matches_grid_module_geometry():
    from services.tilesvc.grid import PIX, SIZE

    check, canon = audit._check_canonical_tile(34.078, -118.555)

    assert check.status == "pass"
    assert canon["pixel_size_m"] == float(PIX)
    assert canon["shape"] == [SIZE, SIZE]
    # Canonical affine encodes (PIX, 0, x0, 0, -PIX, y0); pixel size is
    # the determinant element we can check without recomputing the tile.
    assert canon["affine"][0] == float(PIX)
    assert canon["affine"][4] == -float(PIX)


def test_firms_rasterization_lands_on_canonical_tile_grid():
    _, canon = audit._check_canonical_tile(34.078, -118.555)

    check = audit._check_firms_rasterization(34.078, -118.555, canon)

    assert check.status == "pass"
    assert check.details["shape"] == canon["shape"]
    assert check.details["affine"] == canon["affine"]
    assert check.details["ignition_pixels"] > 0


def test_static_catalog_check_skips_when_env_unset():
    _, canon = audit._check_canonical_tile(34.078, -118.555)

    checks = audit._check_static_catalog(canon, max_residual_m=1.0)

    assert len(checks) == 1
    assert checks[0].status == "skip"
    assert "STATIC_CATALOG_PATH" in checks[0].message


def test_noaa_cache_check_skips_when_no_cache_configured():
    _, canon = audit._check_canonical_tile(34.078, -118.555)

    check = audit._check_noaa_cache(34.078, -118.555, ref_time=None, canon=canon)

    assert check.status == "skip"


def test_noaa_cache_check_passes_when_npz_matches_canonical_shape(tmp_path, monkeypatch):
    from services.tilesvc.grid import SIZE

    monkeypatch.setenv("NOAA_GRID_CACHE_DIR", str(tmp_path))
    ref_time = dt.datetime(2025, 1, 7, 18, tzinfo=dt.timezone.utc)
    arrays = {
        name: np.full((SIZE, SIZE), 1.0, dtype=np.float32)
        for name in ("u", "v", "gust", "tempC", "q", "precip")
    }
    arrays["source"] = np.array("noaa_hrrr")
    np.savez_compressed(tmp_path / "20250107T18_34.08_-118.56.npz", **arrays)
    _, canon = audit._check_canonical_tile(34.078, -118.555)

    check = audit._check_noaa_cache(34.078, -118.555, ref_time=ref_time, canon=canon)

    assert check.status == "pass"
    assert check.details["source"] == "noaa_hrrr"


def test_run_audit_returns_pass_verdict_on_clean_repo(monkeypatch):
    """End-to-end: with no static catalog or NOAA cache the audit still runs."""
    args = audit._parse_args(["--preset", "palisades", "--json"])
    summary = audit.run_audit(args)

    assert summary["verdict"] == "pass"
    # Canonical tile + FIRMS rasterization should both pass; static and
    # NOAA cache should be skipped (no env configured).
    statuses = {c["name"]: c["status"] for c in summary["checks"]}
    assert statuses["canonical_tile"] == "pass"
    assert statuses["firms_rasterization"] == "pass"
    assert statuses["static_catalog"] == "skip"
    assert statuses["noaa_cache"] == "skip"
