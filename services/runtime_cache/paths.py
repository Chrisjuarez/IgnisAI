"""Where runtime caches live.

One definition, because there were six. Every consumer had its own hard-coded
".cache/runtime_cache", so setting IGNIS_CACHE_ROOT to an external volume moved
the writer and left every reader looking at the old path - the builds landed on
the external drive and the validators reported "inputs unavailable" for all of
them.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "IGNIS_CACHE_ROOT"
DEFAULT = Path(".cache/runtime_cache")


def cache_root() -> Path:
    """Root for runtime caches and grib intermediates."""
    return Path(os.getenv(ENV_VAR, str(DEFAULT)))


def profile_dir(profile: str) -> Path:
    return cache_root() / profile


def firms_dir(profile: str) -> Path:
    return profile_dir(profile) / "firms_snapshots"


def noaa_dir(profile: str) -> Path:
    return profile_dir(profile) / "noaa_grid_cache"


def use_profile(profile: str) -> bool:
    """Point the tilesvc builders at one event's cache. False if absent."""
    firms, noaa = firms_dir(profile), noaa_dir(profile)
    if not firms.is_dir() or not noaa.is_dir():
        return False
    os.environ["FIRMS_SNAPSHOT_DIR"] = str(firms)
    os.environ["FIRMS_SNAPSHOT_REQUIRED"] = "1"
    os.environ["NOAA_GRID_CACHE_DIR"] = str(noaa)
    os.environ.setdefault("NOAA_GRIB_ENABLED", "1")
    return True
