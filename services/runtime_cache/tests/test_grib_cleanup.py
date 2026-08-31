"""Downloaded gribs must not survive extraction.

A grib is a pure intermediate: the npz written from it is four orders of
magnitude smaller and is the only thing anything downstream reads. Left behind
they accumulate at roughly 130 MB per hour per fire, and a 197-fire backfill
filled a 460 GB disk before failing with ENOSPC.
"""
import os

import pytest

from services.runtime_cache.pipeline import _discard_intermediate


def test_intermediate_is_removed(tmp_path):
    grib = tmp_path / "20260801T18_hrrr_sfc.grib2"
    grib.write_bytes(b"x" * 1024)

    _discard_intermediate(grib)

    assert not grib.exists()


def test_removal_is_idempotent(tmp_path):
    missing = tmp_path / "never_downloaded.grib2"

    _discard_intermediate(missing)   # must not raise


def test_keeping_intermediates_is_opt_in(tmp_path, monkeypatch):
    # Debugging a bad extraction needs the grib; consuming a disk by default
    # does not.
    monkeypatch.setenv("KEEP_GRIB_INTERMEDIATES", "1")
    grib = tmp_path / "keep_me.grib2"
    grib.write_bytes(b"x")

    _discard_intermediate(grib)

    assert grib.exists()


def test_unreadable_path_does_not_break_the_build(tmp_path):
    # Cleanup runs in a finally; failing there would mask the real error.
    _discard_intermediate(tmp_path)   # a directory, not a file - must not raise


def test_event_out_dir_follows_the_cache_root(monkeypatch, tmp_path):
    """build-event must write where IGNIS_CACHE_ROOT points.

    It defaulted --out-dir to None and the callee substituted a hard-coded
    .cache/runtime_cache/{profile}. With the cache root set to an external
    volume the builds still landed on the boot drive, reported success, and
    refilled the disk that had just been cleared.
    """
    import importlib

    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path / "external"))
    module = importlib.reload(importlib.import_module("services.runtime_cache.__main__"))

    assert module._cache_root() == tmp_path / "external"

    parser = module._build_parser()
    args = parser.parse_args(["build-event", "--profile", "somefire",
                              "--lat", "40", "--lon", "-120",
                              "--ref-time", "2026-07-01T00:00:00Z"])
    resolved = args.out_dir or (module._cache_root() / args.profile)

    assert resolved == tmp_path / "external" / "somefire"
    assert args.work_dir == tmp_path / "external" / "work"
