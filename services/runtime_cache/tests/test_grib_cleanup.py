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
