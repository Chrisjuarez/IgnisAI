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


def test_hrrr_index_bands_pin_the_altitude():
    """UGRD appears at several altitudes in the HRRR surface file.

    Matching the variable alone lands on 80 m or a diagnostic max-wind field
    and produces jet-stream values over a surface fire, which is a documented
    past bug in this pipeline. The index lookup must pin the level.
    """
    from services.runtime_cache.pipeline import HRRR_IDX_BANDS

    levels = dict(HRRR_IDX_BANDS)
    assert levels["UGRD"] == "10 m above ground"
    assert levels["VGRD"] == "10 m above ground"
    assert levels["TMP"] == "2 m above ground"


def test_index_parse_yields_ranges_and_last_record_is_open_ended(monkeypatch):
    """A record's length is the next record's offset; the last runs to EOF."""
    import services.runtime_cache.pipeline as pipeline

    class FakeResponse:
        text = (
            "1:0:d=2026:REFC:entire atmosphere:anl:\n"
            "9:2705323:d=2026:GUST:surface:anl:\n"
            "77:43020972:d=2026:UGRD:10 m above ground:anl:\n"
            "78:45164444:d=2026:VGRD:10 m above ground:anl:\n"
        )
        def raise_for_status(self): pass

    class FakeSession:
        def get(self, url, **kw): return FakeResponse()

    ranges = pipeline._hrrr_index("https://example/x.grib2", FakeSession())

    assert ranges == [(2705323, 43020971), (43020972, 45164443), (45164444, None)]


def test_missing_index_falls_back_to_the_whole_file():
    """A slow correct answer beats a fast missing one."""
    import services.runtime_cache.pipeline as pipeline

    class Failing:
        def get(self, url, **kw): raise OSError("no index")

    assert pipeline._hrrr_index("https://example/x.grib2", Failing()) is None
