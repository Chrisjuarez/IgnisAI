import datetime as dt
import os

import numpy as np
import pytest

from services.tilesvc.hrrr_grids import (
    HRRR_BANDS,
    cycle_for_valid_time,
    fetch_hrrr_tile_grids,
    hrrr_sfc_url,
    parse_grib_index,
    select_bands,
)

UTC = dt.timezone.utc

# Trimmed from a real hrrr.t18z.wrfsfcf01.grib2.idx.
SAMPLE_IDX = """1:0:d=2025010718:REFC:entire atmosphere:1 hour fcst:
2:912345:d=2025010718:GUST:surface:1 hour fcst:
3:1500000:d=2025010718:TMP:2 m above ground:1 hour fcst:
4:2100000:d=2025010718:SPFH:2 m above ground:1 hour fcst:
5:2600000:d=2025010718:UGRD:10 m above ground:1 hour fcst:
6:3200000:d=2025010718:VGRD:10 m above ground:1 hour fcst:
7:3800000:d=2025010718:APCP:surface:0-1 hour acc fcst:
"""


def test_index_records_span_to_the_next_offset():
    records = parse_grib_index(SAMPLE_IDX)
    assert len(records) == 7
    assert records[0].start == 0 and records[0].end == 912345
    assert records[0].byte_range == "bytes=0-912344"
    # The final record has no successor, so it runs to end of file.
    assert records[-1].end is None
    assert records[-1].byte_range == "bytes=3800000-"


def test_select_bands_finds_every_channel():
    bands = select_bands(parse_grib_index(SAMPLE_IDX))
    assert set(bands) == set(HRRR_BANDS)
    assert bands["u"].start == 2600000
    assert bands["v"].start == 3200000


def test_select_bands_ignores_a_matching_variable_at_the_wrong_level():
    idx = "1:0:d=2025010718:UGRD:80 m above ground:1 hour fcst:\n"
    assert select_bands(parse_grib_index(idx)) == {}


def test_malformed_index_lines_are_skipped():
    idx = "not an index line\n:::\n1:0:d=2025010718:GUST:surface:anl:\n"
    records = parse_grib_index(idx)
    assert [r.variable for r in records] == ["GUST"]


def test_past_hours_use_a_one_hour_lead_so_precip_exists():
    valid = dt.datetime(2025, 1, 7, 18, tzinfo=UTC)
    now = dt.datetime(2026, 9, 5, 0, tzinfo=UTC)
    cycle, forecast_hour = cycle_for_valid_time(valid, now=now)
    assert cycle == dt.datetime(2025, 1, 7, 17, tzinfo=UTC)
    assert forecast_hour == 1


def test_future_hours_forecast_from_the_newest_published_cycle():
    now = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 5, 16, tzinfo=UTC)
    cycle, forecast_hour = cycle_for_valid_time(valid, now=now)
    # 12:30 minus the 2h publication lag leaves the 10z cycle as the newest.
    assert cycle == dt.datetime(2026, 9, 5, 10, tzinfo=UTC)
    assert forecast_hour == 6


def test_forecast_hour_is_capped_at_the_end_of_the_hrrr_run():
    now = dt.datetime(2026, 9, 5, 12, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 8, 12, tzinfo=UTC)
    _, forecast_hour = cycle_for_valid_time(valid, now=now)
    assert forecast_hour == 18


def test_url_matches_the_public_archive_layout():
    url = hrrr_sfc_url(dt.datetime(2025, 1, 7, 17, tzinfo=UTC), 1)
    assert url.endswith("/hrrr.20250107/conus/hrrr.t17z.wrfsfcf01.grib2")


def test_missing_index_returns_none_rather_than_raising():
    class MissingIndex:
        def get(self, url, **kwargs):
            return type("Response", (), {"status_code": 404, "text": ""})()

    result = fetch_hrrr_tile_grids(
        34.078, -118.555, dt.datetime(2025, 1, 7, 18, tzinfo=UTC), session=MissingIndex()
    )
    assert result is None


@pytest.mark.skipif(
    os.getenv("IGNIS_NETWORK_TESTS", "0") not in {"1", "true", "yes", "on"},
    reason="hits the public HRRR archive; set IGNIS_NETWORK_TESTS=1 to run",
)
def test_live_fetch_returns_a_spatially_varying_wind_field():
    """
    The whole point of HRRR is spatial structure. A field that comes back
    constant means we have bought nothing over the point fallback.
    """
    grids = fetch_hrrr_tile_grids(34.078, -118.555, dt.datetime(2025, 1, 7, 18, tzinfo=UTC))
    assert grids is not None

    for name in ("u", "v", "gust", "tempC"):
        assert grids[name].shape == (64, 64)
        assert np.isfinite(grids[name]).all()

    speed = np.hypot(grids["u"], grids["v"])
    assert speed.std() > 0.1, f"wind field is effectively constant (std={speed.std():.4f})"

    # Physical ranges, per channel. Records are written in source-file order but
    # HRRR_BANDS is in a different order, so reading them back positionally
    # silently swaps channels — these bounds are what catches that.
    assert -60.0 < grids["tempC"].mean() < 60.0, "tempC out of range; bands may be swapped"
    assert 0.0 < grids["q"].mean() < 0.05, "specific humidity out of range; bands may be swapped"
    assert grids["gust"].max() >= speed.max(), "gust below sustained wind; bands may be swapped"

    # The Palisades ignition hour was a Santa Ana: strong, blowing offshore to
    # the southwest. Getting this wrong is the failure mode that matters.
    toward = np.degrees(np.arctan2(grids["u"].mean(), grids["v"].mean())) % 360
    assert 180.0 < toward < 280.0, f"wind should blow offshore, got toward {toward:.0f} deg"
    assert grids["gust"].max() > 20.0, "Santa Ana gusts should exceed 20 m/s"
