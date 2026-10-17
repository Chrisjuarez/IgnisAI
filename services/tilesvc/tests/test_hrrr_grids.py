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

# Trimmed from a real hrrr.t06z.wrfsfcf30.grib2.idx (2026-10-16). Past f01 the
# file carries APCP twice: the run total first, then the last hour.
SAMPLE_F30_IDX = """1:0:d=2026101606:REFC:entire atmosphere:30 hour fcst:
9:4365156:d=2026101606:GUST:surface:30 hour fcst:
10:5740639:d=2026101606:UGRD:250 mb:30 hour fcst:
71:48032204:d=2026101606:TMP:2 m above ground:30 hour fcst:
72:49293159:d=2026101606:POT:2 m above ground:30 hour fcst:
73:50508588:d=2026101606:SPFH:2 m above ground:30 hour fcst:
74:52011537:d=2026101606:DPT:2 m above ground:30 hour fcst:
77:55746312:d=2026101606:UGRD:10 m above ground:30 hour fcst:
78:58127927:d=2026101606:VGRD:10 m above ground:30 hour fcst:
79:60509542:d=2026101606:WIND:10 m above ground:29-30 hour max fcst:
84:65062122:d=2026101606:APCP:surface:0-30 hour acc fcst:
85:66286701:d=2026101606:WEASD:surface:0-30 hour acc fcst:
89:66798944:d=2026101606:BGRUN:surface:29-30 hour acc fcst:
90:66799312:d=2026101606:APCP:surface:29-30 hour acc fcst:
91:67091994:d=2026101606:WEASD:surface:29-30 hour acc fcst:
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


def test_forecast_hour_precip_is_the_last_hour_not_the_run_total():
    # Training frames and past hours read f01, a one-hour total. A forecast
    # hour has to feed the model the same quantity, not 0-N hours of rain.
    bands = select_bands(parse_grib_index(SAMPLE_F30_IDX))
    assert bands["precip"].start == 66799312
    assert bands["precip"].byte_range == "bytes=66799312-67091993"
    assert bands["precip"].forecast == "29-30 hour acc fcst"


def test_f01_precip_is_still_its_only_one_hour_record():
    bands = select_bands(parse_grib_index(SAMPLE_IDX))
    assert bands["precip"].start == 3800000
    assert bands["precip"].end is None


def test_preferring_hourly_precip_leaves_other_channels_alone():
    bands = select_bands(parse_grib_index(SAMPLE_F30_IDX))
    assert set(bands) == set(HRRR_BANDS)
    assert {name: record.start for name, record in bands.items() if name != "precip"} == {
        "gust": 4365156,
        "tempC": 48032204,
        "q": 50508588,
        "u": 55746312,
        "v": 58127927,
    }


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


def test_every_chosen_file_is_valid_at_the_requested_hour():
    # A forecast day asks for weather at its own hour. A file valid at another
    # hour hands the model the wrong weather under the right label.
    now = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    base = dt.datetime(2026, 9, 5, 12, tzinfo=UTC)
    for hours_ahead in range(-72, 73):
        valid = base + dt.timedelta(hours=hours_ahead)
        run = cycle_for_valid_time(valid, now=now)
        if run is None:
            continue
        cycle, forecast_hour = run
        assert cycle + dt.timedelta(hours=forecast_hour) == valid, (valid, run)


def test_hours_past_the_hourly_run_come_from_the_newest_48_hour_cycle():
    # Newest cycle is 10z, whose run stops at 18 h. The 06z run goes to 48 h.
    now = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 6, 12, tzinfo=UTC)
    assert cycle_for_valid_time(valid, now=now) == (dt.datetime(2026, 9, 5, 6, tzinfo=UTC), 30)


def test_the_hourly_run_is_used_up_to_its_last_hour():
    now = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 6, 4, tzinfo=UTC)
    assert cycle_for_valid_time(valid, now=now) == (dt.datetime(2026, 9, 5, 10, tzinfo=UTC), 18)


def test_the_48_hour_cycle_can_be_the_previous_day():
    # Just after midnight the newest 48-hour run is yesterday's 18z.
    now = dt.datetime(2026, 9, 5, 1, 30, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 6, 0, tzinfo=UTC)
    assert cycle_for_valid_time(valid, now=now) == (dt.datetime(2026, 9, 4, 18, tzinfo=UTC), 30)


def test_a_48_hour_cycle_reaches_its_last_hour():
    now = dt.datetime(2026, 9, 5, 14, 10, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 7, 12, tzinfo=UTC)
    assert cycle_for_valid_time(valid, now=now) == (dt.datetime(2026, 9, 5, 12, tzinfo=UTC), 48)


def test_no_file_for_an_hour_no_published_run_reaches():
    now = dt.datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    valid = dt.datetime(2026, 9, 7, 7, tzinfo=UTC)  # 49 h after the 06z cycle
    assert cycle_for_valid_time(valid, now=now) is None


def test_url_matches_the_public_archive_layout():
    url = hrrr_sfc_url(dt.datetime(2025, 1, 7, 17, tzinfo=UTC), 1)
    assert url.endswith("/hrrr.20250107/conus/hrrr.t17z.wrfsfcf01.grib2")


def test_missing_index_returns_none_rather_than_raising():
    class MissingIndex:
        def get(self, url, **kwargs):
            return type("Response", (), {"status_code": 404, "text": ""})()

    result = fetch_hrrr_tile_grids(
        34.078, -118.555, dt.datetime(2025, 1, 7, 17, tzinfo=UTC), 1, session=MissingIndex()
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
    grids = fetch_hrrr_tile_grids(34.078, -118.555, dt.datetime(2025, 1, 7, 17, tzinfo=UTC), 1)
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
