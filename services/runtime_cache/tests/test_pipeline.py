import datetime as dt

import numpy as np

from services.runtime_cache import pipeline
from services.tilesvc.grid import SIZE


def test_palisades_runtime_cache_hours_match_tilesvc_access_pattern():
    ref = pipeline.parse_ref_time("2025-01-07T18:30:00Z")

    hours = pipeline.cache_hours_for_multistep(ref, t_seq=6, steps=6, step_hours=24)

    assert len(hours) == 11
    assert hours[0] == dt.datetime(2025, 1, 2, 18, tzinfo=dt.timezone.utc)
    assert hours[-1] == dt.datetime(2025, 1, 12, 18, tzinfo=dt.timezone.utc)
    assert pipeline.noaa_cache_filename(hours[0], 34.05, -118.55) == "20250102T18_34.05_-118.55.npz"


def test_palisades_firms_snapshot_dates_include_full_history_window():
    ref = pipeline.parse_ref_time("2025-01-07T18:30:00Z")

    dates = pipeline.firms_snapshot_dates(ref, t_seq=6, step_hours=24)

    assert dates[0] == dt.date(2025, 1, 1)
    assert dates[-1] == dt.date(2025, 1, 7)
    assert len(dates) == 7


def test_write_and_validate_noaa_npz_schema(tmp_path):
    arrays = {
        name: np.full((SIZE, SIZE), idx, dtype=np.float32)
        for idx, name in enumerate(pipeline.REQUIRED_NOAA_CHANNELS)
    }
    path = tmp_path / "20250102T18_34.05_-118.55.npz"

    pipeline.write_noaa_npz(path, arrays)
    stats = pipeline.validate_noaa_npz(path)

    assert stats["keys"] == sorted(pipeline.REQUIRED_NOAA_CHANNELS)
    assert stats["channels"]["u"]["mean"] == 0.0
    assert stats["channels"]["precip"]["mean"] == 5.0


def test_parse_gfs_index_selects_required_records():
    text = "\n".join(
        [
            "1:0:d=2025010218:UGRD:10 m above ground:anl:",
            "2:100:d=2025010218:VGRD:10 m above ground:anl:",
            "3:200:d=2025010218:GUST:surface:anl:",
            "4:300:d=2025010218:TMP:2 m above ground:anl:",
            "5:400:d=2025010218:SPFH:2 m above ground:anl:",
            "6:500:d=2025010218:APCP:surface:anl:",
            "7:600:d=2025010218:OTHER:surface:anl:",
        ]
    )

    records = pipeline.parse_gfs_index(text)
    selected = pipeline._select_gfs_records(records)

    assert selected["u"].start == 0
    assert selected["u"].end == 100
    assert selected["precip"].variable == "APCP"


def test_s3_runtime_key_layout():
    assert (
        pipeline.s3_join(
            "s3://ignisai-static-chrisjuarez-2026/runtime",
            pipeline.runtime_s3_key("firms_snapshots", "palisades", "2025-01-01.csv"),
        )
        == "s3://ignisai-static-chrisjuarez-2026/runtime/firms_snapshots/palisades/2025-01-01.csv"
    )

