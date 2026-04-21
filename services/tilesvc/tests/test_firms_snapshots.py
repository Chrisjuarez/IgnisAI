import datetime as dt

import pytest

from services.tilesvc import dynamic_builder
from services.tilesvc.static_catalog import InputUnavailable


def test_firms_snapshots_compose_history_from_daily_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRMS_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("FIRMS_SNAPSHOT_REQUIRED", "1")
    (tmp_path / "2026-04-19.csv").write_text(
        "latitude,longitude,acq_date,acq_time,frp\n34.5,-118.5,2026-04-19,1200,12\n",
        encoding="utf-8",
    )
    (tmp_path / "2026-04-20.csv").write_text(
        "latitude,longitude,acq_date,acq_time,frp\n34.6,-118.6,2026-04-20,0900,15\n",
        encoding="utf-8",
    )

    points = dynamic_builder._load_firms_points_from_snapshots(
        (-119.0, 34.0, -118.0, 35.0),
        dt.datetime(2026, 4, 19, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 4, 20, 12, tzinfo=dt.timezone.utc),
    )

    assert len(points) == 2
    assert {point["frp"] for point in points} == {12.0, 15.0}


def test_firms_snapshot_required_fails_when_daily_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRMS_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("FIRMS_SNAPSHOT_REQUIRED", "1")

    with pytest.raises(InputUnavailable) as exc:
        dynamic_builder._load_firms_points_from_snapshots(
            (-119.0, 34.0, -118.0, 35.0),
            dt.datetime(2026, 4, 19, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 4, 20, tzinfo=dt.timezone.utc),
        )

    assert exc.value.reason == "firms_snapshot_missing"
    assert exc.value.details["missing_dates"] == ["2026-04-19", "2026-04-20"]
