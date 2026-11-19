"""The FIRMS history store: one western-US day of detections per file.

A dated forecast needs the detections of its day, long after the live FIRMS
API has stopped serving them. Each day is fetched once and kept; the copy the
app saw live (near-real-time) is kept apart from the reprocessed archive.
"""
import datetime as dt

import numpy as np
import pytest

from services.tilesvc import dynamic_builder, firms_history
from services.tilesvc.grid import lonlat_to_tile, tile_affine, tile_bounds_lonlat, lonlat_to_xy_m

UTC = dt.timezone.utc
NRT_FROM = dt.date(2026, 7, 1)   # FIRMS keeps near-real-time products from here on
NOW = dt.datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
LON, LAT = -118.40183, 34.561835


def row(lat, lon, day, hhmm, frp="12.5"):
    return {"latitude": f"{lat}", "longitude": f"{lon}", "acq_date": day.isoformat(), "acq_time": hhmm, "frp": frp}


class Store(dict):
    """In-memory stand-in for the S3 prefix."""

    def get_text(self, name):
        return self.get(name)

    def put_text(self, name, text):
        self[name] = text


class Firms:
    """Records each product queried; answers with the rows given per product."""

    def __init__(self, rows=None):
        self.rows, self.calls = rows or {}, []

    def __call__(self, product, bbox, day):
        self.calls.append((product, day))
        return list(self.rows.get(product, []))


def day_text(day, store, firms, now=NOW):
    return firms_history.day_csv(day, store=store, fetch_rows=firms, nrt_from=NRT_FROM, now=now)


def test_a_missing_finished_day_is_fetched_once_and_kept_as_the_live_copy():
    day = dt.date(2026, 10, 16)
    firms = Firms({"VIIRS_NOAA20_NRT": [row(34.56, -118.40, day, "2033")]})
    store = Store()

    first = day_text(day, store, firms)
    calls = len(firms.calls)
    second = day_text(day, store, firms)

    assert list(store) == ["2026-10-16.nrt.csv"]
    assert first == second and "34.56,-118.4,2026-10-16,2033,12.5" in first
    assert len(firms.calls) == calls, "a stored day is never fetched again"


def test_a_day_before_the_live_window_comes_from_the_archive():
    day = dt.date(2025, 1, 8)
    firms = Firms({"VIIRS_NOAA20_SP": [row(34.07, -118.54, day, "1015")]})
    store = Store()

    day_text(day, store, firms)

    assert list(store) == ["2025-01-08.sp.csv"]
    assert all(product.endswith("_SP") for product, _ in firms.calls)


def test_the_live_copy_is_read_before_the_archive_copy():
    day = dt.date(2026, 7, 2)
    store = Store({"2026-07-02.sp.csv": "latitude,longitude,acq_date,acq_time,frp\n1,1,2026-07-02,0100,1\n",
                   "2026-07-02.nrt.csv": "latitude,longitude,acq_date,acq_time,frp\n2,2,2026-07-02,0200,2\n"})

    assert "2,2,2026-07-02,0200,2" in day_text(day, store, Firms())


def test_a_day_that_is_not_over_is_served_but_not_kept():
    today = NOW.date()
    firms = Firms({"MODIS_NRT": [row(34.56, -118.40, today, "0130")]})
    store = Store()

    text = day_text(today, store, firms)

    assert "2026-10-18,0130" in text
    assert store == {}, "later passes and late arrivals would be missing from a stored partial day"


def test_products_are_merged_without_duplicates():
    day = dt.date(2026, 10, 16)
    same = row(34.56, -118.40, day, "2033")
    firms = Firms({"VIIRS_NOAA20_NRT": [same], "VIIRS_NOAA21_NRT": [same, row(34.57, -118.41, day, "2120")]})

    lines = day_text(day, Store(), firms).strip().splitlines()

    assert lines[0] == "latitude,longitude,acq_date,acq_time,frp"
    assert len(lines) == 3


def test_the_daily_job_keeps_yesterdays_live_copy_and_is_safe_to_rerun():
    firms = Firms({"VIIRS_NOAA20_NRT": [row(34.56, -118.40, dt.date(2026, 10, 17), "2033")]})
    store = Store()

    first = firms_history.keep_yesterday(store=store, fetch_rows=firms, nrt_from=NRT_FROM, now=NOW)
    again = firms_history.keep_yesterday(store=store, fetch_rows=firms, nrt_from=NRT_FROM, now=NOW)

    assert first == again == "2026-10-17.nrt.csv"
    assert list(store) == ["2026-10-17.nrt.csv"]


class ReadOnlyStore(Store):
    def put_text(self, name, text):
        raise PermissionError("AccessDenied")


def test_a_forecast_still_gets_its_day_when_the_store_cannot_keep_it():
    day = dt.date(2026, 10, 16)
    firms = Firms({"VIIRS_NOAA20_NRT": [row(34.56, -118.40, day, "2033")]})

    assert "2026-10-16,2033" in day_text(day, ReadOnlyStore(), firms)


def test_the_daily_job_fails_loudly_when_it_cannot_keep_the_day():
    firms = Firms({"VIIRS_NOAA20_NRT": [row(34.56, -118.40, dt.date(2026, 10, 17), "2033")]})

    with pytest.raises(RuntimeError, match="2026-10-17"):
        firms_history.keep_yesterday(store=ReadOnlyStore(), fetch_rows=firms, nrt_from=NRT_FROM, now=NOW)


def test_a_dated_forecast_sees_its_days_detections_from_the_store(monkeypatch):
    # Six weeks back: the live FIRMS API no longer serves this window.
    issued = dt.datetime(2026, 9, 11, 12, tzinfo=UTC)
    store = Store({f"{d}.nrt.csv": "latitude,longitude,acq_date,acq_time,frp\n" for d in
                   ("2026-09-08", "2026-09-09", "2026-09-10")})
    store["2026-09-11.nrt.csv"] = f"latitude,longitude,acq_date,acq_time,frp\n{LAT},{LON},2026-09-11,0930,40\n"
    monkeypatch.delenv("FIRMS_SNAPSHOT_DIR", raising=False)
    monkeypatch.setenv("FIRMS_HISTORY_S3", "s3://example-bucket/history/firms")
    monkeypatch.setattr(firms_history, "open_store", lambda uri: store)
    monkeypatch.setattr(dynamic_builder, "_fetch_firms_csv", lambda *a, **k: "")
    monkeypatch.setattr(dynamic_builder, "fetch_weather_with_quality",
                        lambda *a, **k: ({n: np.zeros((64, 64), np.float32) for n in
                                          ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")}, {}))

    dyn = dynamic_builder.build_dynamic_for_tile(LAT, LON, T_seq=3, hours_step=24, ref_time=issued,
                                                 channel_order=["fire_t", "u", "v", "gust", "tempC", "q", "precip"])

    col, row_ = ~tile_affine(lonlat_to_tile(LON, LAT)) * lonlat_to_xy_m(LON, LAT)
    assert dyn[-1, 0, int(row_), int(col)] > 0, "the 09:30Z detection belongs in the forecast's latest day"


class UnreadableStore(Store):
    """What S3 answers credentials that cannot read the prefix."""

    def get_text(self, name):
        raise PermissionError("AccessDenied")


def test_a_dated_forecast_runs_as_before_when_the_store_cannot_be_read(monkeypatch, capsys):
    # Turning the store on must not cost the forecasts that worked without it:
    # they fall back to the live API (and an ignition seed, if asked) as before.
    issued = dt.datetime(2026, 9, 11, 12, tzinfo=UTC)
    live_api = f"latitude,longitude,acq_date,acq_time,frp\n{LAT},{LON},2026-09-11,0930,40\n"
    monkeypatch.delenv("FIRMS_SNAPSHOT_DIR", raising=False)
    monkeypatch.setenv("FIRMS_HISTORY_S3", "s3://example-bucket/history/firms")
    monkeypatch.setattr(firms_history, "open_store", lambda uri: UnreadableStore())
    monkeypatch.setattr(dynamic_builder, "_fetch_firms_csv", lambda *a, **k: live_api)
    monkeypatch.setattr(dynamic_builder, "fetch_weather_with_quality",
                        lambda *a, **k: ({n: np.zeros((64, 64), np.float32) for n in
                                          ("u", "v", "gust", "temp", "tempC", "rh", "q", "prcp", "precip")}, {}))

    dyn = dynamic_builder.build_dynamic_for_tile(LAT, LON, T_seq=3, hours_step=24, ref_time=issued,
                                                 channel_order=["fire_t", "u", "v", "gust", "tempC", "q", "precip"])

    col, row_ = ~tile_affine(lonlat_to_tile(LON, LAT)) * lonlat_to_xy_m(LON, LAT)
    assert dyn[-1, 0, int(row_), int(col)] > 0, "the live API's detection is the forecast's fire"
    assert "FIRMS history unavailable" in capsys.readouterr().out, "the fallback is said, not silent"


# ---------------- Writing only into the Ignis account ----------------

IGNIS, OTHER = "111122223333", "444455556666"


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


class FakeSts:
    def __init__(self, account):
        self.account, self.calls = account, 0

    def get_caller_identity(self):
        self.calls += 1
        return {"Account": self.account, "Arn": f"arn:aws:iam::{self.account}:user/someone"}


def s3_store(caller_account):
    s3, sts = FakeS3(), FakeSts(caller_account)
    return firms_history.S3Store("s3://ignis-bucket/history/firms", s3=s3, sts=sts), s3, sts


def test_writes_with_another_accounts_credentials_are_refused(monkeypatch):
    # The shell's default profile is another job's account.
    monkeypatch.setenv("IGNIS_AWS_ACCOUNT_ID", IGNIS)
    store, s3, _ = s3_store(OTHER)

    with pytest.raises(firms_history.WrongAwsAccount, match="6666"):
        store.put_text("2026-10-16.nrt.csv", "latitude\n")
    assert s3.puts == []


def test_writes_are_refused_when_the_ignis_account_is_not_configured(monkeypatch):
    monkeypatch.delenv("IGNIS_AWS_ACCOUNT_ID", raising=False)
    store, s3, _ = s3_store(IGNIS)

    with pytest.raises(firms_history.WrongAwsAccount, match="IGNIS_AWS_ACCOUNT_ID"):
        store.put_text("2026-10-16.nrt.csv", "latitude\n")
    assert s3.puts == []


def test_writes_in_the_ignis_account_name_the_bucket_owner_too(monkeypatch):
    monkeypatch.setenv("IGNIS_AWS_ACCOUNT_ID", IGNIS)
    store, s3, sts = s3_store(IGNIS)

    store.put_text("2026-10-16.nrt.csv", "a\n")
    store.put_text("2026-10-17.nrt.csv", "b\n")

    assert [put["Key"] for put in s3.puts] == ["history/firms/2026-10-16.nrt.csv", "history/firms/2026-10-17.nrt.csv"]
    # S3 rejects the write if the bucket belongs to any other account.
    assert {put["ExpectedBucketOwner"] for put in s3.puts} == {IGNIS}
    assert sts.calls == 1, "the account is checked once per store"


def test_a_forecast_still_gets_its_day_when_the_account_is_wrong(monkeypatch):
    monkeypatch.setenv("IGNIS_AWS_ACCOUNT_ID", IGNIS)
    store, s3, _ = s3_store(OTHER)
    store.get_text = lambda name: None
    day = dt.date(2026, 10, 16)

    text = day_text(day, store, Firms({"VIIRS_NOAA20_NRT": [row(34.56, -118.40, day, "2033")]}))

    assert "2026-10-16,2033" in text and s3.puts == []
