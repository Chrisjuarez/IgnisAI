# services/tilesvc/firms_history.py
#
# FIRMS detections for the western US, one UTC day per file, kept in S3.
#
# A dated forecast needs its days' detections long after the live FIRMS API
# (about five days) stops serving them. FIRMS itself keeps every day, but the
# near-real-time copy the app saw live is replaced by a reprocessed one after a
# few months, so each day is kept apart by which copy it is:
#
#   <prefix>/<YYYY-MM-DD>.nrt.csv   near-real-time, as live forecasts saw it
#   <prefix>/<YYYY-MM-DD>.sp.csv    standard processing, the archive
#
# Standard library plus boto3 only: the daily job runs this file as an AWS
# Lambda, which has no `requests`.

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import urllib.error
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

WEST = (-125.1, 31.0, -101.8, 49.5)
COLUMNS = ("latitude", "longitude", "acq_date", "acq_time", "frp")
#: The products live forecasts merge (dynamic_builder.FIRMS_PRODUCTS).
NRT_PRODUCTS = ("VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT")
SP_PRODUCTS = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP")
#: Detections arrive up to a few hours after a pass; a day is kept only once
#: this long after it ends, so a stored day is complete.
SETTLE = dt.timedelta(hours=6)
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"

FetchRows = Callable[[str, Sequence[float], dt.date], List[Dict[str, str]]]


def day_csv(day: dt.date, *, store, fetch_rows: FetchRows, nrt_from: Optional[dt.date] = None,
            now: Optional[dt.datetime] = None) -> str:
    """One day of western detections as CSV: from the store, else fetched once and kept.

    `nrt_from` is the first day FIRMS still serves near-real-time products for;
    found from FIRMS when needed and not given.
    """
    for copy in ("nrt", "sp"):
        text = store.get_text(f"{day.isoformat()}.{copy}.csv")
        if text is not None:
            return text

    now = now or dt.datetime.now(dt.timezone.utc)
    nrt_from = nrt_from or near_real_time_from()
    copy = "nrt" if day >= nrt_from else "sp"
    text = _to_csv(_merged_rows(day, NRT_PRODUCTS if copy == "nrt" else SP_PRODUCTS, fetch_rows))
    if _day_end(day) + SETTLE <= now:
        name = f"{day.isoformat()}.{copy}.csv"
        try:
            store.put_text(name, text)
        except Exception as exc:
            # Keeping the day saves the next request a FIRMS fetch; this one has its rows.
            print(f"⚠️  FIRMS history could not keep {name}: {exc}")
    return text


def keep_yesterday(*, store, fetch_rows: FetchRows, nrt_from: Optional[dt.date] = None,
                   now: Optional[dt.datetime] = None) -> str:
    """Keep yesterday's near-real-time copy before FIRMS replaces it. Safe to re-run."""
    now = now or dt.datetime.now(dt.timezone.utc)
    yesterday = (now - SETTLE).date() - dt.timedelta(days=1)
    day_csv(yesterday, store=store, fetch_rows=fetch_rows, nrt_from=nrt_from, now=now)
    for copy in ("nrt", "sp"):
        name = f"{yesterday.isoformat()}.{copy}.csv"
        if store.get_text(name) is not None:
            return name
    raise RuntimeError(f"FIRMS history did not keep {yesterday.isoformat()}")


def _day_end(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), dt.timezone.utc)


def _merged_rows(day: dt.date, products: Iterable[str], fetch_rows: FetchRows) -> List[Dict[str, str]]:
    merged: Dict[tuple, Dict[str, str]] = {}
    for product in products:
        for row in fetch_rows(product, WEST, day):
            if not row.get("latitude") or not row.get("acq_date"):
                continue  # FIRMS answers some bad requests with a text line, not CSV
            key = (row["latitude"], row["longitude"], row["acq_date"], row["acq_time"])
            merged.setdefault(key, {name: row.get(name, "") for name in COLUMNS})
    return sorted(merged.values(), key=lambda r: (r["acq_date"], r["acq_time"], r["latitude"], r["longitude"]))


def _to_csv(rows: Iterable[Dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


# ---------------- FIRMS ----------------

def _map_key() -> str:
    key = (os.getenv("NASA_API_KEY") or os.getenv("FIRMS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("NASA_API_KEY is required to fetch FIRMS detections")
    return key


def _get(url: str, timeout: float = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ignis-ai-firms-history"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def firms_rows(product: str, bbox: Sequence[float], day: dt.date) -> List[Dict[str, str]]:
    """One product's detections for one UTC day in `bbox`."""
    area = ",".join(f"{float(v):.5f}" for v in bbox)
    try:
        text = _get(f"{FIRMS_API}/area/csv/{_map_key()}/{product}/{area}/1/{day.isoformat()}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    return list(csv.DictReader(io.StringIO(text)))


def near_real_time_from(product: str = "VIIRS_NOAA20_NRT") -> dt.date:
    """The first day FIRMS still serves near-real-time detections for."""
    text = _get(f"{FIRMS_API}/data_availability/csv/{_map_key()}/ALL")
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("data_id") == product:
            return dt.date.fromisoformat(row["min_date"])
    raise RuntimeError(f"FIRMS lists no availability for {product}")


# ---------------- S3 ----------------

class WrongAwsAccount(PermissionError):
    """Refused: these credentials, or the bucket, are not the Ignis account's."""


def _ignis_account() -> str:
    account = (os.getenv("IGNIS_AWS_ACCOUNT_ID") or "").strip()
    if not account:
        raise WrongAwsAccount("IGNIS_AWS_ACCOUNT_ID is not set; refusing to write FIRMS history")
    return account


class S3Store:
    """Text files under one s3://bucket/prefix, written only in the Ignis account.

    A machine can hold credentials for more than one AWS account, and a tool run
    without a profile picks up whichever is the default. Before its first write
    the store checks whose credentials it has, and every write names the bucket
    owner it expects, so S3 itself rejects a bucket in any other account.
    """

    def __init__(self, uri: str, *, s3=None, sts=None):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected s3://bucket/prefix, got {uri!r}")
        self.bucket, self.prefix = parsed.netloc, parsed.path.strip("/")
        if s3 is None or sts is None:
            import boto3

            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            s3 = s3 or boto3.client("s3", region_name=region)
            sts = sts or boto3.client("sts", region_name=region)
        self.client, self._sts = s3, sts
        self._account_checked = False

    def _check_account(self) -> str:
        expected = _ignis_account()
        if not self._account_checked:
            actual = self._sts.get_caller_identity()["Account"]
            if actual != expected:
                raise WrongAwsAccount(
                    f"credentials are for AWS account …{actual[-4:]}, not the Ignis account …{expected[-4:]}; "
                    "refusing to write FIRMS history"
                )
            self._account_checked = True
        return expected

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def get_text(self, name: str) -> Optional[str]:
        from botocore.exceptions import ClientError

        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(name))["Body"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None
            raise
        return body.read().decode("utf-8")

    def put_text(self, name: str, text: str) -> None:
        owner = self._check_account()
        self.client.put_object(Bucket=self.bucket, Key=self._key(name), Body=text.encode("utf-8"),
                               ContentType="text/csv", ExpectedBucketOwner=owner)


def open_store(uri: str) -> S3Store:
    return S3Store(uri)


def lambda_handler(event, _context):
    """Daily job: keep yesterday's near-real-time copy. Event may name a `day`."""
    store = open_store(os.environ["FIRMS_HISTORY_S3"])
    day = (event or {}).get("day")
    if day:
        when = dt.datetime.combine(dt.date.fromisoformat(day) + dt.timedelta(days=1), dt.time(), dt.timezone.utc) + SETTLE
        return {"stored": keep_yesterday(store=store, fetch_rows=firms_rows, now=when)}
    return {"stored": keep_yesterday(store=store, fetch_rows=firms_rows)}
