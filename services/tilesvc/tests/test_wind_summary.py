import pytest

from services.tilesvc.wind_summary import CALM_MS, compass_point, wind_from_channels, wind_summary


def test_wind_blowing_east_reports_east():
    # u is eastward. A pure +u wind blows toward the east.
    w = wind_summary(10.0, 0.0)

    assert w["toward"] == "E"
    assert w["toward_deg"] == pytest.approx(90.0)


def test_wind_blowing_north_reports_north():
    assert wind_summary(0.0, 10.0)["toward"] == "N"


def test_santa_ana_offshore_wind_reports_southwest():
    # The Palisades case: wind out of the northeast, blowing toward the ocean.
    w = wind_summary(-7.0, -7.0)

    assert w["toward"] == "SW"
    assert 220 < w["toward_deg"] < 230


def test_speed_is_reported_in_both_units():
    w = wind_summary(3.0, 4.0)

    assert w["speed_ms"] == pytest.approx(5.0)
    assert w["speed_mph"] == pytest.approx(11.2, abs=0.1)


def test_calm_air_does_not_claim_a_direction():
    # Drawing a confident arrow for noise would imply a steer the model lacks.
    w = wind_summary(0.05, -0.05)

    assert w["calm"] is True
    assert w["toward_deg"] is None
    assert w["toward"] == "calm"


def test_just_above_calm_does_get_a_direction():
    w = wind_summary(CALM_MS + 0.2, 0.0)

    assert w["calm"] is False
    assert w["toward"] == "E"


def test_missing_wind_channels_are_reported_not_guessed():
    assert wind_summary(None, None) == {"available": False, "reason": "wind_channels_missing"}


def test_direction_convention_is_stated_in_the_payload():
    # FROM vs TOWARD is the single easiest thing to get backwards here.
    assert "TOWARD" in wind_summary(5.0, 5.0)["note"]


def test_reads_the_channel_stats_the_response_already_builds():
    w = wind_from_channels({
        "u": {"mean": 1.05}, "v": {"mean": 1.90}, "gust": {"mean": 8.73},
    })

    assert w["toward"] == "NNE"
    assert w["gust_ms"] == pytest.approx(8.73)


def test_all_sixteen_compass_points_are_reachable():
    seen = {compass_point(deg) for deg in range(0, 360, 5)}

    assert len(seen) == 16
