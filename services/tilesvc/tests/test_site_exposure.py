import numpy as np
import pytest

from services.tilesvc import site_exposure
from services.tilesvc.grid import SIZE, lonlat_to_tile

SITE_LON, SITE_LAT = -118.555, 34.078


def _rollout(values):
    """One frame per value, with that value filled across the whole tile."""
    return [
        {"prob": np.full((SIZE, SIZE), v, dtype=np.float32),
         "lead_hours": (i + 1) * 24,
         "label": f"day {i + 1}"}
        for i, v in enumerate(values)
    ]


def _sample(values, **kwargs):
    tile = lonlat_to_tile(SITE_LON, SITE_LAT)
    params = dict(site_lon=SITE_LON, site_lat=SITE_LAT,
                  ignition_lon=SITE_LON, ignition_lat=SITE_LAT)
    params.update(kwargs)
    return site_exposure.sample_exposure(_rollout(values), tile, **params)


def test_site_inside_the_tile_gets_one_entry_per_forecast_day():
    result = _sample([0.01, 0.2, 0.6])

    assert result["covered"] is True
    assert [e["day"] for e in result["series"]] == [1, 2, 3]
    assert [e["lead_hours"] for e in result["series"]] == [24, 48, 72]


def test_risk_classes_follow_the_same_breaks_the_map_uses():
    result = _sample([0.01, 0.10, 0.30, 0.90])

    assert [e["risk"] for e in result["series"]] == ["low", "medium", "high", "extreme"]


def test_exposure_does_not_fall_once_the_site_has_burned():
    # A delta model reports near-zero on day 3 for a site that burned on day 2.
    # Cumulative exposure must not imply the risk went away.
    result = _sample([0.05, 0.60, 0.02])

    daily = [e["daily_probability"] for e in result["series"]]
    cumulative = [e["cumulative_probability"] for e in result["series"]]

    assert daily[2] < daily[1], "daily delta drops after the burn, as the model reports it"
    assert cumulative == sorted(cumulative), "cumulative exposure must never decrease"
    assert cumulative[2] == pytest.approx(0.60, abs=1e-3)
    assert result["series"][2]["risk"] == "extreme"


def test_headline_figures_describe_the_whole_horizon():
    result = _sample([0.05, 0.60, 0.02])

    assert result["probability_within_horizon"] == pytest.approx(0.60, abs=1e-3)
    assert result["risk_within_horizon"] == "extreme"
    assert result["peak_day"]["day"] == 2


def test_arrival_reports_the_first_day_over_the_threshold():
    result = _sample([0.01, 0.02, 0.4, 0.9], arrival_threshold=0.1)

    assert result["arrival"]["reached"] is True
    assert result["arrival"]["day"] == 3
    assert result["arrival"]["lead_hours"] == 72


def test_a_fire_that_never_reaches_the_site_reports_no_arrival():
    result = _sample([0.001, 0.002, 0.003], arrival_threshold=0.1)

    assert result["arrival"]["reached"] is False
    assert result["arrival"]["day"] is None


def test_peak_day_is_the_worst_single_day_not_the_last():
    result = _sample([0.05, 0.80, 0.10])

    assert result["peak_day"]["day"] == 2
    assert result["peak_day"]["daily_probability"] == pytest.approx(0.80, abs=1e-3)


def test_a_site_outside_the_tile_is_uncovered_rather_than_zero_risk():
    # Half a degree of longitude is ~46 km here — well beyond the 32 km tile.
    result = _sample([0.9, 0.9, 0.9], site_lon=SITE_LON + 0.5)

    assert result["covered"] is False
    assert result["reason"] == "site_outside_forecast_tile"
    assert result["series"] == []
    assert result["peak_day"] is None
    assert result["probability_within_horizon"] is None
    assert result["separation_km"] > 32


def test_separation_is_reported_for_a_site_offset_from_the_ignition():
    result = _sample([0.1], site_lon=SITE_LON + 0.05)

    assert result["covered"] is True
    assert 3 < result["separation_km"] < 6


def test_site_pixel_rejects_coordinates_beyond_the_tile_edge():
    tile = lonlat_to_tile(SITE_LON, SITE_LAT)

    assert site_exposure.site_pixel(tile, SITE_LON, SITE_LAT) is not None
    assert site_exposure.site_pixel(tile, SITE_LON + 1.0, SITE_LAT) is None
