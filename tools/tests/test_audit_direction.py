"""Tests for the directional audit's geometry.

The metric decides whether the model is judged to run downwind, so a flaw in
it is indistinguishable from a flaw in the model.
"""
import math

from tools.audit_direction import angular_difference, growth_origin, spread_bearing


def square(lon: float, lat: float, half: float = 0.01):
    return {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - half, lat - half], [lon + half, lat - half],
                [lon + half, lat + half], [lon - half, lat + half],
                [lon - half, lat - half],
            ]],
        },
        "properties": {"day": 1},
    }


def scene(observed_at, forecast_at, day: int = 3):
    forecast = square(*forecast_at)
    forecast["properties"]["day"] = day
    return {
        "observed": {"features": [square(*observed_at)]},
        "forecast": {"features": [forecast]},
    }


def test_growth_is_measured_from_the_front_not_the_ignition():
    # A fire that has already run 0.1 deg east, still growing east. Measured
    # from the ignition the answer is dominated by spread that already
    # happened; measured from the front it reports what the forecast added.
    ignition = (-118.5, 34.05)
    s = scene(observed_at=(-118.4, 34.05), forecast_at=(-118.35, 34.05))

    assert spread_bearing(s, ignition) == 90.0


def test_ignition_origin_would_have_flattered_a_stalled_forecast():
    # Forecast sits exactly on the current front - the model moved nothing.
    # From the front that is correctly unscorable; from the ignition it would
    # have reported a confident eastward bearing that the model never earned.
    ignition = (-118.5, 34.05)
    front = (-118.4, 34.05)
    s = scene(observed_at=front, forecast_at=front)

    assert spread_bearing(s, ignition) is None


def test_origin_falls_back_to_ignition_before_anything_has_burned():
    ignition = (-118.5, 34.05)
    s = {"observed": {"features": []}, "forecast": {"features": [square(*ignition)]}}

    assert growth_origin(s, ignition) == ignition


def test_bearing_is_compass_convention_not_math_convention():
    ignition = (-118.5, 34.05)
    north = scene(observed_at=ignition, forecast_at=(-118.5, 34.15))

    assert spread_bearing(north, ignition) == 0.0


def test_angular_difference_wraps_the_short_way():
    assert angular_difference(350.0, 10.0) == 20.0
    assert angular_difference(10.0, 350.0) == 20.0
    assert angular_difference(0.0, 180.0) == 180.0


def test_alignment_sign_matches_the_verdict_language():
    # An upwind forecast must produce a negative cosine; this is the number the
    # whole audit reduces to.
    assert math.cos(math.radians(angular_difference(95.0, 224.0))) < 0
