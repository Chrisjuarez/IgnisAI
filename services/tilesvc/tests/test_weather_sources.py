import numpy as np

from services.tilesvc.weather_sources import aggregate_daily_weather, noaa_nomads_gfs_filter_url


def test_hourly_weather_aggregates_to_daily_transition_channels():
    hourly = {
        "precip": np.ones((24, 2, 2), dtype=np.float32),
        "u": np.arange(24, dtype=np.float32).reshape(24, 1, 1).repeat(2, axis=1).repeat(2, axis=2),
        "v": np.full((24, 2, 2), 2.0, dtype=np.float32),
        "gust": np.full((24, 2, 2), 5.0, dtype=np.float32),
        "tempC": np.full((24, 2, 2), 20.0, dtype=np.float32),
        "q": np.full((24, 2, 2), 0.01, dtype=np.float32),
    }

    out = aggregate_daily_weather(hourly, samples_per_day=24)

    assert out["precip"].shape == (1, 2, 2)
    assert np.allclose(out["precip"], 24.0)
    assert np.allclose(out["u"], 11.5)
    assert np.allclose(out["v"], 2.0)
    assert np.allclose(out["gust"], 5.0)
    assert np.allclose(out["tempC"], 20.0)
    assert np.allclose(out["q"], 0.01)


def test_gfs_three_hourly_weather_uses_eight_samples_per_day():
    arr = np.ones((16, 1, 1), dtype=np.float32)

    out = aggregate_daily_weather({"precip": arr, "u": arr * 4}, samples_per_day=8)

    assert out["precip"].shape == (2, 1, 1)
    assert np.allclose(out["precip"], 8.0)
    assert np.allclose(out["u"], 4.0)


def test_nomads_gfs_url_uses_official_filter_endpoint_and_subregion():
    url = noaa_nomads_gfs_filter_url(
        date="20260420",
        cycle="12",
        forecast_hour=24,
        bbox=(-119.0, 34.0, -118.0, 35.0),
    )

    assert "filter_gfs_0p25.pl" in url
    assert "file=gfs.t12z.pgrb2.0p25.f024" in url
    assert "var_UGRD=on" in url
    assert "var_APCP=on" in url
    assert "leftlon=-119.0" in url
    assert "rightlon=-118.0" in url
