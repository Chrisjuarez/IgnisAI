import numpy as np

from services.tilesvc.display_quality import display_mask_from_static, is_static_placeholder_or_missing


def test_water_zero_channel_is_not_reported_as_static_placeholder():
    water = np.zeros((4, 4), dtype=np.float32)

    assert is_static_placeholder_or_missing("water", water) is False


def test_non_water_zero_channel_is_reported_as_static_placeholder():
    ndvi = np.zeros((4, 4), dtype=np.float32)

    assert is_static_placeholder_or_missing("ndvi", ndvi) is True


def test_display_mask_uses_water_and_high_impervious_channels():
    water = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    impervious = np.asarray(
        [
            [0.0, 0.0],
            [0.95, 0.2],
        ],
        dtype=np.float32,
    )
    stat = np.stack([water, impervious], axis=0)

    mask, summary = display_mask_from_static(stat, ["water", "impervious"])

    assert np.array_equal(mask, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert summary["available"] is True
    assert summary["masked_fraction"] == 0.5
    assert summary["masked_by_channel"]["water"] == 0.25
    assert summary["masked_by_channel"]["impervious"] == 0.25
