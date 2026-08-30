import numpy as np
import pytest

from services.tilesvc import spread_bands as sb
from services.tilesvc.grid import SIZE, lonlat_to_tile

LON, LAT = -118.555, 34.078
IDENTITY = lambda x, y: (x, y)  # noqa: E731 - keeps the test in projected metres


def _step(day, filled):
    """A frame whose burn region is `filled` (a boolean array)."""
    prob = np.where(filled, 0.9, 0.0).astype(np.float32)
    return {"prob": prob, "lead_hours": day * 24, "label": f"day {day}"}


def _blob(rows, cols):
    m = np.zeros((SIZE, SIZE), dtype=bool)
    m[rows, cols] = True
    return m


def test_bands_are_cumulative_so_a_burned_area_never_disappears():
    # The model emits a delta: day 2 fires elsewhere and day 1's cells go cold.
    rollout = [_step(1, _blob(slice(10, 14), slice(10, 14))),
               _step(2, _blob(slice(20, 24), slice(20, 24)))]

    masks = sb.cumulative_masks(rollout, sb.DEFAULT_BAND_THRESHOLD)

    assert masks[0].sum() == 16
    assert masks[1].sum() == 32, "day 2 must still contain day 1"
    assert masks[0][10:14, 10:14].all() and masks[1][10:14, 10:14].all()


def test_each_band_contains_the_one_before_it():
    rollout = [_step(1, _blob(slice(28, 32), slice(28, 32))),
               _step(2, _blob(slice(26, 34), slice(26, 34))),
               _step(3, _blob(slice(24, 36), slice(24, 36)))]

    masks = sb.cumulative_masks(rollout, sb.DEFAULT_BAND_THRESHOLD)

    for earlier, later in zip(masks, masks[1:]):
        assert np.all(later[earlier]), "a later band must cover every earlier cell"


def test_later_days_are_emitted_first_so_they_render_underneath():
    rollout = [_step(1, _blob(slice(28, 32), slice(28, 32))),
               _step(2, _blob(slice(24, 36), slice(24, 36)))]

    fc = sb.spread_bands(rollout, lonlat_to_tile(LON, LAT), IDENTITY)
    days = [f["properties"]["day"] for f in fc["features"]]

    assert days == sorted(days, reverse=True)


def test_palette_runs_deepest_on_day_one_and_fades_outward():
    assert sb.band_color(1) == "#bd0026"
    assert sb.band_color(6) == "#ffffb2"
    # Confidence decays with lead time, so day 6 must not out-shout day 1.
    assert sb.band_color(2) != sb.band_color(1)


def test_day_beyond_the_palette_clamps_rather_than_crashing():
    assert sb.band_color(99) == sb.BAND_COLORS[-1]
    assert sb.band_color(0) == sb.BAND_COLORS[0]


def test_single_pixel_specks_are_not_drawn_as_fire_fronts():
    rollout = [_step(1, _blob(slice(30, 31), slice(30, 31)))]

    fc = sb.spread_bands(rollout, lonlat_to_tile(LON, LAT), IDENTITY)

    assert fc["features"] == []


def test_a_real_sized_region_does_produce_a_band():
    rollout = [_step(1, _blob(slice(28, 34), slice(28, 34)))]

    fc = sb.spread_bands(rollout, lonlat_to_tile(LON, LAT), IDENTITY)

    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["day"] == 1
    assert fc["features"][0]["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_an_empty_forecast_produces_no_bands_rather_than_an_error():
    rollout = [_step(1, np.zeros((SIZE, SIZE), dtype=bool))]

    fc = sb.spread_bands(rollout, lonlat_to_tile(LON, LAT), IDENTITY)

    assert fc["features"] == []
    assert fc["properties"]["days"] == 1


def test_threshold_controls_what_counts_as_burned():
    prob = np.full((SIZE, SIZE), 0.30, dtype=np.float32)
    rollout = [{"prob": prob, "lead_hours": 24, "label": "day 1"}]

    assert sb.cumulative_masks(rollout, 0.10)[0].all()
    assert not sb.cumulative_masks(rollout, 0.50)[0].any()


def test_collection_declares_that_bands_are_disjoint():
    rollout = [_step(1, _blob(slice(28, 34), slice(28, 34)))]

    props = sb.spread_bands(rollout, lonlat_to_tile(LON, LAT), IDENTITY)["properties"]

    assert props["disjoint"] is True
    assert props["palette"] == "YlOrRd-reversed"


def test_bands_do_not_overlap_so_colours_never_stack():
    rollout = [_step(1, _blob(slice(28, 32), slice(28, 32))),
               _step(2, _blob(slice(26, 34), slice(26, 34))),
               _step(3, _blob(slice(24, 36), slice(24, 36)))]

    bands = sb.arrival_bands(sb.cumulative_masks(rollout, sb.DEFAULT_BAND_THRESHOLD))

    for i, a in enumerate(bands):
        for b in bands[i + 1:]:
            assert not (a & b).any(), "a cell may only belong to the day it first burned"


def test_the_union_of_bands_is_still_the_cumulative_burn():
    rollout = [_step(1, _blob(slice(28, 32), slice(28, 32))),
               _step(2, _blob(slice(26, 34), slice(26, 34))),
               _step(3, _blob(slice(24, 36), slice(24, 36)))]

    masks = sb.cumulative_masks(rollout, sb.DEFAULT_BAND_THRESHOLD)
    bands = sb.arrival_bands(masks)

    for day in range(len(masks)):
        union = np.zeros_like(masks[0])
        for band in bands[: day + 1]:
            union |= band
        assert np.array_equal(union, masks[day]), "splitting must not change what burned"


def test_a_cell_that_reburns_stays_with_its_first_day():
    rollout = [_step(1, _blob(slice(28, 34), slice(28, 34))),
               _step(2, _blob(slice(28, 34), slice(28, 34)))]

    bands = sb.arrival_bands(sb.cumulative_masks(rollout, sb.DEFAULT_BAND_THRESHOLD))

    assert bands[0].sum() == 36
    assert bands[1].sum() == 0, "day 2 adds nothing new, so it draws nothing"
