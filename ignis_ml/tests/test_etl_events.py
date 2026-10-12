"""Tests for which days of a fire become training samples.

A model only learns what the corpus shows it. These pin down that days when a
fire did not spread are in the corpus, so the model can learn that fires stop.
"""
import numpy as np

from ignis_ml.scripts.etl_events import SEQ_LEN, grown_cells, training_days


def frames_from_fire(*days):
    """Frames shaped like build_dynamic_for_tile's, with the given fire masks."""
    frames = np.zeros((len(days), 7, 4, 4), np.float32)
    for t, cells in enumerate(days):
        for row, col in cells:
            frames[t, 0, row, col] = 1.0
    return frames


def kept_days(frames):
    return [i for i, _, _ in training_days(frames)]


def test_a_day_when_the_fire_did_not_spread_is_kept():
    burning = [(1, 1)]
    frames = frames_from_fire(burning, burning, burning, burning)

    (i, x_dyn, y), = training_days(frames)

    assert i == 0
    assert grown_cells(x_dyn, y) == 0


def test_a_day_when_the_fire_spread_is_kept():
    frames = frames_from_fire([(1, 1)], [(1, 1)], [(1, 1)], [(1, 1), (1, 2)])

    (_, x_dyn, y), = training_days(frames)

    assert grown_cells(x_dyn, y) == 1


def test_a_fire_that_went_quiet_on_the_last_day_is_kept():
    frames = frames_from_fire([(1, 1)], [(1, 1)], [], [])

    assert kept_days(frames) == [0]


def test_a_history_with_no_fire_is_skipped():
    frames = frames_from_fire([], [], [], [(1, 1)], [(1, 1)])

    assert kept_days(frames) == [1]


def test_every_day_with_fire_in_its_history_is_kept():
    burning = [(2, 2)]
    frames = frames_from_fire([], burning, burning, burning, burning, [], [], [], burning)

    assert kept_days(frames) == [0, 1, 2, 3, 4]
    assert len(frames) - SEQ_LEN == 6


def test_grown_cells_counts_only_cells_outside_the_last_input_day():
    x_dyn = frames_from_fire([(0, 0)], [(0, 0)], [(1, 1)])
    y = frames_from_fire([(0, 0), (1, 1), (2, 2)])[0, 0]

    # (0, 0) burned earlier in the history but not on the last day, so it
    # counts as new; (1, 1) was already burning.
    assert grown_cells(x_dyn, y) == 2
