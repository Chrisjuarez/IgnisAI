"""Tests for how fine-tuning chains samples into multi-day rollouts.

A rollout step means "one day later". Chaining samples across a skipped day
would teach the model a two-day jump as if it were one.
"""
from pathlib import Path

import numpy as np

from ignis_ml.scripts.finetune_rollout import load_corpus, rollout_groups


def days(*indices):
    return [(i, Path(f"fire_t{i:03d}.npz")) for i in indices]


def test_three_consecutive_days_make_one_group():
    assert rollout_groups(days(0, 1, 2), 3) == [[Path("fire_t000.npz"), Path("fire_t001.npz"), Path("fire_t002.npz")]]


def test_a_gap_breaks_the_run():
    assert rollout_groups(days(0, 2, 3), 3) == []


def test_runs_on_both_sides_of_a_gap_are_kept():
    groups = rollout_groups(days(0, 1, 2, 4, 5, 6), 3)

    assert [[p.stem for p in g] for g in groups] == [
        ["fire_t000", "fire_t001", "fire_t002"],
        ["fire_t004", "fire_t005", "fire_t006"],
    ]


def test_load_corpus_keeps_day_indices_in_order(tmp_path):
    for fire, day in (("b", 1), ("a", 2), ("a", 0)):
        np.savez(tmp_path / f"{fire}_t{day:03d}.npz", fire_id=np.array(fire), day_index=np.array(day))

    corpus = load_corpus(tmp_path)

    assert {fire: [day for day, _ in samples] for fire, samples in corpus.items()} == {"a": [0, 2], "b": [1]}
