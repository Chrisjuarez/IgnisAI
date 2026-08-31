"""Guards that the deployed model, its checksum and its calibration agree.

These three can drift apart silently, and the failure is invisible: a
calibration fitted for one checkpoint applied to another produces numbers that
look like probabilities and are not. The loader rejects a sha mismatch at
runtime, but that is a failed deploy; these catch it in review.
"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RENDER_YAML = (REPO / "render.yaml").read_text(encoding="utf-8")


def env_value(key: str) -> str:
    match = re.search(rf"- key: {key}\n\s+value: (.+)", RENDER_YAML)
    assert match, f"{key} is not set to a literal value in render.yaml"
    return match.group(1).strip().strip('"')


def calibration_file() -> dict:
    path = REPO / env_value("CALIBRATION_PATH").replace("/app/", "")
    assert path.is_file(), f"calibration file not in the image build context: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_url_and_model_path_name_the_same_checkpoint():
    assert Path(env_value("MODEL_PATH")).name == env_value("MODEL_URL").rsplit("/", 1)[-1]


def test_calibration_is_fitted_for_the_deployed_checkpoint():
    assert calibration_file()["model_sha256"] == env_value("MODEL_SHA256")


def test_calibration_is_required_so_an_uncalibrated_deploy_fails_loudly():
    # Without this the service falls back to the identity curve and serves raw
    # sigmoid scores as though they were probabilities.
    assert env_value("CALIBRATION_REQUIRED") == "1"


def test_calibration_curve_is_monotone_and_spans_the_full_range():
    points = calibration_file()["points"]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    assert xs[0] == 0.0 and xs[-1] == 1.0
    assert all(b > a for a, b in zip(xs, xs[1:])), "input must be strictly increasing"
    # A non-monotone curve would reorder the model's own ranking.
    assert all(b >= a for a, b in zip(ys, ys[1:])), "calibration must never invert"
    assert all(0.0 <= y <= 1.0 for y in ys)


def test_calibration_corrects_the_overconfidence_it_was_fitted_for():
    # The raw model reads ~0.86 where roughly 23% of cells burn. A calibration
    # that left the mid-range near the identity would not be doing its job.
    points = calibration_file()["points"]
    at = dict(points)

    assert at[0.5] < 0.25, "mid-range scores must be pulled well down"
    assert at[0.9] < 0.6


def test_deployed_checkpoint_is_the_one_that_scored_better():
    # v3 measured AP 0.256 and Brier skill -1.409 on held-out cells against
    # control60's 0.808 and +0.147. Swapping back should be a deliberate act.
    assert "control60" in env_value("MODEL_PATH")
