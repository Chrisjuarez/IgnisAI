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
    """The curve must pull the mid-range down, not sit near the identity.

    Interpolated rather than indexed by exact key: the knots move whenever the
    fit is rerun, and an exact lookup made this test about knot placement
    instead of about whether the curve corrects anything.
    """
    import numpy as np

    points = calibration_file()["points"]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    assert float(np.interp(0.5, xs, ys)) < 0.4, "mid-range must be pulled well down"
    assert float(np.interp(0.9, xs, ys)) < 0.6, "high scores must be pulled down"


def test_calibration_has_resolution_where_the_model_decides():
    """Knots must cover the range the model operates in.

    Quantile-only knots put every point where the mass is, and in a fire raster
    97% of cells sit near zero - the first fit had a knot at 0.05 and then
    nothing until 0.99, interpolating straight through the decision range.
    """
    xs = [x for x, _ in calibration_file()["points"]]
    mid = [x for x in xs if 0.2 <= x <= 0.8]

    assert len(mid) >= 5, f"only {len(mid)} knots between 0.2 and 0.8"


def test_deployed_checkpoint_is_the_one_that_scored_better():
    """The deployed model must be one that was measured, not a convenient file.

    Scored on 23 fires held out of its own training, the rollout fine-tune beat
    control60 on every metric - precision 0.121 to 0.253, recall 0.141 to
    0.344, IoU 0.044 to 0.106. control60 in turn beat v3, which had negative
    Brier skill. Each swap should be a deliberate act with numbers behind it.
    """
    deployed = env_value("MODEL_PATH")

    assert any(tag in deployed for tag in ("ft_rollout", "control60")), (
        f"deployed checkpoint {deployed!r} has no recorded evaluation; "
        "v3 measured worse than predicting the base rate"
    )
