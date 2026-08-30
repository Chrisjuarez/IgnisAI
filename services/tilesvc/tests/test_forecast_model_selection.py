"""Guards on how /predict_multistep chooses a forecaster.

The first attempt at a baseline comparison reported that the deterministic
baseline matched the learned model almost exactly. It had not: the deployed
build did not know ?model= at all, FastAPI dropped the unknown query parameter,
and both runs were the learned model. A comparison that cannot fail is worse
than no comparison, so the selection is guarded here.
"""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def _endpoint() -> ast.FunctionDef:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict_multistep":
            return node
    raise AssertionError("predict_multistep not found in app.py")


def _string_constants(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_supported_models_are_declared_as_a_named_set():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    declared = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert "SUPPORTED_FORECAST_MODELS" in declared, (
        "the accepted ?model= values must be declared in one place, not implied "
        "by whichever branches happen to compare against a string literal"
    )
    assert _string_constants(declared["SUPPORTED_FORECAST_MODELS"]) == {"ignis", "downwind"}


def test_unknown_model_is_rejected_rather_than_falling_through():
    body = ast.dump(_endpoint())

    assert "SUPPORTED_FORECAST_MODELS" in body, (
        "predict_multistep must validate ?model= against the declared set; "
        "without this an unknown value silently runs the learned model"
    )
    assert "HTTPException" in body


def test_baseline_and_learned_rollouts_are_mutually_exclusive():
    endpoint = _endpoint()
    calls = {
        node.func.id
        for node in ast.walk(endpoint)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"baseline_rollout", "_rollout_multistep_predictions"} <= calls

    for branch in [n for n in ast.walk(endpoint) if isinstance(n, ast.If)]:
        taken = {
            node.func.id
            for node in ast.walk(branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "baseline_rollout" in taken and "_rollout_multistep_predictions" in taken:
            # Only acceptable if they sit on opposite arms of this branch.
            body_calls = {
                node.func.id
                for stmt in branch.body
                for node in ast.walk(stmt)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert not ("baseline_rollout" in body_calls and "_rollout_multistep_predictions" in body_calls), (
                "the baseline must replace the learned rollout, not run after it - "
                "running both costs a full inference that is then discarded"
            )
