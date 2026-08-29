"""Guards on how torch is pinned.

The CPU-only build was already reinstated as the CUDA build once: the Dockerfile
and CI installed torch from the PyTorch CPU index, then `pip install -r
requirements.txt` upgraded it back off PyPI because requirements.txt pinned a
different version. Nothing failed, nothing logged, and a 512 MB CPU instance
quietly started shipping the CUDA runtime. These tests make that regression
loud.
"""
from pathlib import Path

TILESVC = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _read(name: str) -> str:
    return (TILESVC / name).read_text(encoding="utf-8")


def _requirement_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in _read(name).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_torch_is_pinned_against_the_cpu_index():
    lines = _requirement_lines("requirements-torch.txt")

    assert f"--index-url {CPU_INDEX}" in lines, "torch must resolve from the CPU-only index"
    assert any(line.startswith("torch==") for line in lines), "torch must be pinned to an exact version"


def test_runtime_requirements_do_not_name_torch():
    named = [line for line in _requirement_lines("requirements.txt") if line.split("[")[0].split("=")[0] == "torch"]

    assert not named, (
        "requirements.txt must not name torch: it is installed from the CPU index, "
        "and naming it here lets a version bump pull the CUDA wheel from PyPI"
    )


def test_test_only_dependencies_stay_out_of_the_runtime_image():
    runtime = {line.split("[")[0].split("=")[0] for line in _requirement_lines("requirements.txt")}

    assert "pytest" not in runtime, "pytest belongs in requirements-dev.txt"


def test_dev_requirements_include_the_runtime_set():
    assert "-r requirements.txt" in _requirement_lines("requirements-dev.txt")
