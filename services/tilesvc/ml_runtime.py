import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def configure_ml_source() -> None:
    source_path = os.getenv("IGNIS_ML_SOURCE_PATH")
    if source_path and Path(source_path).exists() and source_path not in sys.path:
        sys.path.insert(0, source_path)


def import_model_class():
    configure_ml_source()
    try:
        from src.models.convlstm_unet import ARCH_VERSION, ConvLSTMUNet
        return ConvLSTMUNet, ARCH_VERSION, "src.models.convlstm_unet"
    except Exception:
        from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet
        try:
            from ignis_ml.src.models.convlstm_unet import ARCH_VERSION
        except Exception:
            ARCH_VERSION = "unknown"
        return ConvLSTMUNet, ARCH_VERSION, "ignis_ml.src.models.convlstm_unet"


def import_feature_helpers():
    configure_ml_source()
    try:
        from src.data.features import DERIVED_FEATURE_NAMES, append_derived_features, expected_channel_count
        return DERIVED_FEATURE_NAMES, append_derived_features, expected_channel_count, "src.data.features"
    except Exception:
        from ignis_ml.src.data.features import DERIVED_FEATURE_NAMES, append_derived_features, expected_channel_count
        return DERIVED_FEATURE_NAMES, append_derived_features, expected_channel_count, "ignis_ml.src.data.features"


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_version_info() -> Dict[str, Any]:
    configure_ml_source()
    source_path = os.getenv("IGNIS_ML_SOURCE_PATH")
    info: Dict[str, Any] = {
        "source_path": source_path,
        "source_exists": bool(source_path and Path(source_path).exists()),
    }
    if source_path and Path(source_path).exists():
        try:
            result = subprocess.run(
                ["git", "-C", source_path, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            info["git_commit"] = result.stdout.strip()
        except Exception:
            info["git_commit"] = None
        try:
            pyproject = Path(source_path) / "pyproject.toml"
            info["pyproject_sha256"] = file_sha256(pyproject)
        except Exception:
            info["pyproject_sha256"] = None
    return info


def runtime_imports() -> Tuple[Any, str, str, Any, Tuple[str, ...], str]:
    ConvLSTMUNet, arch_version, model_module = import_model_class()
    derived_names, append_derived, _expected_count, feature_module = import_feature_helpers()
    return ConvLSTMUNet, arch_version, model_module, append_derived, tuple(derived_names), feature_module
