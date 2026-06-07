"""Data-root resolution so the big datasets/tiles/models can live on an
external drive while the code stays in the repo.

Precedence for the data root:
  1. $IGNIS_DATA_ROOT           (e.g. /Volumes/T7/ignis-data)
  2. <repo>/data               (default; same as before)

Any config path that begins with "data/" is rebased onto the data root, so
config.v4.yaml needs no edits to move to the external drive — just set the env
var once:

    export IGNIS_DATA_ROOT=/Volumes/<YourDrive>/ignis-data

Models default to <data_root>/models unless $IGNIS_MODELS_ROOT is set, so the
multi-GB checkpoints also land on the external drive.
"""
from __future__ import annotations

import os
from pathlib import Path

# repo root = three levels up from this file: ignis_ml/src/utils/paths.py
REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    env = os.environ.get("IGNIS_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else (REPO_ROOT / "data")


def models_root() -> Path:
    env = os.environ.get("IGNIS_MODELS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Keep models next to the data when an external data root is set, otherwise
    # use the repo's models/ dir (where the deployed checkpoints already live).
    if os.environ.get("IGNIS_DATA_ROOT"):
        return data_root() / "models"
    return REPO_ROOT / "models"


def resolve_data_path(p: str | os.PathLike) -> Path:
    """Rebase a config path onto the active data root.

    'data/tssatfire_500m_T6' -> <data_root>/tssatfire_500m_T6
    absolute paths are returned unchanged.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        return data_root().joinpath(*parts[1:]) if len(parts) > 1 else data_root()
    # any other relative path resolves against the repo root
    return (REPO_ROOT / p).resolve()


def describe() -> str:
    return (
        f"data_root={data_root()}  models_root={models_root()}  "
        f"(IGNIS_DATA_ROOT={'set' if os.environ.get('IGNIS_DATA_ROOT') else 'unset'})"
    )
