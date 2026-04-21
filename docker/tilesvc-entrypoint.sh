#!/bin/sh
set -e

if [ -n "$IGNIS_ML_SOURCE_PATH" ] && [ -d "$IGNIS_ML_SOURCE_PATH" ]; then
  export PYTHONPATH="$IGNIS_ML_SOURCE_PATH:${PYTHONPATH:-/app}"
  if [ "${INSTALL_IGNIS_ML:-1}" = "1" ] && [ -f "$IGNIS_ML_SOURCE_PATH/pyproject.toml" ]; then
    echo "Installing Ignis ML package from $IGNIS_ML_SOURCE_PATH ..."
    if ! python -m pip install --no-cache-dir -e "$IGNIS_ML_SOURCE_PATH"; then
      echo "Editable install failed; continuing with PYTHONPATH source mount: $IGNIS_ML_SOURCE_PATH" >&2
    fi
  else
    echo "Using Ignis ML source from PYTHONPATH: $IGNIS_ML_SOURCE_PATH"
  fi
fi

# If MODEL_URL is set and the model file doesn't exist yet, download it.
# This allows hosting the model externally (GitHub Release, Google Drive, etc.)
# instead of baking it into the Docker image.
if [ -n "$MODEL_URL" ] && [ ! -f "${MODEL_PATH:-/app/models/model.pt}" ]; then
  # Strip whitespace/newlines that Render's env var input may inject
  CLEAN_URL=$(echo "$MODEL_URL" | tr -d '[:space:]')
  echo "Downloading model from $CLEAN_URL ..."
  mkdir -p "$(dirname "${MODEL_PATH:-/app/models/model.pt}")"
  curl -fSL -o "${MODEL_PATH:-/app/models/model.pt}" "$CLEAN_URL"
  echo "Model downloaded successfully ($(du -h "${MODEL_PATH:-/app/models/model.pt}" | cut -f1))."
fi

if [ -n "$MODEL_SHA256" ]; then
  MODEL_FILE="${MODEL_PATH:-/app/models/model.pt}"
  if [ ! -f "$MODEL_FILE" ]; then
    echo "MODEL_SHA256 is set but model file is missing: $MODEL_FILE" >&2
    exit 1
  fi
  ACTUAL_SHA=$(python - "$MODEL_FILE" <<'PY'
import hashlib
import sys

h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)
  if [ "$ACTUAL_SHA" != "$MODEL_SHA256" ]; then
    echo "Model checksum mismatch for $MODEL_FILE" >&2
    echo "expected: $MODEL_SHA256" >&2
    echo "actual:   $ACTUAL_SHA" >&2
    exit 1
  fi
  echo "Model checksum verified: $ACTUAL_SHA"
fi

PRED_ENABLED=$(printf '%s' "${PREDICTIONS_ENABLED:-true}" | tr '[:upper:]' '[:lower:]')
if [ "$PRED_ENABLED" != "false" ] && [ "$PRED_ENABLED" != "0" ] && [ "$PRED_ENABLED" != "no" ] && [ "$PRED_ENABLED" != "off" ]; then
  MODEL_FILE="${MODEL_PATH:-/app/models/model.pt}"
  if [ ! -f "$MODEL_FILE" ]; then
    echo "Predictions are enabled but model file is missing: $MODEL_FILE" >&2
    echo "Set MODEL_URL to the pinned GitHub Release asset, mount MODEL_PATH, or set PREDICTIONS_ENABLED=false." >&2
    exit 1
  fi
fi

exec "$@"
