#!/bin/sh
set -e

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

exec "$@"
