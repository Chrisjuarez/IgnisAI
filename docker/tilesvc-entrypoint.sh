#!/bin/sh
set -e

# If MODEL_URL is set and the model file doesn't exist yet, download it.
# This allows hosting the model externally (GitHub Release, Google Drive, etc.)
# instead of baking it into the Docker image.
if [ -n "$MODEL_URL" ] && [ ! -f "${MODEL_PATH:-/app/models/model.pt}" ]; then
  echo "Downloading model from $MODEL_URL ..."
  mkdir -p "$(dirname "${MODEL_PATH:-/app/models/model.pt}")"
  curl -fSL -o "${MODEL_PATH:-/app/models/model.pt}" "$MODEL_URL"
  echo "Model downloaded successfully ($(du -h "${MODEL_PATH:-/app/models/model.pt}" | cut -f1))."
fi

exec "$@"
