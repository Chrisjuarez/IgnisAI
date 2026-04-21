# docker/tilesvc.Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PORT=8008 PYTHONPATH=/app

# Runtime deps — libexpat1, libgeos, libproj needed by rasterio/shapely/pyproj at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
      libexpat1 libgeos-dev libproj-dev libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/tilesvc/requirements.txt /app/requirements.txt

# ---- Geo stack: wheels first, fallback to system GDAL + build ----
RUN set -eux; \
  python -m pip install --upgrade pip; \
  pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 rasterio==1.4.3 shapely==2.1.2 pyproj==3.7.2 \
  || { \
    echo "Wheels unavailable — installing GDAL toolchain and building from source"; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ make \
      gdal-bin libgdal-dev \
      proj-bin libproj-dev \
    && rm -rf /var/lib/apt/lists/*; \
    export GDAL_CONFIG=/usr/bin/gdal-config; \
    python -m pip install --no-cache-dir numpy==1.26.4; \
    python -m pip install --no-cache-dir rasterio==1.4.3 shapely==2.1.2 pyproj==3.7.2; \
  }

# PyTorch CPU-only (installed before requirements.txt to avoid pulling CUDA version from PyPI)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.1

# Remaining Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip check

# Application code. The production ML package is mounted or installed through
# IGNIS_ML_SOURCE_PATH/MODEL_URL at runtime; the local ignis_ml tree remains only
# for legacy utility modules used outside the v3 serving path.
COPY ignis_ml /app/ignis_ml
COPY services/tilesvc /app/services/tilesvc

# Model files: copies whatever is in models/ (at minimum .gitkeep).
# If your .pt file is committed or present in the build context, it gets baked in.
# Otherwise, set MODEL_URL env var and the entrypoint downloads it at startup.
RUN mkdir -p /app/models
COPY models/ /app/models/

# Entrypoint handles model download from MODEL_URL if .pt not in image
COPY docker/tilesvc-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8008
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "services.tilesvc.app:app", "--host", "0.0.0.0", "--port", "8008"]
