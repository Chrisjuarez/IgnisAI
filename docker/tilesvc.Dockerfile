# docker/tilesvc.Dockerfile
FROM python:3.11-slim

ARG INSTALL_TORCH=0
ENV PYTHONUNBUFFERED=1 PORT=8008 PYTHONPATH=/app

# Minimal runtime deps first
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps list early for better caching
COPY services/tilesvc/requirements.txt /app/requirements.txt

# ---- Geo stack: wheels first, fallback to system GDAL + build ----
RUN set -eux; \
  python -m pip install --upgrade pip; \
  # Fast path: wheels only. If any wheel is missing, this fails quickly.
  pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 rasterio==1.4.3 shapely==2.1.2 pyproj==3.7.2 \
  || { \
    echo "⚠️  Wheels unavailable — installing GDAL toolchain and building from source"; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ make \
      gdal-bin libgdal-dev \
      proj-bin libproj-dev \
    && rm -rf /var/lib/apt/lists/*; \
    export GDAL_CONFIG=/usr/bin/gdal-config; \
    python -m pip install --no-cache-dir numpy==1.26.4; \
    python -m pip install --no-cache-dir rasterio==1.4.3 shapely==2.1.2 pyproj==3.7.2; \
  }; \
  # Install the rest, but don't let it re-resolve geo deps
  pip install --no-cache-dir -r /app/requirements.txt --no-deps
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip check
# Optional: only if you truly need torch in tilesvc
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.1 || true; \
    fi

# Code
COPY ignis_ml /app/ignis_ml
COPY services/tilesvc /app/services/tilesvc

EXPOSE 8008
CMD ["uvicorn","services.tilesvc.app:app","--host","0.0.0.0","--port","8008"]