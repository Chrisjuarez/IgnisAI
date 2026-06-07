#!/usr/bin/env bash
# run_v4.sh — one-command v4 retrain pipeline ("press go").
#
# Prereqs (one-time):
#   1. Put data on the big drive:   export IGNIS_DATA_ROOT=/Volumes/<YourDrive>/ignis-data
#   2. Download TS-SatFire into $IGNIS_DATA_ROOT/tssatfire_raw/
#         kaggle datasets download -d z789456sx/ts-satfire -p "$IGNIS_DATA_ROOT/tssatfire_raw" --unzip
#   3. Fill the TODO(dataset) band map in ignis_ml/scripts/ingest_ts_satfire.py
#   4. pip install -r e2e/requirements.txt  (torch, rasterio, pyyaml, scikit-learn, pillow)
#
# Then just:   ./run_v4.sh
set -euo pipefail
cd "$(dirname "$0")"

: "${IGNIS_DATA_ROOT:=$(pwd)/data}"
export IGNIS_DATA_ROOT
echo "==> IGNIS_DATA_ROOT=$IGNIS_DATA_ROOT"

TILES_DIR="$IGNIS_DATA_ROOT/tssatfire_500m_T6"

# 1) Ingest TS-SatFire -> tiles (skip if already populated)
if [ -z "$(ls -A "$TILES_DIR" 2>/dev/null | grep -m1 '\.npz' || true)" ]; then
  echo "==> [1/3] Ingesting TS-SatFire -> $TILES_DIR"
  python -m ignis_ml.scripts.ingest_ts_satfire
else
  echo "==> [1/3] Tiles already present in $TILES_DIR — skipping ingestion"
fi

# 2) Train v4 (delta + derived + Tversky + wind-align + Santa-Ana sampler)
echo "==> [2/3] Training v4"
( cd ignis_ml && python train_v4.py )

# 3) OOD eval vs observed perimeters (needs data/perimeters/*.geojson)
echo "==> [3/3] OOD eval (optional — needs perimeters + a running tilesvc)"
python -m ignis_ml.scripts.eval_historical --mode http \
  --tilesvc "${TILE_SVC:-https://ignisai-tilesvc.onrender.com}" \
  --out "$IGNIS_DATA_ROOT/models/eval/v4_eval.csv" || \
  echo "   (eval skipped — add data/perimeters/*.geojson and/or a reachable tilesvc)"

echo "==> done. Checkpoint + calibration are under: $(python -c 'import sys;sys.path.insert(0,"ignis_ml");from src.utils.paths import models_root;print(models_root())')"
