# v4 Retrain Pipeline — Implementation Scaffold

This is the code scaffold for `docs/v4-retraining-gameplan.md`. It gives you
everything needed to *run* the v4 retrain on your own GPU. What this scaffold
does **not** do: download the ~71 GB TS-SatFire dataset, run training (needs a
CUDA GPU), or fetch perimeters — those require your Kaggle/Earthdata creds and
compute. See `docs/heatmap-diagnosis.md` for the why.

## What shipped in this scaffold

| File | Phase | Status |
| --- | --- | --- |
| `ignis_ml/config.v4.yaml` | 2 | Complete — v4 channel schema, Tversky wired, wind protected, delta threshold sweep, Santa-Ana sampling block |
| `ignis_ml/scripts/ingest_ts_satfire.py` | 1 | Runnable scaffold — grid/tiling/windowing/npz+meta writing done; one `TODO(dataset)` for the TS-SatFire band layout |
| `ignis_ml/src/training/v4_losses.py` | 3 | Complete — `tversky_loss` (v3 never applied it!), `wind_align_loss`, `v4_total_loss` |
| `ignis_ml/src/training/v4_sampler.py` | 3 | Complete — Santa-Ana up-weighted `WeightedRandomSampler` |
| `ignis_ml/scripts/eval_historical.py` | 5 | Runnable scaffold — metrics (IoU/Dice/CSI/Hausdorff) + HTTP predictor complete; needs `data/perimeters/*.geojson` |
| `ignis_ml/notebooks/02_palisades_mini_pipeline.ipynb` | 4 | 14-section serving mirror; loads real `services/tilesvc` modules |
| `ignis_ml/tests/test_ingest_tssatfire.py` | 1 | Passing — channel assembly, schema, Santa-Ana detection, eval metrics |
| `models/CHANGELOG.md` | 7 | v3→v4 change log started |

## Two fixes beyond the original gameplan (found in the audit)

1. **Tversky was a no-op in v3.** `config.yaml` enabled it, but
   `train_nautilus.loss_bce_dice_focal` only computes BCE+Dice(+focal). v4 wires
   it in via `v4_losses.tversky_loss`. This alone should help recall on sparse
   positives.
2. **Wind was being zeroed in training.** `channel_dropout_p: 0.15` includes
   `u/v/gust`. v4 sets `channel_dropout_p: 0.05` + `channel_dropout_exclude:
   [fire_t,u,v,viirs_i4]`. **NOTE:** `dataset.py::_augment` must be taught to
   read `channel_dropout_exclude` (it currently only protects `fire_t`). One-line
   change — see "Trainer wiring" below.

## Run order

```bash
# 0. from repo root, with the ignis_ml deps installed
pip install -r e2e/requirements.txt   # or your training env

# 1. (after Kaggle download to data/tssatfire_raw/) ingest -> tiles
python -m ignis_ml.scripts.ingest_ts_satfire --dry-run --max-fires 3   # smoke
python -m ignis_ml.scripts.ingest_ts_satfire                            # full

# 2. tests
python -m pytest ignis_ml/tests/test_ingest_tssatfire.py -q

# 3. train on GPU (after trainer wiring below)
cd ignis_ml && python train_nautilus.py --config config.v4.yaml

# 4. iterate / sanity in the notebook
jupyter lab ignis_ml/notebooks/02_palisades_mini_pipeline.ipynb

# 5. OOD eval against a running tilesvc (or local once wired)
python -m ignis_ml.scripts.eval_historical --mode http \
  --tilesvc https://ignisai-tilesvc.onrender.com --out models/eval/v3_baseline.csv
```

## Trainer wiring (the edits to `ignis_ml/train_nautilus.py`)

These are intentionally NOT applied so `train_nautilus.py` stays v3-reproducible.
Apply them on a `v4` branch:

1. **`--config` flag** in `main()`: load `config.v4.yaml` instead of `config.yaml`.
2. **Sampler**: when `CFG.training.sampling` exists, swap
   `build_train_sampler(...)` for:
   ```python
   from src.training.v4_sampler import build_santa_ana_sampler
   sampler, sa_stats = build_santa_ana_sampler(
       train_files, meta_dir=Path(CFG.datasets.tssatfire.meta_dir),
       santa_ana_boost=CFG.training.sampling.santa_ana_boost)
   ```
3. **Loss**: after computing the base `loss = loss_bce_dice_focal(...)`, wrap:
   ```python
   from src.training.v4_losses import v4_total_loss
   loss = v4_total_loss(
       logits, y, base_loss=loss,
       tversky_cfg=CFG.training.loss.get("tversky"),
       wind_cfg=CFG.training.loss.get("wind_align"),
       fire_last=x_dyn[:, -1, fire_idx:fire_idx+1],   # [B,1,H,W]
       u_mean=x_dyn[:, :, u_idx].mean(dim=(1,2,3)) * 30 - 15,  # de-normalize [0,1]->m/s
       v_mean=x_dyn[:, :, v_idx].mean(dim=(1,2,3)) * 30 - 15)
   ```
   (u,v are normalized [0,1] from [-15,15]; de-normalize as shown.)
4. **`dataset.py::_augment` channel-dropout exclude**: read
   `channel_dropout_exclude` from config and skip those indices, not just
   `fire_t`.
5. **Bump `ARCH_VERSION`** in `ignis_ml/src/models/convlstm_unet.py` to `"v4"`
   so checkpoints don't collide and tilesvc can refuse a mismatched arch.

## Deployment (gameplan Phase 6, condensed)

After training produces `models/convlstm_unet_v4_delta_Cd18_Cs9_H64_T6.pt` and
`models/calibration_v4.json`, set the tilesvc Render env: `MODEL_PATH`,
`MODEL_SHA256`, `MODEL_CD=18`, `MODEL_CS=9`, `MODEL_DYNAMIC_ORDER`,
`MODEL_STATIC_ORDER`, `REQUIRED_ARCH_VERSION=v4`, `CALIBRATION_PATH`,
`CALIBRATION_REQUIRED=1`. Rebuild the docker image so
`static_catalog.production.json` drops impervious/population. Verify `/healthz`
shows `runtime_arch_version: v4`, `Cd:18`, `Cs:9`.

## Honest limits

- Predictions will still **under-estimate magnitude** on day 2 of Santa-Ana
  events (14,500 acres/day is the extreme tail of any training set). Keep the
  "advisory, not a perimeter" framing.
- The renderer redesign (tight, day-banded, NASA-like) is a **separate** task
  (frontend `addPredictionOverlay.js` + `MapComponent.jsx`) — not in this
  scaffold because you prioritized the retrain pipeline. The diagnosis doc has
  the exact changes when you want them.
