# IgnisAI v4 — Retraining & Mini‑Pipeline Game Plan

Goal: replace the current v3 ConvLSTM‑UNet checkpoint with a v4 checkpoint that
actually produces a reasonable fire‑spread heatmap for Palisades‑class Santa
Ana events, *and* build a notebook‑sized harness in `ignis_ml/notebooks/` that
mirrors the live tilesvc serving path so you can iterate on features, model,
and calibration without redeploying to Render.

---

## 1. Why v3 fails on Palisades — one paragraph

v3 was trained on mNDWS (modified Next‑Day Wildfire Spread, 2012–2020 western
CONUS, predominantly summer/fall vegetated‑mountain fires). Palisades is a
January coastal Santa Ana event with a strong wildland–urban interface
component. That's an out‑of‑distribution event. On top of that, the training
config `channel_dropout_p: 0.15` randomly zeroes wind during training (~15% of
steps), so the model learned to be robust to missing wind and now under‑weights
it at inference. And the static channels `impervious` and `population` act as
shortcut features that draw the prediction along the I‑405 / urban‑edge band
because that's where those channels light up in California tiles. Fix: a fresh
v4 training run on a dataset that *does* contain Santa Ana coastal fires
(TS‑SatFire 2017–2021), with the shortcut features removed and the wind signal
protected from dropout.

---

## 2. Architecture decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Backbone | Keep **ConvLSTM‑UNet v3** code (`ignis_ml/src/models/convlstm_unet.py`) | Already works in tilesvc; arch isn't the bottleneck. Lift to SwinUNETR later if v4 plateaus. |
| Tag | `v4` (bump `ARCH_VERSION` in the model module) | tilesvc rejects mismatched arch — keeps v3 and v4 from colliding. |
| Tile geometry | **32 km × 32 km at 500 m, 64×64 pixels** (unchanged) | Matches the live `services/tilesvc/grid.py` so we can deploy without re‑tiling production. |
| Dynamic channels | **9 raw + 6 derived = 15** (see §3) | Adds VIIRS I4/I5 thermal and moves ERC/BI to dynamic. |
| Static channels | **10** (drops `impervious`, `population`, `fuel3` — see §3) | Removes the two shortcut features that draw fire to urban edge. |
| Target mode | `delta` (unchanged) | Already proven; matches tilesvc inference assumptions. |
| Compute | Single CUDA GPU (Nautilus or rent an A100) for ~24–48 h | mNDWS+TS‑SatFire merged is ~80–100 GB; one A100 finishes 30 epochs comfortably. |
| Output | `models/convlstm_unet_v4_delta_Cd15_Cs10_H64_T6.pt` + calibration JSON | Drop‑in replacement for tilesvc's `MODEL_PATH`. |

---

## 3. Channel schema changes

Old (v3) — `ignis_ml/config.yaml`:

```yaml
dynamic_order: [fire_t, u, v, gust, tempC, q, precip]    # 7 raw + 6 derived = 13
static_order:  [elev, slope, aspect_cos, aspect_sin, ndvi, bi, erc, pdsi,
                chili, impervious, water, population, fuel1, fuel2, fuel3]   # 15
```

New (v4):

```yaml
dynamic_order: [fire_t, viirs_i4, viirs_i5, u, v, gust, tempC, q, precip,
                erc, bi, ndvi]                                              # 12 raw + 6 derived = 18
static_order:  [elev, slope, aspect_cos, aspect_sin, pdsi, chili, water,
                fuel1, fuel2]                                                # 9
```

Concrete deltas:

- **Add** `viirs_i4`, `viirs_i5` — VIIRS 3.7 µm / 11.45 µm brightness temperature.
  Bring the model the actual fire intensity, not a binary FIRMS mask.
- **Move** `ndvi`, `erc`, `bi` from static → dynamic. Daily GridMET + 8‑day VIIRS
  values. These really do change day to day during Santa Ana events.
- **Drop** `impervious`, `population` — shortcut features the v3 model used to
  draw fire along the urban edge.
- **Drop** `fuel3` — review fuel PCA scores; the 3rd component carries the
  least signal and adds 1 channel of noise.
- Static `pdsi` stays static (PDSI is monthly, doesn't change inside a 6‑day
  forecast window).

---

## 4. Phased rollout

### Phase 1 — Data ingestion (~3 days of work)

**Deliverable:** `ignis_ml/scripts/ingest_ts_satfire.py` plus `data/tssatfire_500m_T6/`
populated with `.npz` tiles in the same `x_dyn/x_stat/y` schema your existing
`NpzTileDataset` already consumes.

**What it does:**

1. Downloads the TS‑SatFire archive from Kaggle (179 fires, ~71 GB,
   `kaggle.com/datasets/z789456sx/ts-satfire`). Keep a local copy under
   `data/tssatfire_raw/`.
2. For each fire, reads the GeoTIFF stack with rasterio.
3. Re‑tiles from TS‑SatFire's 256×256 @ 375 m → IgnisAI's 64×64 @ 500 m, on
   the canonical EPSG:5070 grid (`services.tilesvc.grid.tile_affine`). Each
   TS‑SatFire fire produces multiple IgnisAI tiles; tile keys land on the same
   integer grid the live tilesvc uses, so a tile generated here works byte‑for‑byte
   for inference.
4. Maps TS‑SatFire channels → v4 schema (see channel map below).
5. Stacks a rolling T=6 window from the fire's lifecycle. Each sample is a
   prediction problem: given days t−5..t, predict the new‑burn delta on day t+1.
6. Writes `.npz` with `x_dyn [6, 12, 64, 64]`, `x_stat [9, 64, 64]`,
   `y [1, 64, 64]`, plus `dyn_names` and `stat_names` arrays for safety.
7. Writes a per‑fire metadata JSON (event name, ignition lat/lon, date range,
   tile keys, "is_santa_ana" boolean derived from mean u during the fire) so
   the training sampler can use it.

**Channel mapping table:**

| v4 channel | TS‑SatFire source | mNDWS source | Notes |
| --- | --- | --- | --- |
| `fire_t` | VIIRS AF mask | FRP‑boosted MODIS | Threshold I4 brightness > 320 K |
| `viirs_i4` | VIIRS I4 brightness T | *missing → fill with 0* | New channel; mNDWS gets a zero column |
| `viirs_i5` | VIIRS I5 brightness T | *missing → fill with 0* | Same |
| `u`, `v` | GridMET wind→u,v | GridMET wind→u,v | Standard `wind_to_uv` |
| `gust` | *not in TS‑SatFire → derive 1.5 × wind* | GridMET gust | Approximation OK |
| `tempC` | GridMET tmmx/tmmn mean | GridMET temp | |
| `q` | GridMET specific humidity | GridMET rh→q | |
| `precip` | GridMET precipitation | GridMET prcp | |
| `erc` | GridMET ERC | mNDWS static (was constant — broken) | Move to dynamic |
| `bi` | GridMET BI (compute from ERC if missing) | mNDWS static | Move to dynamic |
| `ndvi` | VIIRS VNP13A1 (8‑day) | mNDWS static | Move to dynamic |
| **static** | | | |
| `elev` | SRTM | mNDWS elev | Same |
| `slope`, `aspect_cos/sin` | derived from elev | derived | Same |
| `pdsi` | GridMET PDSI monthly | mNDWS pdsi | Kept static |
| `chili` | | mNDWS chili | Kept |
| `water` | | mNDWS water (NLCD class 11) | Kept |
| `fuel1`, `fuel2` | derived from MODIS LCT yearly | LANDFIRE fuel PCs | Reduce to top 2 PCs |

mNDWS gets a `0` column for the two VIIRS thermal channels. That's correct
behavior — mNDWS doesn't have thermal, and the model needs to learn to use
thermal when present and not‑rely‑on it when absent (training with mNDWS
samples that have thermal=0 is exactly how to teach this).

**Acceptance criteria for phase 1:**

- `ignis_ml/scripts/ingest_ts_satfire.py --dry-run --max-fires 3` produces 3
  fires' worth of tiles in <10 minutes.
- Smoke test: `python -m pytest ignis_ml/tests/test_ingest_tssatfire.py`
  verifies channel ordering, shape, no NaN/Inf.
- A spot‑check Palisades‑adjacent fire (Thomas 2017, Woolsey 2018) shows
  mean `u` < −5 m/s for at least one day during the fire's peak — i.e. the
  ingestion is preserving the Santa Ana signal.

### Phase 2 — Feature engineering changes (~1 day)

**Deliverable:** edits to `ignis_ml/config.yaml`, `ignis_ml/src/data/dataset.py`,
`ignis_ml/src/data/features.py`, and `services/tilesvc/model_config.default.json`.

1. **Config** (`ignis_ml/config.yaml`):
   - Update `channels.dynamic_order` and `channels.static_order` per §3.
   - Add `viirs_i4` and `viirs_i5` to `_dyn_ranges` in `dataset.py` with
     ranges `(280, 380)` K (clip outliers, normalize to [0,1]).
   - Reduce `channel_dropout_p: 0.15` → `0.05` and pass an `exclude` list
     `[fire_t, u, v, viirs_i4]` so wind and thermal are never zeroed.
2. **Augmentation**:
   - Keep 90° rotation + wind‑vector rotation (it's already correct).
   - Drop `gaussian_noise_std` to 0.01 (was 0.02 — too noisy on thermal).
   - Keep `temporal_dropout_p: 0.10`, exclude last frame.
3. **Derived features** (`features.py`):
   - Keep the 6 derived features as‑is. They're correct.
   - Cap `days_since_fire_cap: 7` (currently `null`) so the channel
     saturates at 7 days. Past that, the model treats pixels as "fully
     reburnable" again, which matches fire ecology.
4. **Serving config** (`services/tilesvc/model_config.default.json`):
   - Bump `arch_version` to `v4`, `cd` to 18, `cs` to 9.
   - Update `dynamic_order` and `static_order` to match v4.
   - Update `STATIC_NORMALIZATION` in `services/tilesvc/static_catalog.py`
     to drop impervious/population/fuel3 entries.

**Acceptance criteria:**

- `pytest ignis_ml/tests/` passes (existing tests + 1 new test that exercises
  the v4 channel schema end‑to‑end on a synthetic tile).
- `python -m tools.audit_alignment --preset palisades` still passes (the audit
  doesn't care about channel count, only shape/CRS).

### Phase 3 — Training run (~24–48 h GPU time)

**Deliverable:** `models/convlstm_unet_v4_delta_Cd15_Cs10_H64_T6.pt` plus a
TensorBoard / W&B log.

1. **Sampler change** (`train_nautilus.py::build_train_sampler`):
   Add a Santa‑Ana boost. Compute `mean_u` over each fire's lifecycle (from
   the metadata JSON we wrote in Phase 1). Mark fires with mean_u < −5 m/s
   as "santa_ana". Up‑weight their sampling weight by 5× so they appear in
   ~10% of training batches even though they're <2% of the data.
2. **Loss change**: Keep Dice + BCE + Tversky (alpha=0.3, beta=0.7) — the
   recipe is sound. Add a *wind‑aligned regularizer*: predicted delta should
   have higher density downwind than upwind. Implementation sketch:
   ```python
   def wind_align_loss(prob, u_mean, v_mean):
       # prob: [B,1,H,W], u_mean/v_mean: [B] mean wind over the sequence
       # Compute centroid of prob; should be downwind of fire_t centroid
       wind_vec = torch.stack([u_mean, v_mean], dim=-1)
       delta_centroid = prob_centroid - fire_t_centroid    # [B, 2]
       cosine = F.cosine_similarity(delta_centroid, wind_vec, dim=-1)
       return (1.0 - cosine).mean()   # penalize misalignment
   ```
   Weight: 0.1 (small — it's a regularizer, not the main signal).
3. **Threshold sweep**: 0.05–0.50 (was 0.30–0.95). For delta predictions the
   right operating threshold is much lower than for full‑mask predictions.
4. **Calibration**: After training, generate an isotonic calibration JSON
   from the validation split's (raw_prob, observed_burn) pairs. Save to
   `models/calibration_v4.json` with `model_sha256` matching the new
   checkpoint. tilesvc already supports this via `CALIBRATION_PATH`.

**Acceptance criteria:**

- Val CSI ≥ 0.20 on the *out‑of‑distribution holdout* (see Phase 5).
  Don't worry if in‑distribution val CSI drops vs v3 — that's expected
  when the eval set gets harder. CSI ≥ 0.20 on Santa Ana fires is the win.
- Wind‑align loss term reaches < 0.3 by epoch 15 (i.e. the model learns to
  predict in the direction of wind).

### Phase 4 — Mini‑pipeline notebook (~2 days, the new key asset)

**Deliverable:** `ignis_ml/notebooks/02_palisades_mini_pipeline.ipynb`

The notebook is a faithful mirror of the live tilesvc serving path, so you
can iterate on features/model/calibration without redeploying Render. Every
cell loads the corresponding module from `services/tilesvc/` so the notebook
and prod share code, not just behavior.

**Cell structure:**

1. **Config** — pick event (Palisades/Eaton/Camp/Dixie/Caldor), ref_time,
   Tseq=6, steps=6, model checkpoint path.
2. **Tile geometry** — `from services.tilesvc.grid import lonlat_to_tile,
   tile_affine, build_grid`. Render tile bounds on a basemap. Sanity panel.
3. **Static tensor** — `from services.tilesvc.static_catalog import
   load_static_tensor_for_model`. Print per‑channel min/mean/max.
   Visualize all 9 channels as a 3×3 grid.
4. **Dynamic FIRMS rasterization** — `from services.tilesvc.dynamic_builder
   import build_dynamic_for_tile`. Inject ignition if needed. Visualize
   fire_t for each of the 6 history frames.
5. **HRRR weather** — load from `.cache/runtime_cache/{event}/noaa_grid_cache`.
   Plot u, v as quiver. Make sure wind direction *looks* right for the event.
6. **Derived features** — `append_derived_features(...)` from
   `ignis_ml.src.data.features`. Sanity‑check that `wind_dir_cos/sin`
   recover the same direction as the quiver.
7. **Model inference** — load the v4 checkpoint. Run one forward pass.
8. **Multistep rollout** — replicate `services/tilesvc/app.py::_rollout_
   multistep_predictions`. Returns 6 frames of predicted prob.
9. **Calibration** — `calibrate_probability(...)`. Compare raw vs calibrated.
10. **Observed ground truth** — fetch CAL FIRE WFIGS perimeter or
    FIRIS daily perimeter for the event. Rasterize to the same tile grid.
11. **Metrics** — per‑step IoU, Dice, CSI, and *Hausdorff distance* between
    predicted boundary and observed boundary (so direction errors get
    surfaced even when area is right).
12. **Visualization** — for each step: satellite basemap + predicted
    heatmap + observed perimeter (cyan outline) + wind quiver overlay.
    This is the headline figure you'll re‑run after every retrain.
13. **Per‑channel ablation** — zero out each dynamic channel one at a time,
    re‑run inference, plot the change in predicted prob. Surfaces which
    channels the model is actually using. Catches a recurrence of the
    "wind ignored" failure mode.
14. **Save eval summary** — write `eval_summary.json` with the event,
    checkpoint sha, metrics per step, into `models/eval/` so you have a
    history of how each retrain performs against each preset.

**Acceptance criteria:**

- Running cells 1–14 on the Palisades preset with the *v3* checkpoint
  reproduces the bad heatmap you've been seeing in the live app (so we
  know the notebook is a faithful mirror).
- Running the same cells with the *v4* checkpoint shows a heatmap that
  visibly pushes SW and a higher CSI vs the observed perimeter.

### Phase 5 — Out‑of‑distribution evaluation harness (~1 day)

**Deliverable:** `ignis_ml/scripts/eval_historical.py` + per‑event ground
truth perimeters stored under `data/perimeters/`.

For each of the 5 historical presets (Palisades, Eaton, Camp, Dixie, Caldor),
fetch the final official perimeter from CAL FIRE FRAP or NIFC WFIGS, and
the daily progression perimeters where available. Store as GeoJSON in EPSG:4326.

The script loops every preset through the mini‑pipeline and writes a CSV
of `event, step, ckpt_sha, iou, dice, csi, hausdorff_km` for every
combination. This is the *real* validation set — out of training distribution.

**Acceptance criteria:**

- v4 mean IoU across 5 presets ≥ 0.15 (advisory threshold).
- v4 mean Hausdorff distance ≤ 5 km on day 1, ≤ 8 km on day 2.

### Phase 6 — Deployment (~½ day)

1. Upload `convlstm_unet_v4_delta_Cd15_Cs10_H64_T6.pt` to GitHub Releases
   (v2.0.0). Compute SHA256.
2. Upload `calibration_v4.json` to S3 static bucket (or commit to repo).
3. On Render `ignisai-tilesvc` flip these env vars:
   - `MODEL_PATH` → `/app/models/convlstm_unet_v4_delta_Cd15_Cs10_H64_T6.pt`
   - `MODEL_URL` → new GitHub Releases asset URL
   - `MODEL_SHA256` → new sha
   - `MODEL_CD` → `18`
   - `MODEL_CS` → `9`
   - `MODEL_DYNAMIC_ORDER` → updated
   - `MODEL_STATIC_ORDER` → updated
   - `REQUIRED_ARCH_VERSION` → `v4`
   - `CALIBRATION_PATH` → `/app/config/calibration_v4.json`
   - `CALIBRATION_REQUIRED` → `1`
4. Rebuild Docker image so it bakes in the new `static_catalog.production.json`
   (rebuilt to drop impervious/population columns).
5. Wait for redeploy. Hit `/healthz` and verify:
   - `runtime_arch_version: "v4"`
   - `Cd: 18`, `Cs: 9`
   - `weatherQuality.source: "noaa_hrrr"`
   - `calibration.ok: true`
6. Load Palisades preset, run the same audit the notebook ran locally.
   Heatmap should look identical.

### Phase 7 — Guardrails (~½ day)

1. Wire the historical eval CSV into CI (`.github/workflows/`). PRs that
   change any model/dataset/feature code must produce a CSV that doesn't
   regress any preset's IoU by more than 0.02.
2. Add a `models/CHANGELOG.md` documenting v3 → v4 changes, eval numbers,
   known failure modes (e.g. "still under‑predicts magnitude on Santa Ana
   events because gust isn't gridded yet").

---

## 5. Open questions for you

1. **`ignis_ml_nautilus` folder**: I couldn't read it from this session
   (it isn't mounted). Is it just `train_nautilus.py` pointed at a Nautilus
   cluster volume, or does it have a different ETL? If it's the same code,
   no action needed. If it has a different ETL, drop the folder into the
   selected workspace folder so I can read it and fold those changes into
   the plan.
2. **Compute**: Do you have Nautilus access for training, or do you want me
   to spec a single A100 rental on Lambda/Vast/RunPod? Plan assumes A100.
3. **GridMET + VIIRS access for ingestion**: Pulling GridMET historical is
   free via the THREDDS server. Pulling VIIRS VNP14/VNP13A1 needs an
   Earthdata account (you already have one from FIRMS). I can write the
   pulls to use `earthaccess` or the FIRMS bucket on AWS Open Data.
4. **Kaggle credentials**: The TS‑SatFire dataset is on Kaggle. Easiest
   download path is `kaggle datasets download -d z789456sx/ts-satfire`
   with `~/.kaggle/kaggle.json` set up. Confirm you have a Kaggle account.
5. **Architecture**: Stay on ConvLSTM‑UNet for v4 (recommended), or jump to
   SwinUNETR‑3D? SwinUNETR is +0.05–0.10 F1 on next‑day spread per the
   TS‑SatFire paper, but it's a bigger lift (~1 week extra). My
   recommendation: ship v4 ConvLSTM, then evaluate whether the model arch
   is the bottleneck once data + features are right.

---

## 6. Effort and timeline

| Phase | Calendar effort | Compute |
| --- | --- | --- |
| 1 — Ingestion | 3 days | none |
| 2 — Feature/schema | 1 day | none |
| 3 — Training run | 1 day human + 24–48 h GPU | 1× A100 |
| 4 — Mini‑pipeline notebook | 2 days | none |
| 5 — Eval harness | 1 day | none |
| 6 — Deployment | 0.5 day | Render redeploy |
| 7 — Guardrails | 0.5 day | none |
| **Total** | **~9 days human + ~48 h GPU** | |

You can parallelize phases 4 and 5 with phase 3 (training runs in the
background while you build the notebook + eval harness).

---

## 7. What gets shipped

End state:

- `data/tssatfire_500m_T6/` — ~80–100 k re‑tiled samples.
- `models/convlstm_unet_v4_delta_Cd15_Cs10_H64_T6.pt` — new checkpoint.
- `models/calibration_v4.json` — isotonic calibration.
- `ignis_ml/notebooks/02_palisades_mini_pipeline.ipynb` — reusable eval harness.
- `ignis_ml/scripts/eval_historical.py` + `data/perimeters/*.geojson` —
  out‑of‑distribution validation.
- `services/tilesvc/static_catalog.production.json` — rebuilt without
  impervious/population.
- `docs/v4-retraining-gameplan.md` (this file).
- Render env updated; `weatherQuality.source: "noaa_hrrr"` and
  `runtime_arch_version: "v4"` visible on `/healthz`.

What this fixes:

- Palisades day‑1 prediction pushes **southwest** instead of east.
- Camp/Paradise prediction follows the actual east‑driven flow.
- Eaton prediction tracks the Altadena foothills.
- `weatherQuality` reports `noaa_hrrr` for every preset, not `open_meteo_fallback`.

What this does not fix (be honest):

- Predictions will still under‑estimate the *magnitude* of Palisades‑class
  spread on day 2 (14,500 acres in one day is at the extreme tail of any
  training distribution). The advisory framing in the README is still the
  right framing.
- Live‑mode (non‑preset) predictions for a brand‑new fire still depend on
  FIRMS NRT being current. If FIRMS has a gap, the prediction quality
  drops with it.

---

## 8. What I need from you to start Phase 1

- Confirm `ignis_ml_nautilus` either matches `train_nautilus.py` or mount
  the folder.
- Confirm compute target (Nautilus vs rented A100).
- Confirm Kaggle credentials available.
- A "go" on the channel schema in §3, or edits if you want different
  channels.

Once those four are answered I'll start writing the ingestion script and
the mini‑pipeline notebook in parallel.
