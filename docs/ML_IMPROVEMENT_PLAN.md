# IgnisAI ML improvement plan

This document is the engineering plan for the next two model+product
iterations after the current ConvLSTM-UNet v3 (delta target, Tversky loss,
6-frame temporal window, 64-channel hidden state). It assumes the existing
training pipeline in `tilesvc/` and the tile/serve path through
`backend/routes/predictFireSpread.js`.

The work is grouped so that each section is independently shippable and
each carries its own success metric. Don't merge sections in flight — pick
one, finish its acceptance test, ship, then start the next.

---

## Where we are today

Current production stack:

- **Architecture:** ConvLSTM-UNet (v3) trained on Sentinel-2 RGB-NIR + DEM
  + wind + per-pixel FIRMS hotspot history. Delta-target formulation
  (predict change in burn probability vs. predict absolute probability)
  with Tversky loss biased toward recall (alpha=0.7).
- **Inputs at inference time:** static rasters from
  `IGNIS_STATIC_BUCKET`, NOAA grid points (currently disabled by default
  via `NOAA_GRIB_ENABLED=0`), recent FIRMS hotspots (last 24h).
- **Outputs:** per-pixel burn probability raster, smoothed by Gaussian
  (`PRED_SMOOTH_SIGMA=0.5`), upscaled 8×, displayed above
  `PRED_DISPLAY_FLOOR=0.10`.
- **Validation:** Palisades scenario in `tools/palisades_validation.py`.
  No public benchmark numbers reported.

Known gaps that bound prediction quality:

1. No explicit fuel-moisture input. The model implicitly learns dryness
   from NIR + climatology, which works in the training distribution but
   degrades hard during atypical conditions (wet winter → dry spring).
2. Fire weather indices (ERC, BI, PDSI) are computed but treated as
   static — they enter the static catalog as a daily snapshot, not as a
   live channel. We're losing diurnal and synoptic signal.
3. Output is a single deterministic raster. Decision-makers asked
   repeatedly (see `docs/results.md`) for *uncertainty bands* — they
   want to see the 50/80/95% spread cone, not just the 50%.
4. No published evaluation against an external benchmark. WSTS+ exists,
   has a held-out CONUS-wide eval set, and would let us claim a real
   number.
5. Calibration is loosely controlled (`CALIBRATION_REQUIRED=0`). When
   we finally turn it on, we'll need a proper isotonic-regression step.

The plan below addresses these in priority order.

---

## Phase 1 — Live fuel moisture content (LFMC)

**Why first:** highest expected lift per engineering hour, no architecture
change required. Fuel moisture is the single most predictive variable for
ignition probability and rate of spread; in the Palisades validation run,
the worst false negatives all clustered on slopes where modeled LFMC
diverged from the chaparral phenology curve.

### 1.1 Data source

Ingest **NOAA's GOES-derived LFMC product** (10-day rolling composite,
500m resolution, CONUS) plus, where available, the **National Fuel
Moisture Database** point observations (sparse but high-quality ground
truth) for bias correction.

Fallback chain:

1. NOAA gridded LFMC product (primary).
2. MODIS/VIIRS NDWI-derived LFMC if NOAA latency exceeds 36h.
3. Climatological LFMC by ecoregion + day-of-year (last-resort static).

### 1.2 Pipeline change

Add `lfmc` as a **runtime channel** (not a static one) in the tilesvc
ingest:

- New module `tilesvc/ingest/lfmc.py`:
  - `def fetch_lfmc(bbox, target_dt) -> np.ndarray` returns a
    [H, W] float array in fraction units (0.0–1.5; >1.0 is allowed and
    common in early spring).
  - Cache to `IGNIS_RUNTIME_CACHE_BUCKET` keyed by `(bbox, date)`. TTL
    10 days (matches NOAA composite cadence).
  - Emit metric `tilesvc_lfmc_fetch_seconds` and
    `tilesvc_lfmc_fallback_tier` (0/1/2/3) per fetch so we can alert if
    we silently degrade to climatology in production.

- Model input change: extend the channel stack from C=13 (current) to
  C=14. Update `tilesvc/model/convlstm_unet.py` `__init__` signature
  and bump model version to v4.

### 1.3 Training

- Re-train v3 → v4 on the same dataset windowed to `[t-6, t-1]` with the
  new LFMC channel back-filled from NOAA archive.
- Same loss (Tversky α=0.7), same optimizer (AdamW 3e-4), same scheduler
  (cosine to 1e-5 over 80 epochs).
- Hold the existing val/test split byte-identical so we can A/B v3 vs.
  v4 on the same scenes.

### 1.4 Acceptance metric

Block release of v4 unless **both** hold on the held-out test set:

- Recall@FAR=0.05 ≥ v3 + 3.0 absolute points.
- AUPRC ≥ v3 + 0.02.

Plus on the Palisades validation: at least 80% of v3's missed slopes
recovered (defined by the `palisades_missed_slopes.geojson` fixture
to be added in this phase).

### 1.5 Engineering checklist

- [ ] `tilesvc/ingest/lfmc.py` with three-tier fallback + tests
- [ ] `tilesvc/config/static_catalog.production.json` updated with
      LFMC manifest pointer
- [ ] Model retrained, checkpoint uploaded, `MODEL_PATH` and
      `MODEL_SHA256` rotated in `render.yaml`
- [ ] `palisades_validation.py` extended with the missed-slopes fixture
- [ ] Dashboard: add LFMC layer toggle (cyan→brown ramp, low values
      highlighted)

---

## Phase 2 — Live fire-weather indices (ERC / BI / PDSI)

**Why second:** the variables exist, the static catalog already carries
them, but they're frozen at midnight UTC. Promoting them to dynamic
channels gives us a free intra-day signal at the cost of one extra fetch
per inference.

### 2.1 What to promote

| Index | Definition | Cadence we want |
|---|---|---|
| **ERC** (Energy Release Component) | NFDRS index of total available combustion energy | Hourly |
| **BI** (Burning Index) | NFDRS rate-of-spread × intensity proxy | Hourly |
| **PDSI** (Palmer Drought Severity Index) | Multi-month drought | Daily (already daily; just needs to be live, not snapshotted weekly) |

### 2.2 Source

- ERC + BI: WIMS / FEMS (Fire Environment Mapping System) hourly
  rasters. If FEMS is rate-limited, derive locally from the NOAA grid
  RH/temp/wind we already pull.
- PDSI: NOAA CPC weekly, downsampled to our grid.

### 2.3 Pipeline change

Add a `weather_dynamic` ingest module (`tilesvc/ingest/wx_dynamic.py`)
that returns a [3, H, W] stack at the target hour:

```python
def fetch_dynamic_wx(bbox: BBox, target_dt: datetime) -> WxStack:
    erc = fetch_erc_hourly(bbox, target_dt)   # NFDRS, normalized 0..100
    bi  = fetch_bi_hourly(bbox, target_dt)
    pdsi = fetch_pdsi(bbox, target_dt)        # -10..+10
    return np.stack([erc, bi, pdsi], axis=0)
```

Cache hourly slices in `IGNIS_RUNTIME_CACHE_BUCKET` with the same key
scheme as LFMC. Failure to fetch any of the three should fall back to
the static catalog value with a one-time per-process warning, not a
hard failure (we don't want a FEMS outage to take down the predict
endpoint).

### 2.4 Model change

Channels: 14 → 17. Same retraining recipe as Phase 1.

### 2.5 Acceptance metric

Block release unless on the held-out test set:

- AUPRC ≥ v4 + 0.01.
- **Reliability under windy conditions** (wind ≥ 25 mph subset):
  Brier score improvement ≥ 0.005 absolute. This is the diagnostic that
  catches the 'looks fine on average, awful in red-flag conditions' case.

### 2.6 Engineering checklist

- [ ] `tilesvc/ingest/wx_dynamic.py` + tests with mocked FEMS fixtures
- [ ] `CALIBRATION_REQUIRED=1` in `render.yaml` and isotonic regression
      step in eval (see Phase 4)
- [ ] Update `docs/static-pipeline.md` to clarify which fields are now
      dynamic vs. static (catalog stays for fallback only)
- [ ] Dashboard tooltip: when hovering a high-probability cell, show
      ERC/BI/PDSI values that contributed (debug only behind a feature
      flag for now)

---

## Phase 3 — Probability cone (uncertainty visualization)

**Why third:** unlocks better decisions even without a more accurate
model. Operators consistently ask "how sure are you?". Today we ship a
deterministic raster and bury that question.

### 3.1 Approach

We have two viable options. Pick (b) for v5, hold (a) in reserve.

(a) **Test-time augmentation (TTA) ensemble.** Run inference 8× with
    rotations and small noise on wind direction (±10°) and ignition
    point. Cheap (8× compute), well-understood, no training change.

(b) **MC-dropout in the ConvLSTM bottleneck + decoder.** Add dropout
    p=0.2 at training time, keep it on at inference, run 32 forward
    passes, take per-pixel mean and quantiles.

We've already trained with weight decay; adding dropout requires
retraining but yields a better-calibrated distribution than TTA.

### 3.2 Output schema

`/api/predict/spread` should return, in addition to the current `mean`
raster, three quantile rasters:

```jsonc
{
  "mean": "<COG URL>",
  "p50":  "<COG URL>",
  "p80":  "<COG URL>",
  "p95":  "<COG URL>",
  "method": "mc_dropout_n32",
  "ensemble_size": 32
}
```

Backend route stays in `backend/routes/predictFireSpread.js`. The
frontend renders the cone as nested isolines; default visible band is
p50, with p80 toggleable and p95 only on the legend hover.

### 3.3 Acceptance metric

Calibration on the held-out set:

- The fraction of cells whose actual outcome falls inside the
  predicted [p10, p90] interval should be 0.80 ± 0.03.
- The Continuous Ranked Probability Score (CRPS) should be < 0.18.

### 3.4 Engineering checklist

- [ ] `tilesvc/model/mc_dropout_head.py` with toggleable inference-time
      dropout
- [ ] Predict route returns the 4 raster URLs; legacy single-raster
      response stays under `?legacy=1` for two releases
- [ ] Frontend: cone overlay component, with a hard rule to never paint
      p95 below display floor (visual noise)
- [ ] Field-tester one-page guide: how to read the cone (added to
      `docs/runbook.md`)

---

## Phase 4 — Calibration as a first-class step

Right now `CALIBRATION_REQUIRED=0`. The implicit assumption is that the
sigmoid output is a probability. It's not — it's a score that happens
to lie in [0, 1]. We need to fit isotonic regression on a held-out
calibration split and ship the calibration table as part of the model
artifact.

### 4.1 Steps

1. Hold out 5% of the existing training set as a *calibration* split
   (separate from val/test).
2. Train v4 / v5 as before.
3. Predict on the calibration split, fit `sklearn.isotonic.IsotonicRegression`.
4. Save the fitted calibrator alongside the `.pt` file as
   `<model>.calib.json` (just a list of (raw, calibrated) pairs).
5. At serve time, apply the calibrator before the display floor and
   before any probability cone quantiles.

### 4.2 Acceptance metric

Reliability diagram with 10 bins should have all bins within ±0.05 of
the diagonal on the held-out test set. Expected Calibration Error
(ECE) < 0.03.

### 4.3 Engineering checklist

- [ ] Add `tilesvc/model/calibration.py` with fit + predict
- [ ] Bake calibration into the model artifact build step
- [ ] Set `CALIBRATION_REQUIRED=1` once Phase 2 ships
- [ ] Reliability diagram added to the eval notebook in `docs/results.md`

---

## Phase 5 — External benchmark (WSTS+)

**Why last:** every previous phase changes the model; locking in a
WSTS+ number now means re-running it after each phase. So we wire up
the benchmark *infrastructure* now and run it once at the end of
Phases 1+2+3+4.

### 5.1 What to claim

Report on the public WSTS+ test set:

- AUPRC overall and by ecoregion.
- Recall@FAR=0.05.
- Brier score, CRPS.
- Wall-clock inference time for 1024×1024 tile on the deployed
  Render `starter` plan (this is honest about cost, which other
  benchmarks gloss over).

### 5.2 Engineering checklist

- [ ] `tilesvc/eval/wsts_plus.py` runner that consumes the public
      WSTS+ dataset format and emits a JSON results card
- [ ] CI job (manual-trigger only, the dataset is large) that runs the
      runner against a pinned model artifact
- [ ] Results card committed to `docs/results.md` per release

---

## Stretch (post-v5)

- **Multi-fire interaction.** When two fires are within ~5km, treat
  the wind shadow of one as input to the other. Today we predict
  independently and the seams show.
- **Spotting model.** ConvLSTM predicts contiguous spread well but
  cannot model long-range ember spotting. Train a small Poisson-process
  side-model on historical spotting events (CALFIRE incident data) and
  composite its output with the main raster.
- **Suppression-aware predictions.** Add a 'suppression effort' channel
  derived from CALFIRE / NIFC engine and aircraft positions. Today the
  model assumes nobody is fighting the fire, which biases predictions
  toward over-spread in well-resourced incidents.

---

## Tracking

Each phase corresponds to a milestone tag (`ml-phase-1`, `ml-phase-2`,
…). Open an issue per checklist item under that milestone. Don't open
the next milestone until the previous one's acceptance metrics are
recorded in `docs/results.md`.

The order above is deliberate. Skipping ahead — e.g. doing the
probability cone before LFMC — produces a beautifully calibrated
visualization of a model that's still missing the most predictive
input it could have. That's worse than no cone at all.
