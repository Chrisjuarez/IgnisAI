# IgnisAI — Fire-Spread Heatmap Diagnosis (June 2026)

Root-cause analysis of why the fire-spread heatmap (live and historical) is
inaccurate, based on a live walkthrough of the deployed app plus a code audit of
`ignis_ml/`, `services/tilesvc/`, and `frontend/`.

This complements `docs/v4-retraining-gameplan.md` (which is correct and remains
the plan). This file is the *evidence* behind that plan plus two issues the
gameplan did not call out.

---

## TL;DR

The heatmap is wrong for **three independent reasons**, in priority order:

1. **The model (v3) is the core problem — needs retraining.** Trained on mNDWS
   (next-day, 500 m, 2012–2020 summer/fall vegetated-mountain CONUS fires), it is
   out-of-distribution for a January coastal Santa-Ana WUI fire like Palisades.
   Two training choices actively suppress the signal that drives such fires
   (wind dropout) and inject a spurious one (urban shortcut features). Reported
   quality is CSI ≈ 0.235 — weak.
2. **The multistep rollout amplifies the weakness (fixable without retraining).**
   The autoregressive feedback is a monotonic `max()` ratchet that only adds
   probability mass and never lets it decay, so the field diffuses and confidence
   decays across the 6-day timeline. It also re-fetches weather *per step*, which
   is why multistep times out (>180 s) while single-step raster returns in ~18 s.
3. **The renderer makes it look mushy (fully fixable now).** The heatmap PNG is
   2× upscaled with image smoothing (the soft-blob blur), alpha ramps from a 0.3
   floor (so trivial probabilities still show), and the crisp contour outline
   layers are never populated for the timeline.

The tight, *accurate*, directional Palisades spread requires fixing #1. #2 and #3
make the current output cleaner and faster but cannot manufacture spread the
model didn't predict.

---

## Evidence from the live app (Palisades historical preset)

Ran History → Historical Fire Testing → "Palisades Fire (2025-01-07 10:30 PT)".
The forecast returned 6 frames. Per-frame numbers off the timeline panel:

| Frame (day) | Peak advisory prob | Above 95% decision thr | Visible cells (floor 10%) | Next-fire area |
| --- | --- | --- | --- | --- |
| 1 | 90.5% | **0.0%** | 5.5% | 0.9% |
| 2 | 82.7% | **0.0%** | 12.5% | 0.9% |
| 3 | 72.9% | **0.0%** | 11.6% | 0.9% |
| 6 | 79.3% | **0.0%** | 4.8% | 0.9% |

Observations:

- **Never crosses its own decision threshold.** Peak ≈ 90% < 95% threshold, so
  "above decision threshold" is 0.0% on *every* frame. The displayed heatmap is
  entirely sub-threshold mass shown only because `display_floor = 0.1`.
- **Confidence decays and area diffuses** over the timeline (90.5 → 82.7 → 72.9%),
  the opposite of the real fire, which *accelerated* (14,500 acres on Jan 8).
- **Mislocated.** The prediction crop window sits around San Vicente Mountain;
  the real catastrophic spread went SW/S to the Pacific Palisades neighborhoods
  and coast — outside the predicted footprint — then north.

Compared against the NASA/observed Jan 7–12 progression, the app matches on
**none** of: footprint, direction, magnitude, or temporal acceleration.

### Infra notes seen live

- `GET /api/predict-fire-spread/multistep` hangs and the frontend logs
  `safety-net timeout fired … Forecast timed out after 180s` (it retries 5×).
- `/health` returns `DEGRADED`, `tilesvc: disconnected` — but tilesvc is actually
  reachable server-side (single-step `/raster` returned 200 in ~18 s). The
  `disconnected` flag comes from a too-aggressive 2.5 s `healthz` probe against a
  CPU service plus `TILE_SVC` env fragility (defaults to `http://localhost:8008`
  if unset; render.yaml marks it `sync:false`).

---

## Code-level root causes

### 1. Model / training (`ignis_ml/`)

- **Dataset mismatch.** `config.yaml` → `datasets.mndws` (Next-Day Wildfire
  Spread). No Santa-Ana coastal fires in distribution.
- **Wind is dropped during training.** `augmentation.channel_dropout_p: 0.15`
  zeroes non-fire dynamic channels — including `u`, `v`, `gust` — ~15% of the
  time (`dataset.py::_augment`). The model learns wind is optional and
  under-weights it at inference. For Santa-Ana fires wind is *the* driver.
- **Urban shortcut features.** Static `impervious` and `population` light up on
  California WUI tiles and the model uses them as a shortcut, pulling predictions
  toward the urban edge / I-405 band regardless of wind/terrain.
- **Tversky is configured but never applied.** `config.yaml` enables
  `loss.tversky` (alpha=0.3, beta=0.7), but the training loss
  `train_nautilus.py::loss_bce_dice_focal` only implements BCE + Dice + optional
  focal. **Tversky is silently a no-op.** This matters: Tversky is the recipe
  meant to counter the sparse-positive imbalance, and it isn't running. (Not
  noted in the v4 gameplan — fix in v4.)
- **Decision threshold vs. output distribution.** Serving threshold is 0.95 but
  delta-mode outputs peak ~0.9; the right operating threshold for delta is much
  lower (see gameplan Phase 3: sweep 0.05–0.50). Without calibration
  (`CALIBRATION_PATH` unset → identity), raw probabilities are shown.

### 2. Rollout (`services/tilesvc/app.py::_rollout_multistep_predictions`)

- **Monotonic AR ratchet.** Between steps:
  `next_fire = np.maximum(prev_fire, prob)` (soft mode). Probability mass can
  only grow; nothing decays or is suppressed. Over 6 steps this smears a
  low-confidence blob outward instead of advancing a coherent front.
- **Per-step weather fetch in the loop.** `fetch_weather_grids(...)` is called
  once per step (5 fetches for 6 steps) plus 6 CPU forward passes. On the
  free-tier CPU `tilesvc` this exceeds the 180 s budget → timeout. Single-step
  raster (one fetch) returns ~18 s. Fix: prefetch all lead-time weather grids
  concurrently, or make multistep an async job + poll.

### 3. Renderer (`frontend/src/utils/addPredictionOverlay.js`)

- **2× smoothing blur.** `colorizeGrayscalePngToHeatmapDataUrl(smooth=true)`
  upscales 2× with `imageSmoothingEnabled` → the diffuse soft-cloud look.
- **Alpha floor too low.** `a = (0.3 + v*0.7) * opacity` paints even faint cells.
- **Outlines unused on the timeline.** `renderPredictionRasterFrame` can draw
  `frame.contour` / `frame.contour_50` line layers, but
  `prepareMultistepRasterFrames` never attaches contour geometry to frames, so
  the timeline shows only the smoothed fill — no crisp NASA-style perimeter.

---

## What fixes what

| Symptom | Cause | Fix | Needs retrain? |
| --- | --- | --- | --- |
| Spread points wrong direction / mislocated | Wind dropout + urban shortcut features + OOD data | v4 retrain (TS-SatFire, protect wind, drop impervious/population, wind-align loss) | **Yes** |
| Under-confident; never crosses threshold | Uncalibrated delta output + 0.95 threshold | Isotonic calibration JSON + lower delta threshold | Partly (calibration can use current model) |
| Footprint diffuses / confidence decays over days | `np.maximum` AR ratchet | Replace ratchet with decay-aware feedback | No |
| Predictions time out | Per-step weather fetch | Concurrent prefetch / async job | No |
| Heatmap looks mushy | 2× smoothing + low alpha floor + no outlines | Renderer redesign (tight core, crisp day-banded perimeters) | No |
| Tversky not improving recall | Loss term not wired in | Implement Tversky in training loss | Yes (retrain) |

---

## Recommended order

1. **v4 retrain pipeline** (this is the only path to *accurate* spread). Scaffold
   shipped in `ignis_ml/scripts/`, `ignis_ml/src/training/`, `ignis_ml/notebooks/`,
   and `ignis_ml/config.v4.yaml`. See `docs/v4-implementation-README.md`.
2. **Calibration + threshold** (can start against current v3 to de-bias display).
3. **Rollout fix** (decay-aware feedback + concurrent weather prefetch).
4. **Renderer redesign** (tight, day-banded, NASA-like).

Items 2–4 are independent of retraining and can land immediately; item 1 needs a
GPU + the TS-SatFire dataset + Earthdata/Kaggle credentials.
