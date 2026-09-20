# IgnisAI × WildfireSpreadTS — parallel research track

**Status:** research only. Nothing here touches `services/tilesvc`, which keeps
serving the ConvLSTM v3/v4 checkpoint. This track earns promotion by beating the
production model on the five OOD presets — not before.

---

## The question

> Does the published state-of-the-art next-day wildfire spread model generalize
> to Santa Ana–driven, wildland–urban-interface fires?

Nobody has published this. WSTS/WSTS+ evaluate with 12-fold leave-one-year-out
CV over western-US fires 2016–2023, reporting AP ≈ 0.478 for the best model
(`Res18UTAE_T5`). But the benchmark is dominated by summer/fall vegetated-terrain
events. Palisades (Jan 2025) is a January coastal Santa Ana event with a heavy
WUI component — the exact regime `docs/v4-retraining-gameplan.md` §1 identifies
as IgnisAI v3's failure mode.

Either outcome is worth writing up:

- **SOTA fails on Santa Ana** → the gap is real and under-studied. That is a
  strong, evidence-backed justification for the wind-alignment regularizer and
  Santa-Ana-weighted sampling this repo already prototypes.
- **SOTA succeeds** → we have a 0.478-AP starting checkpoint, and fine-tuning
  from it dominates training ConvLSTM from scratch.

## Why parallel, not a migration

`services/tilesvc` is deployed and its grid (EPSG:5070, 500 m, 64×64) is baked
into the static catalog, the Render env, and the frontend. WSTS is 375 m,
128×128, dataset-native CRS, 23 bands, full-mask targets. Rewriting production
to chase a research result that has not yet beaten the incumbent is the wrong
order of operations. So: separate pipeline, shared evaluation, head-to-head on
the presets, promote only on evidence.

---

## What we need vs. what we have

Run `python src/wsts_spec.py` for the live report. Summary:

| | count | bands |
| --- | --- | --- |
| **have** | 16 | NDVI, precip, wind speed/dir, tmin/tmax, ERC, humidity, elevation, PDSI, all 5 HRRR forecast bands, active fire |
| **derive** | 4 | EVI2, slope, aspect, landcover (NLCD→MODIS IGBP crosswalk) |
| **missing** | 3 | VIIRS M11, I2, I1 — surface reflectance, via Earthdata |

**The whole gap is three VIIRS reflectance bands.** EVI2 derives from I1/I2, so
it unblocks with them. Everything else is either in S3 already or computable
from the SRTM DEM.

### Fidelity caveats that belong in the paper's limitations section

These are places where our reconstruction differs from native WSTS data. They
must be stated, not buried:

1. **NDVI / ERC / PDSI are fire-season composites in S3**, not per-day fields.
   WSTS supplies these per timestep. Using a static composite where the
   benchmark used a daily value is a genuine distribution shift, and plausibly
   depresses measured performance.
2. **Landcover crosswalk is lossy.** NLCD's classes do not map cleanly onto
   MODIS IGBP's 17. The crosswalk needs documenting and its ambiguities
   reporting.
3. **Active fire is a detection-*time* channel in WSTS** (hhmm → hh), not a
   binary mask. IgnisAI's `fire_t` is binary, so the time channel has to be
   rebuilt from raw FIRMS/VIIRS records.
4. **Angles must be degrees.** WSTS applies `sin(deg2rad(x))` to wind direction
   (7), aspect (13), and forecast wind direction (19). IgnisAI stores wind as
   `u,v` and aspect as `cos/sin` — both must be converted *back* to degrees or
   the model receives nonsense in three channels.
5. **Resampling 500 m → 375 m** upsamples; it cannot create detail the source
   rasters lack.

---

## Layout

```
research/wsts/
  README.md            this file
  src/
    wsts_spec.py       authoritative 23-band / 40-channel spec + source gap map
    presets.py         the 5 OOD events (shared with ignis_ml/scripts/eval_historical.py)
  notebooks/
    01_sota_ood_evaluation.ipynb    the paper artifact
```

## Getting the weights

```bash
pip install huggingface_hub
huggingface-cli download saadlahrichi/WSTSPlus \
  --include "trained_model_weights/Res18UTAE_T5/*" \
  --local-dir "$IGNIS_DATA_ROOT/pretrained"
```

Reference code (dataloader + model definitions) —
`git clone https://github.com/slahrichi/WildfireSpreadTS third_party/WildfireSpreadTS`

## Order of work

1. **Validate the harness.** Load `Res18UTAE_T5`, run it on real WSTS data
   (Zenodo), confirm ≈0.478 AP. Until this reproduces, no downstream number is
   trustworthy.
2. **Source the 3 VIIRS bands** for the five preset events + build the adapter.
3. **Score SOTA on the presets.** The actual experiment.
4. **Compare against IgnisAI v3/v4** on identical events and metrics.
5. Only then consider fine-tuning or promotion.

Step 1 is non-negotiable and is the cheapest. Do not skip to step 3.

## Sources

- Lahrichi, Bova, Johnson, Malof. *Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark.* WACV 2026. [arXiv:2502.12003](https://arxiv.org/abs/2502.12003) · [weights](https://huggingface.co/saadlahrichi/WSTSPlus) · [code](https://github.com/slahrichi/WildfireSpreadTS)
- Gerard, Zhao, Sullivan. *WildfireSpreadTS.* NeurIPS D&B 2023. [code](https://github.com/SebastianGer/WildfireSpreadTS) · [data](https://zenodo.org/records/8006177)
