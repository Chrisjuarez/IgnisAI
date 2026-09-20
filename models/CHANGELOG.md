# Model Changelog

## v4 (in progress) — Santa-Ana-aware retrain

Goal: produce a tight, directional fire-spread heatmap for Santa-Ana / WUI
events (Palisades-class), where v3 fails. See `docs/v4-retraining-gameplan.md`
and `docs/heatmap-diagnosis.md`.

Changes vs v3:

- **Data**: add TS-SatFire (2017–2021, includes Santa-Ana coastal fires) merged
  with mNDWS. Ingestion: `ignis_ml/scripts/ingest_ts_satfire.py`.
- **Channels**: dynamic 12 raw + 6 derived = 18 (add VIIRS I4/I5 thermal; move
  erc/bi/ndvi to dynamic). static 9 (drop `impervious`, `population`, `fuel3` —
  the urban shortcut features that mislocated v3).
- **Loss**: actually apply Tversky (alpha=0.3, beta=0.7) — it was configured but
  never wired into v3's loss. Add a wind-alignment regularizer (weight 0.1).
- **Augmentation**: protect `u/v/gust/viirs_i4` from channel dropout (v3 zeroed
  wind ~15% of the time); lower dropout 0.15→0.05.
- **Sampling**: up-weight Santa-Ana fires 5× (`v4_sampler.build_santa_ana_sampler`).
- **Threshold**: sweep 0.05–0.50 (delta operating point), add isotonic
  calibration JSON.
- **arch_version**: bump `v3` → `v4`.

Expected: Palisades day-1 pushes SW (not urban-east); CSI on Santa-Ana holdout
≥ 0.20; mean Hausdorff ≤ 5 km day 1.

Known remaining gap: day-2 magnitude under-estimate on extreme wind events.

## v3 — delta + derived features (deployed)

`convlstm_unet_v3_delta_Cd13_Cs15_H64_T6_nautilus.pt`. ConvLSTM-UNet, delta
target, 6 derived features. Trained on mNDWS only. CSI ≈ 0.235. Out of
distribution for coastal Santa-Ana fires; under-weights wind (training dropout)
and over-uses urban static features.
