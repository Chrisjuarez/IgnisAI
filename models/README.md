# models/

This directory holds the PyTorch checkpoints consumed by `ignisai-tilesvc`
at startup. Checkpoints are **not** committed to git (see `.gitignore`);
they are downloaded at deploy time. The `.gitkeep` file is here only so
the directory exists in fresh clones.

## How the tilesvc finds a model

`tilesvc` reads three env vars in priority order (see `tilesvc/app.py`
and the `ignisai-tilesvc` block in `render.yaml`):

| Var | Purpose |
|---|---|
| `MODEL_PATH` | Absolute path inside the container. If the file already exists, it's used as-is. |
| `MODEL_URL` | If `MODEL_PATH` is missing, download from this URL into `MODEL_PATH`. |
| `MODEL_SHA256` | Optional integrity check. Accepts bare hex or `sha256:<hex>`. If set, the download is rejected when the hash doesn't match. |

**Local development.** Drop the `.pt` file directly into this directory
and point `MODEL_PATH` at it:

```sh
# tilesvc/.env
MODEL_PATH=./models/convlstm_unet_v3_delta_Cd13_Cs15_H64_T6_nautilus.pt
```

**Production / Render.** Don't bake checkpoints into the Docker image (the
free tier image-size budget is too small for that). Instead:

1. Upload the `.pt` to a private S3 bucket or a signed-URL host.
2. Set `MODEL_URL` in the `ignisai-tilesvc` env vars.
3. Set `MODEL_SHA256` so a corrupt or swapped file fails the boot check
   instead of silently serving a wrong model.
4. Set `MODEL_PATH=/app/models/<filename>.pt` to control where it lands
   inside the container.

## Naming convention

Checkpoint filenames encode their training config so the deployed model
is auditable from the filename alone:

```
convlstm_unet_<vN>_<target>_Cd<delta_channels>_Cs<static_channels>_H<hidden>_T<temporal>_<env>.pt
```

Example: `convlstm_unet_v3_delta_Cd13_Cs15_H64_T6_nautilus.pt`

- `v3` — model architecture version
- `delta` — target formulation (delta vs. absolute burn probability)
- `Cd13` — 13 dynamic input channels
- `Cs15` — 15 static input channels
- `H64` — 64-dim ConvLSTM hidden state
- `T6` — 6-frame temporal window
- `nautilus` — training cluster tag (helps trace lineage)

When you train v4 with the LFMC channel (see `docs/ML_IMPROVEMENT_PLAN.md`
Phase 1), increment `Cd` and `vN` together: `convlstm_unet_v4_delta_Cd14_…`.
Never overwrite a published checkpoint in place — bump the filename and
re-point `MODEL_URL`.

## Rotating a checkpoint

1. Train and validate the new model. Capture metrics in `docs/results.md`.
2. Compute the SHA256 of the artifact:
   ```sh
   shasum -a 256 models/convlstm_unet_v4_delta_Cd14_…_nautilus.pt
   ```
3. Upload the artifact to S3 (or your hosting bucket).
4. Update `MODEL_URL`, `MODEL_PATH`, and `MODEL_SHA256` in the Render
   `ignisai-tilesvc` environment.
5. Manual deploy `ignisai-tilesvc`. Watch logs for
   `[model] verified sha256` and `[model] loaded`.
6. Smoke `/healthz` and a real `/predict` request before announcing.
7. Keep the previous `MODEL_URL` reachable for at least one week so a
   rollback is one env var change away.

## Files in this directory

- `.gitkeep` — keeps the directory present in clones; do not delete.
- `*.pt` — locally-downloaded checkpoints; never committed.
