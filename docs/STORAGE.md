# Moving IgnisAI's data to an external drive

You have ~15 GB free internally and the v4 retrain needs ~70 GB raw + ~100 GB of
processed tiles. **You do not need to move the project or rewire anything** — the
code already centralizes paths, and v4 adds one env var so the big files live on
the external drive while the code stays in `~/Downloads/IgnisAI`.

## Recommended: keep code where it is, put data on the drive

1. Plug in the 1 TB drive. On a Mac it mounts at `/Volumes/<DriveName>`.
2. Make a folder for the data:
   ```bash
   mkdir -p /Volumes/<DriveName>/ignis-data
   ```
3. Point IgnisAI at it (add to `~/.zshrc` so it sticks across terminals):
   ```bash
   export IGNIS_DATA_ROOT=/Volumes/<DriveName>/ignis-data
   ```
4. That's it. Everything that reads/writes large data now uses that path:
   - TS-SatFire download target: `$IGNIS_DATA_ROOT/tssatfire_raw/`
   - Processed tiles: `$IGNIS_DATA_ROOT/tssatfire_500m_T6/`
   - Trained checkpoints + calibration: `$IGNIS_DATA_ROOT/models/`
   - mNDWS tiles (if you also move them): `$IGNIS_DATA_ROOT/mNDWS_500m_T6/`

How it works: `ignis_ml/src/utils/paths.py` resolves a data root from
`IGNIS_DATA_ROOT` (falling back to `<repo>/data`), and any config path that
starts with `data/` is rebased onto it. The ingestion script, `train_v4.py`, and
`run_v4.sh` all use it. Set `IGNIS_MODELS_ROOT` too if you want checkpoints
somewhere other than `<data_root>/models`.

So your sequence is:
```bash
export IGNIS_DATA_ROOT=/Volumes/<DriveName>/ignis-data
kaggle datasets download -d z789456sx/ts-satfire \
  -p "$IGNIS_DATA_ROOT/tssatfire_raw" --unzip
./run_v4.sh
```

The internal disk only holds the small code repo; the 70–100 GB never touches it.

## Alternative: move the entire project to the drive

Also fine and needs no rewiring (all in-repo paths are relative):
```bash
cp -R ~/Downloads/IgnisAI /Volumes/<DriveName>/IgnisAI
```
Then re-open the project from its new location and run everything from there. Trade-offs vs. the env-var approach:
- The whole repo (and git history) lives on the external drive, so the drive must
  be connected to do anything, and git/IO go over USB.
- With the env-var approach the code stays on your fast internal SSD and only the
  bulk data is external — generally the nicer setup.

## Caveats

- **Keep the drive plugged in** during ingestion and training.
- **Filesystem:** APFS or Mac OS Extended is ideal. exFAT works for storing data
  but doesn't do POSIX symlinks/permissions well — fine here since we use real
  folders, not symlinks.
- **Speed:** USB-3/Thunderbolt is plenty for this workload; training reads tiles
  in DataLoader workers, not in a tight latency-bound loop.
- **Don't commit the data:** it lives outside the repo (or under `data/`, which is
  gitignored), so it won't bloat git.
