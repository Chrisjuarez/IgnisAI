"""v4 training extras: losses and samplers added for the Santa-Ana retrain.

Kept separate from train_nautilus.py so the v3 trainer stays byte-for-byte
reproducible. Wire these in behind a `--config config.v4.yaml` branch; see
docs/v4-implementation-README.md for the exact call sites.
"""
