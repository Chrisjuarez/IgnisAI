"""Build training tiles from the same pipeline that serves predictions.

The NDWS corpus cannot support what this model is asked to do. Its weather is
identical across every frame of a sequence, so a recurrent model cannot learn
that wind drives spread over time; its fire history is synthetic, fabricated
from a single day; and it holds no multi-day sequences at all, so there is no
ground truth at t+2 for the same fire and rollout supervision is impossible.
tools/inspect_training_corpus.py measures all three.

This builds the corpus that is missing, out of the runtime caches the service
already uses. Each sample comes from build_dynamic_for_tile at a real date, so
the weather varies frame to frame because it actually varied, and the fire
mask evolves because it actually evolved. Samples from one fire are
consecutive days and record their fire and day index, so they chain - which is
what rollout supervision needs.

It also closes the train/serve gap by construction: the tiles are built by the
serving builder, on the serving grid, from the serving sources.

    python -m ignis_ml.scripts.etl_events --out data/events_500m_T3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("NOAA_GRIB_ENABLED", "1")

#: Channel order the existing checkpoints expect.
DYN_NAMES = ("fire_t", "u", "v", "gust", "tempC", "q", "precip")
STAT_NAMES = ("elev", "ndvi", "bi", "erc", "pdsi", "chili",
              "impervious", "water", "population", "fuel1", "fuel2", "fuel3")

#: Days of history per sample. Matches the deployed control60 checkpoint.
SEQ_LEN = 3

#: A sample needs SEQ_LEN frames of history and a target the day after, so a
#: window of N days yields N - SEQ_LEN samples.
CACHE_ROOT = _REPO / ".cache" / "runtime_cache"


def use_profile(profile: str) -> bool:
    base = CACHE_ROOT / profile
    firms, noaa = base / "firms_snapshots", base / "noaa_grid_cache"
    if not firms.is_dir() or not noaa.is_dir():
        return False
    os.environ["FIRMS_SNAPSHOT_DIR"] = str(firms)
    os.environ["FIRMS_SNAPSHOT_REQUIRED"] = "1"
    os.environ["NOAA_GRID_CACHE_DIR"] = str(noaa)
    return True


def build_event_samples(profile: str, lat: float, lon: float, ref_iso: str,
                        window: int, out_dir: Path) -> List[Dict[str, Any]]:
    """One builder call per fire, sliced into consecutive chained samples."""
    from services.tilesvc.dynamic_builder import build_dynamic_for_tile
    from services.tilesvc.grid import lonlat_to_tile
    from services.tilesvc.static_catalog import load_static_tensor_for_model

    if not use_profile(profile):
        return []

    ref = dt.datetime.fromisoformat(ref_iso.replace("Z", "+00:00"))
    frames = np.asarray(build_dynamic_for_tile(
        lat, lon, T_seq=window, hours_step=24, ignition=False,
        ref_time=ref, channel_order=list(DYN_NAMES)), dtype=np.float32)

    tile = lonlat_to_tile(lon, lat)
    stat_raw, stat_summary = load_static_tensor_for_model(tile, list(STAT_NAMES))
    stat = np.asarray(stat_raw, dtype=np.float32)

    # A failed static read does not raise - it returns a placeholder tensor of
    # constants, one value per channel across all 4096 cells. Training on that
    # teaches the model that terrain and fuel are uniform everywhere, which is
    # worse than the corpus this is meant to replace. The giveaway is that
    # every channel has exactly one distinct value.
    flat = [n for i, n in enumerate(STAT_NAMES) if len(np.unique(stat[i])) <= 1]
    if len(flat) >= len(STAT_NAMES) // 2:
        raise RuntimeError(
            f"static tensor looks like a placeholder ({len(flat)} of {len(STAT_NAMES)} "
            f"channels constant: {', '.join(flat[:6])}). Set STATIC_CATALOG_PATH and "
            f"AWS credentials for the bucket the catalog points at."
        )

    written: List[Dict[str, Any]] = []
    # Sample i uses frames[i : i+SEQ_LEN] and targets frames[i+SEQ_LEN].
    for i in range(0, window - SEQ_LEN):
        x_dyn = frames[i:i + SEQ_LEN]
        y = (frames[i + SEQ_LEN, 0] > 0.5).astype(np.float32)
        prior = x_dyn[-1, 0] > 0.5
        # A sample with no fire to grow from, or no growth to predict, teaches
        # nothing and would dominate a corpus this size.
        if not prior.any() or not (y > 0.5)[~prior].any():
            continue
        path = out_dir / f"{profile}_t{i:03d}.npz"
        np.savez_compressed(
            path, x_dyn=x_dyn, x_stat=stat, y=y,
            dyn_names=np.array(DYN_NAMES), stat_names=np.array(STAT_NAMES),
            fire_id=np.array(profile), day_index=np.array(i),
        )
        written.append({"path": path.name, "fire_id": profile, "day_index": i,
                        "prior_cells": int(prior.sum()), "new_cells": int((y > 0.5).sum() - prior.sum())})
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ignis_ml.scripts.etl_events")
    ap.add_argument("--events", type=Path, required=True,
                    help="JSON of [[profile, lat, lon, iso_ref_time], ...]")
    ap.add_argument("--out", type=Path, default=_REPO / "data" / "events_500m_T3")
    ap.add_argument("--window", type=int, default=7,
                    help="Days of cache per fire; yields window - SEQ_LEN samples")
    args = ap.parse_args(argv)

    events = json.loads(args.events.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for row in events:
        profile, lat, lon, iso = row[0], float(row[1]), float(row[2]), row[3]
        try:
            rows = build_event_samples(profile, lat, lon, iso, args.window, args.out)
        except Exception as exc:  # noqa: BLE001 - one bad fire is not fatal
            print("  %-24s failed: %s: %s" % (profile, type(exc).__name__, str(exc)[:60]))
            continue
        print("  %-24s %d samples" % (profile, len(rows)))
        all_rows.extend(rows)

    manifest = {
        "source": "runtime_cache events (serving pipeline)",
        "dyn_order": list(DYN_NAMES),
        "stat_order": list(STAT_NAMES),
        "seq_len": SEQ_LEN,
        "window_days": args.window,
        "tiles": len(all_rows),
        "fires": len({r["fire_id"] for r in all_rows}),
        "samples": all_rows,
        "note": ("Weather varies frame to frame because it did; fire evolves because "
                 "it did. Samples from one fire are consecutive days and carry "
                 "fire_id/day_index, so they chain for rollout supervision."),
    }
    (args.out / "_etl_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n  %d samples from %d fires -> %s" % (len(all_rows), manifest["fires"], args.out))
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
