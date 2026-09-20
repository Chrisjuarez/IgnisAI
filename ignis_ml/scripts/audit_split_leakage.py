#!/usr/bin/env python3
"""Audit the train/val split for spatial leakage.

The problem
-----------
`train_v4.py` splits tiles with a plain random shuffle:

    rng.shuffle(idx); val = files[idx[:15%]]

That is only valid if tiles are independent samples. They usually are not.
NDWS-derived corpora draw multiple windows from the SAME fire event, often
overlapping in space and adjacent in time. Under a random shuffle, near-copies
of the same ground land on both sides of the split, so the model is scored on
data it effectively memorized. Val metrics come out optimistic, and — worse for
your current experiment — the inflation may differ between arms, which
contaminates the phase_a-vs-control comparison.

This is exactly why the WSTS benchmark uses 12-fold leave-one-year-out CV
instead of a random split (see docs/v5-research-informed-redesign.md §2.6).

The test
--------
Elevation is a static, location-specific field: two tiles covering the same
ground have near-identical elevation. So elevation is a fingerprint for "where".

  1. read the `elev` static band for every tile (mmap, one band only)
  2. fingerprint it two ways:
       exact  — hash of elevation rounded to 1 m  (same window, same place)
       coarse — hash of an 8x8 downsample rounded to 10 m (overlapping windows)
  3. group tiles by fingerprint
  4. measure how many groups STRADDLE the train/val boundary, and what
     fraction of val tiles have a twin in train

A val tile with a twin in train is a leaked tile. If that fraction is
material, every val number reported so far is optimistic.

Read-only. Changes nothing.

Usage
-----
    python -m ignis_ml.scripts.audit_split_leakage --config IgnisAI/ignis_ml/config.phase_a.yaml
    python -m ignis_ml.scripts.audit_split_leakage --config ... --max-tiles 6000
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_yaml(p: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(p.read_text())


def collect_tiles(cfg: Dict[str, Any]) -> List[Path]:
    from ignis_ml.src.utils.paths import resolve_data_path
    files: List[Path] = []
    for name, ds in (cfg.get("datasets") or {}).items():
        td = ds.get("tiles_dir")
        if not td:
            continue
        p = resolve_data_path(td)
        if p.is_dir():
            found = sorted(p.glob("*.npz"))
            print(f"[data] {name}: {len(found)} tiles in {p}")
            files += found
    return files


def split(files: List[Path], seed: int) -> Tuple[List[int], List[int]]:
    """Indices into `files`, byte-identical to train_v4.py."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * 0.15))
    return list(idx[n_val:]), list(idx[:n_val])


def fingerprints(p: Path, stat_names: Optional[List[str]] = None) -> Tuple[str, str]:
    """(exact, coarse) location fingerprints from the elevation band."""
    try:
        with np.load(p, mmap_mode="r", allow_pickle=False) as d:
            names = ([str(s) for s in d["stat_names"].astype(str)]
                     if "stat_names" in d.files else (stat_names or []))
            ei = names.index("elev") if "elev" in names else 0
            elev = np.asarray(d["x_stat"][ei], dtype=np.float64)
    except Exception:
        return "", ""

    if not np.isfinite(elev).any() or float(np.ptp(elev)) == 0.0:
        # flat/empty elevation carries no location information
        return "", ""

    exact = hashlib.blake2b(np.round(elev, 0).tobytes(), digest_size=12).hexdigest()

    h, w = elev.shape
    ch, cw = h // 8, w // 8
    small = elev[: ch * 8, : cw * 8].reshape(8, ch, 8, cw).mean(axis=(1, 3))
    coarse = hashlib.blake2b(
        np.round(small / 10.0, 0).tobytes(), digest_size=12).hexdigest()
    return exact, coarse


def report(tag: str, fps: List[str], tr: set, va: set) -> Dict[str, float]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, fp in enumerate(fps):
        if fp:
            groups[fp].append(i)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    straddling = {k: v for k, v in multi.items()
                  if any(i in tr for i in v) and any(i in va for i in v)}

    leaked_val = {i for v in straddling.values() for i in v if i in va}
    n_val = len(va)

    sizes = Counter(len(v) for v in groups.values())
    print(f"\n=== {tag} fingerprint ===")
    print(f"  tiles fingerprinted   : {sum(1 for f in fps if f)}")
    print(f"  distinct locations    : {len(groups)}")
    print(f"  groups with >1 tile   : {len(multi)}")
    print(f"  groups straddling split: {len(straddling)}")
    print(f"  VAL tiles with a twin in TRAIN: {len(leaked_val)} / {n_val} "
          f"({100.0 * len(leaked_val) / max(n_val, 1):.1f}%)")
    top = sorted(sizes.items())[:6]
    print(f"  group-size histogram  : {dict(top)}{' ...' if len(sizes) > 6 else ''}")
    return {"leak_frac": len(leaked_val) / max(n_val, 1),
            "n_groups": len(groups), "n_straddle": len(straddling)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit train/val split for spatial leakage")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)

    from ignis_ml.src.utils.paths import describe
    print(f"[audit] {describe()}")

    cfg = load_yaml(args.config)
    seed = int(cfg_get(cfg, "training.seed", 42))
    stat_order = cfg_get(cfg, "channels.static_order")
    files = collect_tiles(cfg)
    if not files:
        print("[audit] no tiles found")
        return 1

    tr_idx, va_idx = split(files, seed)
    print(f"[audit] train={len(tr_idx)} val={len(va_idx)} (seed={seed})")

    keep = None
    if args.max_tiles and len(files) > args.max_tiles:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(len(files), args.max_tiles, replace=False).tolist())
        print(f"[audit] subsampling to {args.max_tiles} tiles")

    print("[audit] fingerprinting elevation (mmap, 1 band per tile)...")
    exact_fps: List[str] = [""] * len(files)
    coarse_fps: List[str] = [""] * len(files)
    for i, p in enumerate(files):
        if keep is not None and i not in keep:
            continue
        exact_fps[i], coarse_fps[i] = fingerprints(p, stat_order)
        if (i + 1) % 4000 == 0:
            print(f"   ..{i + 1}/{len(files)}")

    tr, va = set(tr_idx), set(va_idx)
    if keep is not None:
        tr &= keep
        va &= keep

    r_exact = report("EXACT (identical window)", exact_fps, tr, va)
    r_coarse = report("COARSE (overlapping ground, ~10 m)", coarse_fps, tr, va)

    worst = max(r_exact["leak_frac"], r_coarse["leak_frac"])
    print("\n=== verdict ===")
    if worst >= 0.20:
        print(f"  ⛔ SEVERE leakage: {100*worst:.1f}% of val tiles have a twin in train.")
        print("     Every val number reported so far (phase_a AP=0.267, control, ...)")
        print("     is optimistic, and the arms are NOT safely comparable — the")
        print("     inflation need not be equal across runs.")
    elif worst >= 0.05:
        print(f"  ⚠  MATERIAL leakage: {100*worst:.1f}% of val tiles have a twin in train.")
        print("     Val metrics are somewhat optimistic. Relative comparisons between")
        print("     arms are probably still directionally valid, absolute numbers are not.")
    else:
        print(f"  ✓ Low leakage ({100*worst:.1f}%). The random split is defensible")
        print("    for this corpus; absolute val numbers can be taken at face value.")

    if worst >= 0.05:
        print("\n  Fixes, in order of rigor:")
        print("   1. GROUP SPLIT — split on the location fingerprint, so every tile")
        print("      sharing ground lands on one side. Cheapest correct fix; needs no")
        print("      re-ingestion. (This script already computes the groups.)")
        print("   2. Re-ingest carrying fire_id/date, then split by FIRE, and ideally")
        print("      by YEAR — matching the WSTS leave-one-year-out protocol, which")
        print("      is what makes published numbers comparable.")
        print("   3. Until then, treat val as 'seen-ish' and lean on the 5-preset OOD")
        print("      harness (eval_historical.py) for any claim about generalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
