"""Group-aware train/val splitting.

Why this module exists
----------------------
`train_v4.py` originally split tiles with a plain random shuffle. An audit
(`scripts/audit_split_leakage.py`) measured the consequence on
`mNDWS_500m_T3`:

    6000 tiles -> 5160 distinct locations
    683 location groups with >1 tile
    194 groups straddling the split
    201/890 val tiles (22.6%) had a byte-identical twin in train

Exact and coarse elevation fingerprints agreed exactly, meaning these are not
merely overlapping windows — they are the same ground on different days. NDWS
samples each fire repeatedly across its lifecycle, so a random shuffle trains
on day 5 of a fire and validates on day 7 of the same fire.

That inflates every val metric, and it does so unevenly across experiment arms,
which makes A/B comparisons unreliable.

The fix
-------
Split on LOCATION GROUPS rather than individual tiles: fingerprint each tile by
its elevation band (static, location-specific), group identical fingerprints,
and assign whole groups to train or val. Every tile sharing ground lands on one
side. This needs no re-ingestion.

Fingerprints are cached beside the tiles (`_split_fingerprints.json`) so the
cost is paid once, not on every run.

Limits — be honest about these
------------------------------
This groups by LOCATION, which is strictly better than random but weaker than
the field standard. WSTS uses 12-fold leave-one-year-out CV because the real
generalization question is "a fire year you have never seen." Grouping by
location does not prevent two DIFFERENT fires in the same region from spanning
the split, and it cannot group by year because stock NDWS tiles carry no date.
Re-ingesting with fire_id/date and splitting by year remains the right
long-term fix (see docs/v5-research-informed-redesign.md §2.6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FINGERPRINT_CACHE = "_split_fingerprints.json"


def _elev_fingerprint(p: Path, stat_order: Optional[Sequence[str]] = None) -> str:
    """Location fingerprint from the elevation band; "" when uninformative."""
    try:
        with np.load(p, mmap_mode="r", allow_pickle=False) as d:
            names = ([str(s) for s in d["stat_names"].astype(str)]
                     if "stat_names" in d.files else list(stat_order or []))
            ei = names.index("elev") if "elev" in names else 0
            elev = np.asarray(d["x_stat"][ei], dtype=np.float64)
    except Exception:
        return ""
    if not np.isfinite(elev).any() or float(np.ptp(elev)) == 0.0:
        return ""
    return hashlib.blake2b(np.round(elev, 0).tobytes(), digest_size=12).hexdigest()


def build_fingerprints(
    files: Sequence[Path],
    stat_order: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Map tile filename -> location fingerprint, cached on disk."""
    cache_path = Path(cache_dir) / FINGERPRINT_CACHE if cache_dir else None
    cache: Dict[str, str] = {}
    if cache_path and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    missing = [p for p in files if p.name not in cache]
    if missing:
        if verbose:
            print(f"[split] fingerprinting {len(missing)} tiles "
                  f"({len(cache)} cached)...")
        for i, p in enumerate(missing):
            cache[p.name] = _elev_fingerprint(p, stat_order)
            if verbose and (i + 1) % 5000 == 0:
                print(f"[split]   ..{i + 1}/{len(missing)}")
        if cache_path:
            try:
                cache_path.write_text(json.dumps(cache))
                if verbose:
                    print(f"[split] cached fingerprints -> {cache_path}")
            except Exception as e:
                if verbose:
                    print(f"[split] could not write cache ({e}) — continuing")
    return cache


def group_aware_split(
    files: Sequence[Path],
    *,
    seed: int = 42,
    val_frac: float = 0.15,
    stat_order: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Path], List[Path], dict]:
    """Split so that all tiles sharing a location land on the same side.

    Groups are shuffled with `seed` and greedily assigned to val until the tile
    budget is met, so the val fraction lands close to `val_frac` while never
    splitting a group. Tiles with no usable fingerprint are treated as their own
    singleton groups (they carry no location signal to leak).

    Returns (train_files, val_files, stats).
    """
    files = [Path(f) for f in files]
    fps = build_fingerprints(files, stat_order, cache_dir, verbose)

    groups: Dict[str, List[Path]] = {}
    for i, p in enumerate(files):
        fp = fps.get(p.name) or f"__singleton_{i}"
        groups.setdefault(fp, []).append(p)

    keys = sorted(groups)                       # deterministic before shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    target = int(len(files) * val_frac)
    val: List[Path] = []
    val_keys = set()
    for k in keys:
        if len(val) >= target:
            break
        val.extend(groups[k])
        val_keys.add(k)
    train = [p for k in keys if k not in val_keys for p in groups[k]]

    multi = sum(1 for v in groups.values() if len(v) > 1)
    stats = {
        "n_tiles": len(files),
        "n_groups": len(groups),
        "n_multi_tile_groups": multi,
        "n_train": len(train),
        "n_val": len(val),
        "val_frac_actual": len(val) / max(len(files), 1),
        "split": "group_aware",
        "leaked_val_tiles": 0,          # 0 by construction
    }
    if verbose:
        print(f"[split] group-aware: {len(groups)} location groups "
              f"({multi} with >1 tile) -> train={len(train)} val={len(val)} "
              f"({100 * stats['val_frac_actual']:.1f}%)")
    return train, val, stats


def random_split(
    files: Sequence[Path], *, seed: int = 42, val_frac: float = 0.15,
) -> Tuple[List[Path], List[Path], dict]:
    """The original leaky split. Kept ONLY so historical runs stay reproducible.

    Do not use for new experiments — measured 22.6% val/train twin overlap on
    mNDWS_500m_T3.
    """
    files = [Path(f) for f in files]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * val_frac))
    val = [files[i] for i in idx[:n_val]]
    train = [files[i] for i in idx[n_val:]]
    return train, val, {"n_train": len(train), "n_val": len(val),
                        "split": "random_LEAKY"}


def make_split(
    files: Sequence[Path], *, mode: str = "group", **kw
) -> Tuple[List[Path], List[Path], dict]:
    """Dispatch: mode="group" (default, correct) or "random" (legacy, leaky)."""
    if mode == "group":
        return group_aware_split(files, **kw)
    if mode == "random":
        return random_split(
            files, seed=kw.get("seed", 42), val_frac=kw.get("val_frac", 0.15))
    raise ValueError(f"split mode must be 'group' or 'random', got {mode!r}")
