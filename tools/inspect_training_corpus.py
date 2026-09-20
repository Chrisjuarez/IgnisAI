"""What the training tiles actually contain, as opposed to what they imply.

A ConvLSTM trained on a corpus whose weather never changes across the input
sequence cannot learn that wind drives spread over time, however long it is
trained. That is not visible in any loss curve, so it needs checking directly.

    python tools/inspect_training_corpus.py <tiles_dir>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List

import numpy as np

#: Below this a channel is constant across the sequence to within float noise.
CONSTANT = 1e-9


def inspect(tiles_dir: Path, sample: int = 200) -> int:
    # Exclude macOS AppleDouble sidecars: an external volume writes a ._name
    # companion per file, and they match *.npz without being npz.
    files = sorted(p for p in glob.glob(str(tiles_dir / "*.npz"))
                   if not Path(p).name.startswith("._"))[:sample]
    if not files:
        print(f"no tiles under {tiles_dir}", file=sys.stderr)
        return 1

    first = np.load(files[0], allow_pickle=True)
    dyn_names: List[str] = list(first["dyn_names"])
    change = np.zeros(len(dyn_names))
    fire_per_frame: List[List[float]] = []
    identical = 0
    zero_static = None
    stat_names = list(first["stat_names"])

    for path in files:
        d = np.load(path, allow_pickle=True)
        x = d["x_dyn"]
        for c in range(x.shape[1]):
            change[c] += float(np.abs(x[1:, c] - x[:-1, c]).mean())
        fire_per_frame.append([float((x[t, 0] > 0.5).sum()) for t in range(x.shape[0])])
        if all(np.array_equal(x[0], x[t]) for t in range(1, x.shape[0])):
            identical += 1
        st = d["x_stat"]
        z = np.array([not st[i].any() for i in range(st.shape[0])])
        zero_static = z if zero_static is None else (zero_static & z)

    n = len(files)
    change /= n
    print(f"  {n} tiles from {tiles_dir.name}\n")
    print("  frame-to-frame change per dynamic channel:")
    constant = []
    for name, value in zip(dyn_names, change):
        flag = ""
        if value < CONSTANT:
            constant.append(name)
            flag = "   <- CONSTANT across the sequence"
        print(f"    {name:<10} {value:.6f}{flag}")

    ff = np.array(fire_per_frame)
    print(f"\n  mean fire pixels per frame: {np.round(ff.mean(axis=0), 1)}")
    print(f"  tiles with all frames identical: {identical} of {n}")
    zeroed = [s for s, z in zip(stat_names, zero_static) if z]
    print(f"  static channels zero in every tile: {zeroed or 'none'}")

    # Only the channels that steer spread matter here. precip is legitimately
    # constant through a dry fire week, and flagging it hides the real case.
    STEERING = {"u", "v", "gust", "tempC"}
    weather = [c for c in constant if c in STEERING]
    benign = [c for c in constant if c not in STEERING and c != "fire_t"]
    print()
    if benign and not weather:
        print(f"  Constant but benign: {', '.join(benign)} - unchanged through a dry")
        print("  week is a real observation, not a corpus defect.")
    if weather:
        print("  FINDING: the weather channels that steer spread do not vary across")
        print("  the input sequence.")
        print("  A recurrent model cannot learn wind-driven spread from frames whose")
        print("  wind is identical - the only thing changing is the fire mask, so the")
        print("  recurrence can only learn the shape of that synthetic growth.")
    if ff.shape[1] > 1 and ff[:, 0].mean() < 1.0:
        print()
        print("  FINDING: the first frame is empty in essentially every tile. That")
        print("  pattern never occurs at serving time, where a live incident has been")
        print("  burning for days.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/inspect_training_corpus.py")
    ap.add_argument("tiles_dir", type=Path)
    ap.add_argument("--sample", type=int, default=200)
    return inspect(ap.parse_args(argv).tiles_dir, ap.parse_args(argv).sample)


if __name__ == "__main__":
    raise SystemExit(main())
