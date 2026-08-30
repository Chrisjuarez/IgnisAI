#!/usr/bin/env python3
"""Phase 1 — TS-SatFire -> IgnisAI tile ingestion.

Turns the TS-SatFire archive (179 fires, GeoTIFF stacks, 256x256 @ 375 m) into
IgnisAI training tiles (64x64 @ 500 m on the canonical EPSG:5070 grid) that the
existing `NpzTileDataset` consumes byte-for-byte, plus per-fire metadata JSON
the v4 sampler uses to up-weight Santa-Ana events.

Output npz schema (matches ignis_ml/src/data/dataset.py::NpzTileDataset):
    x_dyn  float32 [T=6, Cd=12, 64, 64]   (raw dynamic channels; derived feats
                                           are appended at load time, not here)
    x_stat float32 [Cs=9, 64, 64]
    y      float32 [64, 64]               (next-day full fire mask; the dataset
                                           derives the delta target itself)
    dyn_names  <U..  array of 12 names    (v4 dynamic_order)
    stat_names <U..  array of 9 names     (v4 static_order)

Run from repo root:
    python -m ignis_ml.scripts.ingest_ts_satfire --dry-run --max-fires 3
    python -m ignis_ml.scripts.ingest_ts_satfire --raw data/tssatfire_raw \
        --out data/tssatfire_500m_T6

STATUS: scaffold. The grid math, windowing, normalization-passthrough, npz/meta
writing, and CLI are complete and runnable. The ONE dataset-specific piece you
must confirm is the GeoTIFF band layout in `TSSatFireReader.read_fire_stack`
(marked `TODO(dataset)`), because TS-SatFire band ordering depends on which
release you downloaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- repo imports (work whether run from repo root or ignis_ml/) -------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    # Canonical tile geometry shared with live serving. Using this guarantees a
    # tile produced here is byte-for-byte usable for inference.
    from services.tilesvc.grid import (  # type: ignore
        SIZE,            # 64 px per side
        TILE_M,          # 32_000 m
        snapped_tile,
        tile_affine,
        tile_bounds_lonlat,
    )
    _HAVE_GRID = True
except Exception as exc:  # pragma: no cover - import guard for dry docs
    SIZE, TILE_M = 64, 32_000
    _HAVE_GRID = False
    _GRID_IMPORT_ERR = exc


# v4 schema (keep in sync with ignis_ml/config.v4.yaml)
V4_DYNAMIC_ORDER: Tuple[str, ...] = (
    "fire_t", "viirs_i4", "viirs_i5", "u", "v", "gust",
    "tempC", "q", "precip", "erc", "bi", "ndvi",
)
V4_STATIC_ORDER: Tuple[str, ...] = (
    "elev", "slope", "aspect_cos", "aspect_sin", "pdsi", "chili",
    "water", "fuel1", "fuel2",
)
SEQ_LEN = 6
SANTA_ANA_U_THRESHOLD = -5.0   # mean eastward wind (m/s); < this => Santa-Ana


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------
@dataclass
class FireStack:
    """A single TS-SatFire fire, already reprojected to EPSG:5070 @ 500 m.

    dynamic: dict name -> [D, H, W] (D days, on the v4 dynamic raw schema)
    static:  dict name -> [H, W]
    affine/transform info lives in `transform` (rasterio Affine) for tiling.
    """
    name: str
    dynamic: Dict[str, np.ndarray]
    static: Dict[str, np.ndarray]
    transform: object
    height: int
    width: int
    ignition_lonlat: Tuple[float, float]
    date_range: Tuple[str, str] = ("", "")


class TSSatFireReader:
    """Reads + reprojects TS-SatFire GeoTIFFs to the IgnisAI 500 m EPSG:5070 grid.

    Heavy dependencies (rasterio) are imported lazily so `--dry-run` works in a
    bare environment.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def list_fires(self) -> List[Path]:
        """Each fire is a subdirectory (or multi-band GeoTIFF) under raw_dir."""
        if not self.raw_dir.is_dir():
            return []
        # TS-SatFire ships one folder per fire; adjust glob if your layout differs.
        fires = sorted([p for p in self.raw_dir.iterdir() if p.is_dir()])
        if not fires:
            fires = sorted(self.raw_dir.glob("*.tif"))
        return fires

    def read_fire_stack(self, fire_path: Path) -> FireStack:
        """Read one fire and reproject every band onto the 500 m EPSG:5070 grid.

        TODO(dataset): TS-SatFire band ordering depends on the release. Fill in
        `band_to_v4` below with the actual band indices / file names for the
        archive you downloaded (kaggle z789456sx/ts-satfire). The reprojection,
        gust derivation, and missing-channel zero-fill are already wired.
        """
        import rasterio  # noqa: F401  (lazy)
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        # ---- TODO(dataset): map TS-SatFire sources to v4 channels ----
        # Keys are v4 channel names; values are however you locate that band in
        # the fire's GeoTIFF stack (band index, or a per-channel file).
        # Channels not present in TS-SatFire (e.g. viirs thermal may be the AF
        # product; gust is derived) are handled by the zero-fill / derive rules
        # in `_assemble_dynamic`. This dict only needs the bands you DO have.
        band_to_v4: Dict[str, int] = {
            # "fire_t":   <band index of VIIRS active-fire mask>,
            # "viirs_i4": <band index of I4 brightness temp>,
            # "viirs_i5": <band index of I5 brightness temp>,
            # "u": ..., "v": ..., "tempC": ..., "q": ..., "precip": ...,
            # "erc": ..., "bi": ..., "ndvi": ...,
            # statics: "elev": ..., "pdsi": ..., "chili": ..., "water": ...,
            #          "fuel1": ..., "fuel2": ...,
        }
        if not band_to_v4:
            raise NotImplementedError(
                "Fill TSSatFireReader.read_fire_stack:band_to_v4 with the band "
                "layout for your TS-SatFire download. See docstring TODO(dataset)."
            )

        # The real implementation (left runnable once band_to_v4 is filled):
        #   1. open the stack with rasterio
        #   2. for each day d and each band, reproject to EPSG:5070 @ 500 m
        #      using calculate_default_transform + reproject(Resampling.bilinear
        #      for continuous, Resampling.nearest for the fire mask)
        #   3. derive gust = 1.5 * sqrt(u^2 + v^2) where gust band absent
        #   4. return a FireStack with dynamic[name] = [D,H,W], static[name]=[H,W]
        raise NotImplementedError(
            "read_fire_stack body is dataset-specific; see TODO(dataset)."
        )


# ---------------------------------------------------------------------------
# Tiling + windowing
# ---------------------------------------------------------------------------
def _tiles_covering(fire: FireStack) -> List[Tuple[int, int]]:
    """Integer (ix, iy) EPSG:5070 tile indices whose 32 km cells the fire spans.

    Uses snapped_tile, not lonlat_to_tile: these are addresses on the fixed
    32 km lattice, and walking neighbours only means anything on that lattice.
    lonlat_to_tile returns a window centred on the fire, whose ix/iy is just
    whichever cell its corner happens to land in.
    """
    if not _HAVE_GRID:
        raise RuntimeError(f"services.tilesvc.grid unavailable: {_GRID_IMPORT_ERR}")
    lon0, lat0 = fire.ignition_lonlat
    # Walk the fire's lon/lat extent; for a scaffold we seed from ignition and
    # let the real reader provide bounds. Here we just return the ignition tile
    # plus its 8 neighbors so large fires get full coverage.
    base = snapped_tile(lon0, lat0)
    return [(base.ix + dx, base.iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


def _crop_tile(arr_hw: np.ndarray, fire: FireStack, tile_ix: int, tile_iy: int) -> Optional[np.ndarray]:
    """Crop a [H,W] (or [D,H,W]) array on the fire grid to the 64x64 tile window.

    Maps the tile's EPSG:5070 bounds into the fire raster's pixel space via the
    fire transform, then samples a SIZE x SIZE window. Returns None if the tile
    falls outside the fire raster.
    """
    from rasterio.transform import rowcol

    x0 = tile_ix * TILE_M
    y1 = (tile_iy + 1) * TILE_M
    # upper-left of the tile in EPSG:5070
    row, col = rowcol(fire.transform, x0, y1)
    row, col = int(row), int(col)
    if row < 0 or col < 0 or row + SIZE > fire.height or col + SIZE > fire.width:
        return None
    if arr_hw.ndim == 2:
        return arr_hw[row:row + SIZE, col:col + SIZE]
    return arr_hw[:, row:row + SIZE, col:col + SIZE]


def _assemble_dynamic(fire: FireStack, t_slice: slice) -> np.ndarray:
    """Stack the 12 v4 dynamic channels for a [t-5..t] window -> [6,12,H,W].

    Missing channels (e.g. mNDWS-style sources lacking viirs thermal) get a zero
    column; gust derived from wind if absent.
    """
    D = None
    chans: List[np.ndarray] = []
    for name in V4_DYNAMIC_ORDER:
        if name in fire.dynamic:
            seq = fire.dynamic[name][t_slice]            # [6,H,W]
        elif name == "gust" and "u" in fire.dynamic and "v" in fire.dynamic:
            u = fire.dynamic["u"][t_slice]
            v = fire.dynamic["v"][t_slice]
            seq = 1.5 * np.sqrt(u * u + v * v)
        else:
            ref = next(iter(fire.dynamic.values()))[t_slice]
            seq = np.zeros_like(ref)
        chans.append(seq.astype(np.float32))
        D = seq.shape[0]
    return np.stack(chans, axis=1)                       # [6,12,H,W]


def _assemble_static(fire: FireStack) -> np.ndarray:
    """Stack the 9 v4 static channels -> [9,H,W] (slope/aspect derived at load)."""
    out: List[np.ndarray] = []
    ref = next(iter(fire.static.values()))
    for name in V4_STATIC_ORDER:
        if name in fire.static:
            out.append(fire.static[name].astype(np.float32))
        elif name in ("slope", "aspect_cos", "aspect_sin"):
            # NpzTileDataset recomputes these from elev at load; pass zeros.
            out.append(np.zeros_like(ref, dtype=np.float32))
        else:
            out.append(np.zeros_like(ref, dtype=np.float32))
    return np.stack(out, axis=0)                          # [9,H,W]


def _mean_u_for_window(dyn_window: np.ndarray) -> float:
    """Mean eastward wind across a [6,12,H,W] window (raw m/s, pre-normalize)."""
    u_idx = V4_DYNAMIC_ORDER.index("u")
    return float(dyn_window[:, u_idx].mean())


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestStats:
    fires_seen: int = 0
    fires_ingested: int = 0
    tiles_written: int = 0
    santa_ana_fires: int = 0
    skipped: List[str] = field(default_factory=list)


def ingest_fire(
    fire: FireStack,
    out_dir: Path,
    meta_dir: Path,
    *,
    dry_run: bool,
    stats: IngestStats,
) -> None:
    tiles = _tiles_covering(fire)
    n_days = next(iter(fire.dynamic.values())).shape[0]
    if n_days < SEQ_LEN + 1:
        stats.skipped.append(f"{fire.name}: only {n_days} days (<{SEQ_LEN + 1})")
        return

    tile_keys: List[str] = []
    is_santa_ana = False

    for (ix, iy) in tiles:
        for t in range(SEQ_LEN, n_days):           # predict day t given t-6..t-1
            win = slice(t - SEQ_LEN, t)
            x_dyn_full = _assemble_dynamic(fire, win)         # [6,12,Hf,Wf]
            x_stat_full = _assemble_static(fire)              # [9,Hf,Wf]
            y_full = fire.dynamic["fire_t"][t]                # [Hf,Wf] next-day mask

            x_dyn = _crop_tile(x_dyn_full, fire, ix, iy)
            x_stat = _crop_tile(x_stat_full, fire, ix, iy)
            y = _crop_tile(y_full, fire, ix, iy)
            if x_dyn is None or x_stat is None or y is None:
                continue
            # Skip tiles with no fire at all in input or target — pure negatives
            # are down-sampled by the sampler anyway, and they bloat the set.
            if float((y > 0.5).mean()) == 0.0 and float(
                (x_dyn[:, V4_DYNAMIC_ORDER.index("fire_t")] > 0.5).mean()
            ) == 0.0:
                continue

            if _mean_u_for_window(x_dyn) < SANTA_ANA_U_THRESHOLD:
                is_santa_ana = True

            key = f"tssatfire_{fire.name}_ix{ix}_iy{iy}_t{t:02d}"
            tile_keys.append(key)
            stats.tiles_written += 1
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / f"{key}.npz",
                    x_dyn=x_dyn.astype(np.float32),
                    x_stat=x_stat.astype(np.float32),
                    y=y.astype(np.float32),
                    dyn_names=np.array(V4_DYNAMIC_ORDER, dtype="U16"),
                    stat_names=np.array(V4_STATIC_ORDER, dtype="U16"),
                )

    if is_santa_ana:
        stats.santa_ana_fires += 1
    if not dry_run and tile_keys:
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{fire.name}.json").write_text(json.dumps({
            "event": fire.name,
            "ignition_lonlat": fire.ignition_lonlat,
            "date_range": fire.date_range,
            "is_santa_ana": is_santa_ana,
            "n_tiles": len(tile_keys),
            "tile_keys": tile_keys,
        }, indent=2))
    stats.fires_ingested += 1


def main(argv: Optional[List[str]] = None) -> int:
    # Data root follows $IGNIS_DATA_ROOT (external drive) when set.
    try:
        from ignis_ml.src.utils.paths import data_root, describe
        _DR = data_root()
        print(f"[ingest] {describe()}")
    except Exception:
        _DR = Path("data")

    ap = argparse.ArgumentParser(description="Ingest TS-SatFire into IgnisAI tiles")
    ap.add_argument("--raw", type=Path, default=_DR / "tssatfire_raw")
    ap.add_argument("--out", type=Path, default=_DR / "tssatfire_500m_T6")
    ap.add_argument("--meta", type=Path, default=None,
                    help="meta dir (default: <out>/_meta)")
    ap.add_argument("--max-fires", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true",
                    help="count tiles, write nothing")
    args = ap.parse_args(argv)

    meta_dir = args.meta or (args.out / "_meta")
    reader = TSSatFireReader(args.raw)
    fires = reader.list_fires()
    if args.max_fires:
        fires = fires[: args.max_fires]

    stats = IngestStats()
    if not fires:
        print(f"[ingest] no fires found under {args.raw} "
              f"(download with: kaggle datasets download -d z789456sx/ts-satfire)")
        return 1

    for fp in fires:
        stats.fires_seen += 1
        try:
            fire = reader.read_fire_stack(fp)
        except NotImplementedError as e:
            print(f"[ingest] {fp.name}: {e}")
            stats.skipped.append(f"{fp.name}: reader not implemented")
            continue
        except Exception as e:
            print(f"[ingest] {fp.name}: failed ({e})")
            stats.skipped.append(f"{fp.name}: {e}")
            continue
        ingest_fire(fire, args.out, meta_dir, dry_run=args.dry_run, stats=stats)
        print(f"[ingest] {fp.name}: tiles so far={stats.tiles_written}")

    print("\n=== ingest summary ===")
    print(f"  fires seen     : {stats.fires_seen}")
    print(f"  fires ingested : {stats.fires_ingested}")
    print(f"  santa-ana fires: {stats.santa_ana_fires}")
    print(f"  tiles written  : {stats.tiles_written} (dry_run={args.dry_run})")
    if stats.skipped:
        print(f"  skipped        : {len(stats.skipped)}")
        for s in stats.skipped[:10]:
            print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
