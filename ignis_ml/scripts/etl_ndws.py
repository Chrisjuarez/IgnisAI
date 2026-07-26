#!/usr/bin/env python3
"""ETL: mNDWS (modified Next-Day Wildfire Spread) -> IgnisAI training tiles.

Rebuild of the `etl_ndws` module that produced `data/mNDWS_500m_T6/` for v3.
The original lived on Nautilus and is not present in this repo (home-wide search
found no source and no __pycache__), so this reconstructs it from the contracts
it must satisfy:

  * `ignis_ml/src/data/dataset.py::NpzTileDataset` — the npz schema it reads,
    including the legacy 12-static fallback order (`keys_guess`), which is the
    strongest surviving evidence of what the original ETL wrote.
  * `ignis_ml/config.yaml` (v3) and `config.v4.yaml` (v4) — channel orders.
  * `ignis_ml/scripts/ingest_ts_satfire.py` — sibling ingester; same output
    schema, same CLI conventions, so the two corpora merge cleanly.

Output npz schema (identical to the TS-SatFire ingester):
    x_dyn  float32 [T, Cd, 64, 64]   raw dynamic channels (derived feats are
                                     appended at load time by the dataset)
    x_stat float32 [Cs, 64, 64]      slope/aspect are recomputed from elev at
                                     load time; written as zeros here
    y      float32 [64, 64]          next-day full fire mask
    dyn_names  <U16 array
    stat_names <U16 array

Run from repo root:
    python -m ignis_ml.scripts.etl_ndws --dry-run --max-samples 32
    python -m ignis_ml.scripts.etl_ndws --schema v4 --out "$IGNIS_DATA_ROOT/mNDWS_500m_T6"

------------------------------------------------------------------------------
READ THIS BEFORE TRUSTING THE OUTPUT
------------------------------------------------------------------------------
Two things about NDWS are genuinely ambiguous without the raw archive in hand,
and both are surfaced as explicit CLI flags rather than silently guessed:

1. TEMPORAL DEPTH.  Canonical NDWS (Huot et al. 2022) is a *single-day* problem:
   one PrevFireMask + same-day covariates -> next-day FireMask. It has no
   multi-day sequences. But the v3 tiles directory is named `mNDWS_500m_T6`,
   i.e. T=6. So the "modified" in mNDWS did one of:
     (a) replicated the single day T times  -> `--seq-strategy replicate`
     (b) grouped consecutive days per fire  -> `--seq-strategy group`
   Default is `replicate`, because it is reproducible from stock NDWS. It is
   also *degenerate*: every frame identical means the ConvLSTM's recurrence sees
   no temporal signal on this corpus, and the 6 derived temporal features
   (delta_*, days_since_fire) are constant. If v3 was trained this way, that is
   a material finding — see docs/v5-research-informed-redesign.md §1.2(6), where
   time-series input is the one thing that reliably helps.
   Use `--seq-strategy group` only if your archive carries fire-id + date.

2. STATIC COVERAGE.  NDWS supplies only 5 of the statics IgnisAI wants
   (elev, ndvi, erc, pdsi, population); `bi` is derived from erc. The rest
   (chili, impervious, water, fuel1..3) are NOT in NDWS and are ZERO-FILLED.
   Concretely:
       v3 schema -> zero-filled: chili, impervious, water, fuel1, fuel2, fuel3
       v4 schema -> zero-filled: chili, water, fuel1, fuel2   (+ dynamic
                    viirs_i4/viirs_i5, which is correct and intended: the
                    gameplan's channel map says mNDWS gets a zero thermal column)
   Every zero-filled channel is listed in the run summary and recorded in
   `_etl_manifest.json`, so a training run can never silently depend on a
   channel that was never populated. Populating them properly means sampling
   `data/source-rasters/` (nlcd -> water/impervious, landfire -> fuel PCs) at
   each tile's footprint — which needs per-sample georeferencing that stock
   NDWS TFRecords do not carry. If your archive has lon/lat per sample, that is
   the hook to add next.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --- repo imports (work whether run from repo root or ignis_ml/) -------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from ignis_ml.src.data.transforms import wind_to_uv  # type: ignore
except Exception:  # pragma: no cover - keep --dry-run usable in a bare env
    def wind_to_uv(speed_ms, dir_deg):
        a = np.deg2rad(dir_deg)
        return -np.sin(a) * speed_ms, -np.cos(a) * speed_ms


# ---------------------------------------------------------------------------
# Schemas (keep in sync with config.yaml / config.v4.yaml)
# ---------------------------------------------------------------------------
V3_DYNAMIC_ORDER: Tuple[str, ...] = (
    "fire_t", "u", "v", "gust", "tempC", "q", "precip",
)
# NOTE: the v3 *file* order is the 12 statics below. slope/aspect_cos/aspect_sin
# are computed from elev by NpzTileDataset at load time (12 + 3 = 15 = v3 Cs).
# This ordering is taken verbatim from dataset.py's `keys_guess` fallback.
V3_STATIC_ORDER: Tuple[str, ...] = (
    "elev", "ndvi", "bi", "erc", "pdsi", "chili",
    "impervious", "water", "population", "fuel1", "fuel2", "fuel3",
)

V4_DYNAMIC_ORDER: Tuple[str, ...] = (
    "fire_t", "viirs_i4", "viirs_i5", "u", "v", "gust",
    "tempC", "q", "precip", "erc", "bi", "ndvi",
)
V4_STATIC_ORDER: Tuple[str, ...] = (
    "elev", "slope", "aspect_cos", "aspect_sin", "pdsi", "chili",
    "water", "fuel1", "fuel2",
)

SCHEMAS = {
    "v3": (V3_DYNAMIC_ORDER, V3_STATIC_ORDER),
    "v4": (V4_DYNAMIC_ORDER, V4_STATIC_ORDER),
}

SEQ_LEN = 6
TILE_PX = 64            # output tile is 64x64
NDWS_PX = 64            # NDWS samples are 64x64
NDWS_RES_M = 1000.0     # NDWS native resolution
OUT_RES_M = 500.0       # IgnisAI tile resolution

# NDWS canonical TFRecord feature names (Huot et al. 2022).
NDWS_FEATURES: Tuple[str, ...] = (
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr",
    "pdsi", "NDVI", "population", "erc", "PrevFireMask", "FireMask",
)
# FireMask / PrevFireMask use -1 for "uncertain / unlabeled".
NDWS_UNCERTAIN = -1.0


# ---------------------------------------------------------------------------
# Raw sample container
# ---------------------------------------------------------------------------
@dataclass
class NdwsSample:
    """One NDWS record: 64x64 @ 1 km, all bands on the same grid.

    `bands` holds raw NDWS feature names -> [H,W] float32 (native units).
    `fire_id` / `date` are optional and only present in modified archives; they
    are what `--seq-strategy group` needs.
    """
    bands: Dict[str, np.ndarray]
    index: int
    fire_id: Optional[str] = None
    date: Optional[str] = None

    @property
    def key(self) -> str:
        if self.fire_id:
            return f"{self.fire_id}_{self.date or self.index:04}"
        return f"s{self.index:06d}"


# ---------------------------------------------------------------------------
# Readers — auto-detect the archive layout
# ---------------------------------------------------------------------------
class NdwsReader:
    """Reads NDWS samples from TFRecord, npz, or per-sample GeoTIFF layouts.

    Heavy deps (tensorflow / rasterio) are imported lazily so `--dry-run` and
    `--list-only` work in a bare environment.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.kind = self._detect()

    def _detect(self) -> str:
        if not self.raw_dir.is_dir():
            return "missing"
        if any(self.raw_dir.rglob("*.tfrecord")) or any(self.raw_dir.rglob("*.tfrecord*")):
            return "tfrecord"
        if any(self.raw_dir.rglob("*.npz")):
            return "npz"
        if any(self.raw_dir.rglob("*.tif")) or any(self.raw_dir.rglob("*.tiff")):
            return "geotiff"
        return "unknown"

    def describe(self) -> str:
        return f"raw_dir={self.raw_dir} kind={self.kind}"

    # -- iteration ---------------------------------------------------------
    def iter_samples(self, max_samples: int = 0) -> Iterator[NdwsSample]:
        if self.kind == "tfrecord":
            it = self._iter_tfrecord()
        elif self.kind == "npz":
            it = self._iter_npz()
        elif self.kind == "geotiff":
            it = self._iter_geotiff()
        elif self.kind == "missing":
            raise FileNotFoundError(
                f"NDWS raw dir not found: {self.raw_dir}\n"
                f"  Expected the mNDWS archive (see config.yaml "
                f"datasets.mndws.raw_dir), or pull the already-tiled "
                f"mNDWS_500m_T6 from Nautilus and skip this ETL entirely."
            )
        else:
            raise RuntimeError(
                f"Could not detect NDWS layout under {self.raw_dir}. "
                f"Expected *.tfrecord, *.npz, or *.tif. Contents: "
                f"{[p.name for p in list(self.raw_dir.iterdir())[:10]]}"
            )
        for i, s in enumerate(it):
            if max_samples and i >= max_samples:
                return
            yield s

    def _iter_tfrecord(self) -> Iterator[NdwsSample]:
        """Canonical NDWS distribution: TFRecords of 64x64 float lists."""
        import tensorflow as tf  # lazy; only needed for this layout

        files = sorted(
            [str(p) for p in self.raw_dir.rglob("*.tfrecord")]
            or [str(p) for p in self.raw_dir.rglob("*.tfrecord*")]
        )
        spec = {
            name: tf.io.FixedLenFeature([NDWS_PX * NDWS_PX], tf.float32)
            for name in NDWS_FEATURES
        }
        ds = tf.data.TFRecordDataset(files, compression_type="")
        for i, raw in enumerate(ds):
            parsed = tf.io.parse_single_example(raw, spec)
            bands = {
                k: np.asarray(v, dtype=np.float32).reshape(NDWS_PX, NDWS_PX)
                for k, v in parsed.items()
            }
            yield NdwsSample(bands=bands, index=i)

    def _iter_npz(self) -> Iterator[NdwsSample]:
        """A pre-converted archive: one npz per sample, arrays keyed by feature."""
        for i, p in enumerate(sorted(self.raw_dir.rglob("*.npz"))):
            with np.load(p, allow_pickle=False) as d:
                bands = {k: d[k].astype(np.float32) for k in d.files
                         if d[k].ndim == 2}
                meta_id = str(d["fire_id"]) if "fire_id" in d.files else None
                meta_dt = str(d["date"]) if "date" in d.files else None
            if not bands:
                continue
            yield NdwsSample(bands=bands, index=i, fire_id=meta_id, date=meta_dt)

    def _iter_geotiff(self) -> Iterator[NdwsSample]:
        """One multi-band GeoTIFF per sample; band order = NDWS_FEATURES."""
        import rasterio

        for i, p in enumerate(sorted(self.raw_dir.rglob("*.tif"))):
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)      # [B,H,W]
                names = list(src.descriptions or [])
            if names and all(names):
                bands = {n: arr[j] for j, n in enumerate(names)}
            else:
                bands = {n: arr[j] for j, n in enumerate(NDWS_FEATURES[: arr.shape[0]])}
            yield NdwsSample(bands=bands, index=i, fire_id=p.stem)


# ---------------------------------------------------------------------------
# NDWS -> IgnisAI channel mapping
# ---------------------------------------------------------------------------
def _clean(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(a.astype(np.float32), nan=fill, posinf=fill, neginf=fill)


def _fire_mask(a: np.ndarray) -> np.ndarray:
    """NDWS fire masks use -1 for uncertain. Treat uncertain as no-fire."""
    m = _clean(a, 0.0)
    return (m > 0.5).astype(np.float32)


def _ndvi(a: np.ndarray) -> np.ndarray:
    """NDWS ships NDVI scaled by 1e4 in most releases; auto-detect and rescale."""
    v = _clean(a, 0.0)
    if np.nanmax(np.abs(v)) > 1.5:          # clearly not in [-1,1]
        v = v / 10000.0
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def map_ndws_to_channels(bands: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray],
                                                                Dict[str, np.ndarray],
                                                                np.ndarray]:
    """Map one NDWS sample onto IgnisAI channel names.

    Returns (dynamic, static, y) where dynamic/static values are [H,W] and y is
    the next-day full fire mask. Channels IgnisAI wants but NDWS lacks are simply
    absent from the dicts — the assembler zero-fills and reports them.
    """
    b = bands

    def get(*names: str) -> Optional[np.ndarray]:
        for n in names:
            if n in b:
                return b[n]
        return None

    # ---- wind: NDWS gives speed (vs, m/s) + direction (th, deg FROM) --------
    vs = get("vs", "wind_speed")
    th = get("th", "wind_direction")
    dynamic: Dict[str, np.ndarray] = {}
    if vs is not None and th is not None:
        u, v = wind_to_uv(_clean(vs), _clean(th))
        dynamic["u"] = u.astype(np.float32)
        dynamic["v"] = v.astype(np.float32)
        # NDWS has no gust; the gameplan's channel map says derive 1.5 x wind.
        dynamic["gust"] = (1.5 * _clean(vs)).astype(np.float32)

    # ---- temperature: NDWS tmmn/tmmx are Kelvin ----------------------------
    tmmn, tmmx = get("tmmn"), get("tmmx")
    if tmmn is not None and tmmx is not None:
        tmean_k = 0.5 * (_clean(tmmn) + _clean(tmmx))
        dynamic["tempC"] = (tmean_k - 273.15).astype(np.float32)
    elif get("tempC") is not None:
        dynamic["tempC"] = _clean(get("tempC"))

    # ---- humidity / precip -------------------------------------------------
    sph = get("sph", "q")
    if sph is not None:
        dynamic["q"] = _clean(sph)                     # already kg/kg
    pr = get("pr", "precip")
    if pr is not None:
        dynamic["precip"] = _clean(pr)                 # mm/day

    # ---- previous fire mask -> fire_t --------------------------------------
    prev = get("PrevFireMask", "prev_fire_mask", "fire_t")
    if prev is not None:
        dynamic["fire_t"] = _fire_mask(prev)

    # ---- v4 moves erc/bi/ndvi to dynamic; provide both, assembler picks -----
    erc = get("erc", "ERC")
    ndvi = get("NDVI", "ndvi")
    if erc is not None:
        dynamic["erc"] = _clean(erc)
    if ndvi is not None:
        dynamic["ndvi"] = _ndvi(ndvi)
    # NDWS has no Burning Index; BI correlates strongly with ERC. The gameplan's
    # channel map says "compute from ERC if missing". Linear proxy, flagged in
    # the run manifest so it is never mistaken for observed BI.
    if erc is not None:
        dynamic["bi"] = np.clip(_clean(erc) * 1.5, 0.0, 200.0).astype(np.float32)

    # ---- statics -----------------------------------------------------------
    static: Dict[str, np.ndarray] = {}
    elev = get("elevation", "elev")
    if elev is not None:
        static["elev"] = _clean(elev)
    if ndvi is not None:
        static["ndvi"] = _ndvi(ndvi)
    if erc is not None:
        static["erc"] = _clean(erc)
        static["bi"] = np.clip(_clean(erc) * 1.5, 0.0, 200.0).astype(np.float32)
    pdsi = get("pdsi", "PDSI")
    if pdsi is not None:
        static["pdsi"] = _clean(pdsi)
    pop = get("population", "pop")
    if pop is not None:
        static["population"] = _clean(pop)

    # ---- target ------------------------------------------------------------
    fm = get("FireMask", "fire_mask", "y")
    if fm is None:
        raise KeyError(
            f"No FireMask band in sample; available={sorted(b.keys())[:15]}"
        )
    y = _fire_mask(fm)

    return dynamic, static, y


# ---------------------------------------------------------------------------
# Resampling: NDWS 64x64 @1 km  ->  IgnisAI 64x64 @500 m
# ---------------------------------------------------------------------------
def resample_to_tile(a: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Center-crop 32x32 (=32 km at 1 km) then 2x upsample -> 64x64 @ 500 m.

    A 64x64 @1 km NDWS sample covers 64 km; an IgnisAI tile covers 32 km at
    500 m. So we take the central 32 km and resample it up by 2. Nearest for
    masks (keeps them binary), bilinear-ish for continuous fields.
    """
    h, w = a.shape[-2:]
    ch, cw = h // 2, w // 2                      # 32x32 center crop
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    crop = a[..., r0:r0 + ch, c0:c0 + cw]

    if nearest:
        out = np.repeat(np.repeat(crop, 2, axis=-2), 2, axis=-1)
    else:
        # separable linear interpolation without scipy
        out = np.repeat(np.repeat(crop, 2, axis=-2), 2, axis=-1).astype(np.float32)
        out[..., 1:-1, :] = 0.5 * (out[..., 1:-1, :] + out[..., 2:, :])
        out[..., :, 1:-1] = 0.5 * (out[..., :, 1:-1] + out[..., :, 2:])
    if out.shape[-2:] != (TILE_PX, TILE_PX):
        out = out[..., :TILE_PX, :TILE_PX]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_dynamic(
    dyn_frames: Sequence[Dict[str, np.ndarray]],
    dyn_order: Sequence[str],
    missing: set,
) -> np.ndarray:
    """Stack per-frame dicts -> [T, Cd, 64, 64]."""
    T = len(dyn_frames)
    out = np.zeros((T, len(dyn_order), TILE_PX, TILE_PX), dtype=np.float32)
    for t, frame in enumerate(dyn_frames):
        for c, name in enumerate(dyn_order):
            arr = frame.get(name)
            if arr is None:
                missing.add(name)
                continue
            out[t, c] = resample_to_tile(arr, nearest=(name == "fire_t"))
    return out


def assemble_static(
    stat: Dict[str, np.ndarray],
    stat_order: Sequence[str],
    missing: set,
) -> np.ndarray:
    """Stack statics -> [Cs, 64, 64]. slope/aspect are left as zeros by design."""
    out = np.zeros((len(stat_order), TILE_PX, TILE_PX), dtype=np.float32)
    for c, name in enumerate(stat_order):
        if name in ("slope", "aspect_cos", "aspect_sin"):
            continue                       # recomputed from elev at load time
        arr = stat.get(name)
        if arr is None:
            missing.add(name)
            continue
        out[c] = resample_to_tile(arr)
    return out


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(
    samples: List[NdwsSample],
    strategy: str,
    seq_len: int,
) -> Iterator[Tuple[str, List[NdwsSample], NdwsSample]]:
    """Yield (key, history_frames, target_sample).

    replicate : each NDWS sample becomes one T-frame window with all frames
                identical. Reproducible from stock NDWS; temporally degenerate.
    group     : consecutive samples sharing fire_id, ordered by date, are cut
                into rolling T-windows. Requires fire_id/date in the archive.
    """
    if strategy == "replicate":
        for s in samples:
            yield s.key, [s] * seq_len, s
        return

    if strategy != "group":
        raise ValueError(f"unknown --seq-strategy {strategy!r}")

    by_fire: Dict[str, List[NdwsSample]] = {}
    for s in samples:
        if not s.fire_id:
            continue
        by_fire.setdefault(s.fire_id, []).append(s)
    if not by_fire:
        raise RuntimeError(
            "--seq-strategy group needs fire_id/date on each sample, but none "
            "were found. Stock NDWS TFRecords do not carry them; use "
            "--seq-strategy replicate, or supply a modified archive."
        )
    for fid, group in by_fire.items():
        group.sort(key=lambda s: (s.date or "", s.index))
        if len(group) < seq_len + 1:
            continue
        for t in range(seq_len, len(group)):
            yield f"{fid}_t{t:03d}", group[t - seq_len:t], group[t]


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------
@dataclass
class EtlStats:
    samples_seen: int = 0
    tiles_written: int = 0
    skipped_empty: int = 0
    missing_dynamic: set = field(default_factory=set)
    missing_static: set = field(default_factory=set)
    errors: List[str] = field(default_factory=list)


def run_etl(
    raw_dir: Path,
    out_dir: Path,
    *,
    schema: str = "v3",
    seq_strategy: str = "replicate",
    seq_len: int = SEQ_LEN,
    max_samples: int = 0,
    dry_run: bool = False,
    keep_empty: bool = False,
) -> EtlStats:
    dyn_order, stat_order = SCHEMAS[schema]
    reader = NdwsReader(raw_dir)
    print(f"[etl_ndws] {reader.describe()}")
    print(f"[etl_ndws] schema={schema} Cd={len(dyn_order)} Cs={len(stat_order)} "
          f"seq={seq_strategy} T={seq_len}")

    stats = EtlStats()
    samples = list(reader.iter_samples(max_samples=max_samples))
    stats.samples_seen = len(samples)
    if not samples:
        raise RuntimeError(f"No NDWS samples read from {raw_dir}")

    for key, history, target in build_sequences(samples, seq_strategy, seq_len):
        try:
            dyn_frames, stat_any = [], {}
            for s in history:
                d, st, _ = map_ndws_to_channels(s.bands)
                dyn_frames.append(d)
                if not stat_any:
                    stat_any = st
            _, _, y_full = map_ndws_to_channels(target.bands)
        except Exception as e:                       # noqa: BLE001
            stats.errors.append(f"{key}: {e}")
            continue

        x_dyn = assemble_dynamic(dyn_frames, dyn_order, stats.missing_dynamic)
        x_stat = assemble_static(stat_any, stat_order, stats.missing_static)
        y = resample_to_tile(y_full, nearest=True)

        # Drop tiles with no fire in input *or* target — they are pure negatives
        # and only bloat the corpus (same rule as the TS-SatFire ingester).
        fire_idx = list(dyn_order).index("fire_t")
        if not keep_empty:
            if float((y > 0.5).mean()) == 0.0 and \
               float((x_dyn[:, fire_idx] > 0.5).mean()) == 0.0:
                stats.skipped_empty += 1
                continue

        stats.tiles_written += 1
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / f"mndws_{key}.npz",
                x_dyn=x_dyn,
                x_stat=x_stat,
                y=y,
                dyn_names=np.array(dyn_order, dtype="U16"),
                stat_names=np.array(stat_order, dtype="U16"),
            )

    # Manifest — records exactly which channels were zero-filled, so a training
    # run can never silently depend on a channel that was never populated.
    if not dry_run and stats.tiles_written:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_etl_manifest.json").write_text(json.dumps({
            "source": "mNDWS",
            "raw_dir": str(raw_dir),
            "schema": schema,
            "dyn_order": list(dyn_order),
            "stat_order": list(stat_order),
            "seq_strategy": seq_strategy,
            "seq_len": seq_len,
            "native_res_m": NDWS_RES_M,
            "output_res_m": OUT_RES_M,
            "tiles": stats.tiles_written,
            "zero_filled_dynamic": sorted(stats.missing_dynamic),
            "zero_filled_static": sorted(stats.missing_static),
            "derived_bi_from_erc": True,
            "temporally_degenerate": seq_strategy == "replicate",
        }, indent=2))
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    try:
        from ignis_ml.src.utils.paths import data_root, describe
        _DR = data_root()
        print(f"[etl_ndws] {describe()}")
    except Exception:
        _DR = Path("data")

    ap = argparse.ArgumentParser(
        description="Convert mNDWS raw -> IgnisAI 64x64 @500 m T-window tiles",
    )
    ap.add_argument("--raw", type=Path,
                    default=_DR / "mNDWS_raw" / "ndws_western_dataset")
    ap.add_argument("--out", type=Path, default=_DR / "mNDWS_500m_T6")
    ap.add_argument("--schema", choices=sorted(SCHEMAS), default="v3",
                    help="v3 = legacy 7/12 channels; v4 = 12/9 (merges with TS-SatFire)")
    ap.add_argument("--seq-strategy", choices=("replicate", "group"),
                    default="replicate",
                    help="replicate: 1 NDWS day -> T identical frames (default, "
                         "temporally degenerate). group: real multi-day windows "
                         "(needs fire_id+date in the archive).")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep tiles with no fire in input or target")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--list-only", action="store_true",
                    help="detect layout and print the first sample's bands, then exit")
    args = ap.parse_args(argv)

    if args.list_only:
        reader = NdwsReader(args.raw)
        print(f"[etl_ndws] {reader.describe()}")
        for s in reader.iter_samples(max_samples=1):
            print(f"  sample keys : {sorted(s.bands.keys())}")
            for k, v in sorted(s.bands.items()):
                print(f"    {k:16s} shape={v.shape} "
                      f"min={float(np.nanmin(v)):.3f} max={float(np.nanmax(v)):.3f}")
            print(f"  fire_id={s.fire_id} date={s.date}")
        return 0

    stats = run_etl(
        args.raw, args.out,
        schema=args.schema,
        seq_strategy=args.seq_strategy,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        keep_empty=args.keep_empty,
    )

    print("\n=== etl_ndws summary ===")
    print(f"  samples read   : {stats.samples_seen}")
    print(f"  tiles written  : {stats.tiles_written} (dry_run={args.dry_run})")
    print(f"  skipped (empty): {stats.skipped_empty}")
    if stats.missing_dynamic:
        print(f"  ZERO-FILLED dyn: {sorted(stats.missing_dynamic)}")
    if stats.missing_static:
        print(f"  ZERO-FILLED stat: {sorted(stats.missing_static)}")
    if args.seq_strategy == "replicate":
        print("  NOTE: --seq-strategy replicate produces T identical frames; "
              "temporal features carry no signal on this corpus.")
    if stats.errors:
        print(f"  errors         : {len(stats.errors)}")
        for e in stats.errors[:10]:
            print(f"    - {e}")
    return 0 if stats.tiles_written or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
