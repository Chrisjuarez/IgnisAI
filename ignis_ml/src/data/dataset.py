# src/data/dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

from .features import append_derived_features, DERIVED_FEATURE_NAMES

PathLike = Union[str, Path]


# ----------------- tiny LRU cache (for slope/aspect) -----------------
class _LRU(OrderedDict):
    def __init__(self, max_items=1024):
        super().__init__()
        self.max_items = max_items

    def get(self, k):
        if self.max_items <= 0:
            return None
        v = super().get(k)
        if v is not None:
            self.move_to_end(k)
        return v

    def put(self, k, v):
        if self.max_items <= 0:
            return
        self[k] = v
        self.move_to_end(k)
        if len(self) > self.max_items:
            self.popitem(last=False)


# ----------------- helpers -----------------
def _load_filelist(src: Union[PathLike, Iterable[PathLike]]) -> List[Path]:
    """
    Accept a .txt file (one path per line), a directory of .npz, or a python list of paths.
    Return a sorted list of absolute Paths.
    """
    if isinstance(src, (list, tuple)):
        return [Path(s).resolve() for s in src]
    p = Path(src)
    if p.is_file() and p.suffix == ".txt":
        return [Path(line.strip()).resolve()
                for line in p.read_text().splitlines() if line.strip()]
    if p.is_dir():
        return sorted([q.resolve() for q in p.glob("*.npz")])
    raise ValueError(f"Expected a .txt file, a directory of .npz, or a list; got: {src}")


def _to01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = (x - lo) / (hi - lo + 1e-8)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _to01_from_minus1_plus1(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def _compute_slope_aspect(elev: np.ndarray, res_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    slope (deg), aspect_cos, aspect_sin from elevation using central diffs.
    elev: [H,W], res_m: pixel size in meters (500 for your tiles).
    """
    dy, dx = np.gradient(elev.astype(np.float32), float(res_m), float(res_m))  # axis=(y,x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)
    # Downslope direction; with y=rows, x=cols, use -dx,-dy for descent
    aspect_rad = np.arctan2(-dx, -dy)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    return slope_deg, aspect_cos, aspect_sin


# ----------------- Dataset -----------------
class NpzTileDataset(Dataset):
    """
    Loads standardized tiles produced by your ETL (NPZ with x_dyn, x_stat, y, dyn_names, stat_names).

    Adds at load time:
      • channel normalization to ~[0,1]
      • previous-fire boost on 'fire_t' (×5, clipped)
      • slope & aspect (cos/sin) computed from elevation (+3 static bands)
      • optional vector-aware flips/rotations

    Returns:
      x_dyn:  torch.FloatTensor [T,Cd,H,W]
      x_stat: torch.FloatTensor [Cs,H,W]   (original 12 + 3 derived = 15 when expected_stat provided)
      y:      torch.FloatTensor [1,H,W]
    """

    def __init__(
        self,
        file_or_dir: Union[PathLike, Iterable[PathLike]],
        *,
        augment: bool = False,
        cache_items: int = 1024,
        expected_dyn_order: List[str] | None = None,   # e.g. ["fire_t","u","v","gust","tempC","q","precip"]
        expected_stat_order: List[str] | None = None,  # e.g. ["elev","slope","aspect_cos","aspect_sin",...]
        seq_len: int | None = None,
        res_m: float = 500.0,
        normalize: bool = True,
        fire_boost: float = 5.0,
        compute_slope_aspect: bool = True,
        # Stage-3 non-spatial augmentations. All default to 0.0 so existing
        # callers keep their old behavior when they don't pass these.
        channel_dropout_p: float = 0.0,
        # Names that must NEVER be channel-dropped. v3 protected only fire_t,
        # which let wind (u/v/gust) get zeroed ~15% of the time and taught the
        # model to under-weight wind. v4 passes [fire_t,u,v,viirs_i4].
        channel_dropout_exclude: Optional[List[str]] = None,
        gaussian_noise_std: float = 0.0,
        temporal_dropout_p: float = 0.0,
        # Stage-4 derived feature engineering.
        derived_features_enable: bool = False,
        derived_features_include: Optional[List[str]] = None,
        days_since_fire_cap: Optional[int] = None,
        # Stage-4 target reformulation. "mask" = full fire mask (original).
        # "delta" = new-pixels-only; __getitem__ returns a 4-tuple
        # (x_dyn, x_stat, y_delta, y_persist).
        target_mode: str = "mask",
    ):
        self.files = _load_filelist(file_or_dir)
        if not self.files:
            raise RuntimeError(f"No tiles found in {file_or_dir}")

        self.augment = augment
        self.cache = _LRU(max_items=int(cache_items))
        self._slope_cache = _LRU(max_items=int(cache_items))
        self.expected_dyn = expected_dyn_order
        self.expected_stat = expected_stat_order
        self.seq_len = seq_len
        self.res_m = float(res_m)
        self.normalize = normalize
        self.fire_boost = float(fire_boost)
        self.compute_slope_aspect = compute_slope_aspect
        self.channel_dropout_p = float(channel_dropout_p)
        self.channel_dropout_exclude = (
            list(channel_dropout_exclude) if channel_dropout_exclude else ["fire_t"]
        )
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.temporal_dropout_p = float(temporal_dropout_p)
        # Stage-4
        self.derived_features_enable = bool(derived_features_enable)
        self.derived_features_include = (
            list(derived_features_include) if derived_features_include is not None else None
        )
        self.days_since_fire_cap = (
            int(days_since_fire_cap) if days_since_fire_cap is not None else None
        )
        if target_mode not in ("mask", "delta"):
            raise ValueError(f"target_mode must be 'mask' or 'delta', got {target_mode!r}")
        self.target_mode = target_mode

        # Dynamic normalization ranges keyed by NAME in expected_* order
        self._dyn_ranges: Dict[str, Tuple[float, float]] = {
            "fire_t": (0, 1),       # already {0,1}
            "u": (-15, 15),         # m/s
            "v": (-15, 15),         # m/s
            "gust": (0, 25),        # m/s (cap/robust)
            "tempC": (-10, 45),     # °C
            "q": (0, 0.02),         # kg/kg
            "precip": (0, 50),      # mm/day
        }
        # Static scalers keyed by NAME
        self._stat_specs: Dict[str, Tuple[str, Tuple[float, ...] | None]] = {
            "elev": ("minmax", (0, 4000)),
            "slope": ("minmax", (0, 45)),               # degrees
            "aspect_cos": ("pm1_to_01", None),          # [-1,1] -> [0,1]
            "aspect_sin": ("pm1_to_01", None),
            "ndvi": ("minmax", (0, 1)),
            "bi": ("minmax", (0, 200)),
            "erc": ("minmax", (0, 100)),
            "pdsi": ("shift_scale", (-10, 10)),         # (x+10)/20 -> [0,1]
            "chili": ("minmax", (0, 255)),
            "impervious": ("minmax", (0, 100)),         # %
            "water": ("minmax", (0, 100)),              # %
            "population": ("log1p_div", (1000,)),       # log1p(x)/log1p(1000)
            "fuel1": ("clip_minmax", (-3, 3)),
            "fuel2": ("clip_minmax", (-3, 3)),
            "fuel3": ("clip_minmax", (-8, 3)),
        }

    def __len__(self) -> int:
        return len(self.files)

    # ---------- I/O ----------
    def _read_npz(self, path: Path):
        key = str(path)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        with np.load(path, mmap_mode="r", allow_pickle=False) as d:  # <— mmap when available
            x_dyn  = d["x_dyn"].astype(np.float32, copy=False)
            x_stat = d["x_stat"].astype(np.float32, copy=False)
            y      = d["y"].astype(np.float32, copy=False)
            dyn_names = (
                [str(s) for s in d["dyn_names"].astype(str)]
                if "dyn_names" in d.files else []
            )
            stat_names = (
                [str(s) for s in d["stat_names"].astype(str)]
                if "stat_names" in d.files else []
            )
        self.cache.put(key, (x_dyn, x_stat, y, dyn_names, stat_names))
        return x_dyn, x_stat, y, dyn_names, stat_names

    # ---------- slope/aspect (cached by file) ----------
    def _get_slope_aspect(self, path: Path, elev: np.ndarray):
        ckey = f"{path}|res={self.res_m}"
        hit = self._slope_cache.get(ckey)
        if hit is not None:
            return hit
        slope, ac, asn = _compute_slope_aspect(elev, self.res_m)
        self._slope_cache.put(ckey, (slope, ac, asn))
        return slope, ac, asn

    # ---------- normalization ----------
    def _normalize_dynamic_by_name(self, x_dyn: np.ndarray, dyn_names: List[str]) -> np.ndarray:
        """
        Reorder channels by name to match expected order (if provided) and scale to [0,1].
        Also boosts previous-fire channel (fire_t) by self.fire_boost, then clips to [0,1].
        """
        if not self.normalize:
            # Still reorder if expected order given
            if self.expected_dyn and dyn_names:
                have = {n: i for i, n in enumerate(dyn_names)}
                T, _, H, W = x_dyn.shape
                out = np.zeros((T, len(self.expected_dyn), H, W), dtype=np.float32)
                for j, name in enumerate(self.expected_dyn):
                    i = have.get(name, None)
                    if i is not None:
                        out[:, j] = x_dyn[:, i]
                x_dyn = out
            return x_dyn

        # Reorder first
        if self.expected_dyn and dyn_names:
            have = {n: i for i, n in enumerate(dyn_names)}
            T, _, H, W = x_dyn.shape
            out = np.zeros((T, len(self.expected_dyn), H, W), dtype=np.float32)
            for j, name in enumerate(self.expected_dyn):
                i = have.get(name, None)
                if i is not None:
                    out[:, j] = x_dyn[:, i]
            x_dyn = out
            dyn_names = self.expected_dyn[:]

        # Scale per expected name
        T, C, H, W = x_dyn.shape
        for c, name in enumerate(dyn_names):
            lo, hi = self._dyn_ranges.get(name, (0.0, 1.0))
            x_dyn[:, c] = _to01(x_dyn[:, c], lo, hi)

        # Boost previous fire (after scaling)
        if dyn_names and dyn_names[0] == "fire_t":
            x_dyn[:, 0] = np.clip(x_dyn[:, 0] * self.fire_boost, 0.0, 1.0)

        return x_dyn

    def _normalize_stat_by_name(self, x_stat_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize static channels per-name into approximately [0,1].
        """
        if not self.normalize:
            return {k: v.astype(np.float32) for k, v in x_stat_dict.items()}

        out: Dict[str, np.ndarray] = {}
        for name, arr in x_stat_dict.items():
            mode, spec = self._stat_specs.get(name, ("minmax", (0.0, 1.0)))
            a = arr.astype(np.float32)
            if mode == "minmax":
                lo, hi = spec  # type: ignore
                a = _to01(a, float(lo), float(hi))
            elif mode == "clip_minmax":
                lo, hi = spec  # type: ignore
                a = np.clip(a, float(lo), float(hi)).astype(np.float32)
                a = _to01(a, float(lo), float(hi))
            elif mode == "pm1_to_01":
                a = _to01_from_minus1_plus1(a)
            elif mode == "shift_scale":
                lo, hi = spec  # type: ignore
                a = (a - float(lo)) / (float(hi) - float(lo) + 1e-8)
                a = np.clip(a, 0.0, 1.0).astype(np.float32)
            elif mode == "log1p_div":
                (denom,) = spec  # type: ignore
                a = np.log1p(np.maximum(a, 0.0)) / np.log1p(float(denom))
                a = np.clip(a, 0.0, 1.0).astype(np.float32)
            else:
                # fallback robust scaling by per-tile percentiles
                p1, p99 = np.percentile(a, [1, 99])
                if p99 <= p1:
                    p99 = p1 + 1e-6
                a = np.clip((a - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)
            out[name] = a
        return out

    # ---------- temporal alignment ----------
    def _align_seq(self, x_dyn: np.ndarray) -> np.ndarray:
        """
        Ensure the dynamic sequence has exactly self.seq_len timesteps.
        If longer: take the last S. If shorter: repeat the last frame.
        """
        if self.seq_len is None:
            return x_dyn
        T, C, H, W = x_dyn.shape
        S = int(self.seq_len)
        if T == S:
            return x_dyn
        if T > S:
            return x_dyn[-S:]
        # pad by repeating last
        pad = np.repeat(x_dyn[-1:, :, :, :], repeats=(S - T), axis=0)
        return np.concatenate([x_dyn, pad], axis=0)

    # ---------- augmentation ----------
    def _augment(self, x_dyn: np.ndarray, x_stat: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vector-aware spatial aug on *normalized* winds.
        Because u,v were mapped from [-15,15] -> [0,1], negation is (1 - value).
        """
        # indices for u,v after _normalize_dynamic_by_name()
        u_idx = v_idx = None
        if self.expected_dyn:
            try:
                u_idx = self.expected_dyn.index("u")
                v_idx = self.expected_dyn.index("v")
            except ValueError:
                u_idx = v_idx = None  # winds not present

        # ----- Horizontal flip (mirror over vertical axis) -----
        if np.random.rand() < 0.5:
            x_dyn = x_dyn[..., :, ::-1].copy()   # [T,C,H,W]
            x_stat = x_stat[..., ::-1].copy()    # [Cs,H,W]
            y = y[..., ::-1].copy()              # [H,W]
            if u_idx is not None:
                # u' = -u  ->  u_norm' = 1 - u_norm
                x_dyn[:, u_idx, :, :] = 1.0 - x_dyn[:, u_idx, :, :]

        # ----- Vertical flip (mirror over horizontal axis) -----
        if np.random.rand() < 0.5:
            x_dyn = x_dyn[..., ::-1, :].copy()
            x_stat = x_stat[..., ::-1, :].copy()
            y = y[..., ::-1, :].copy()
            if v_idx is not None:
                # v' = -v  ->  v_norm' = 1 - v_norm
                x_dyn[:, v_idx, :, :] = 1.0 - x_dyn[:, v_idx, :, :]

        # ----- Random 90° rotation (CCW; k ∈ {0,1,2,3}) -----
        k = np.random.randint(0, 4)
        if k:
            x_dyn = np.rot90(x_dyn, k, axes=(-2, -1)).copy()
            x_stat = np.rot90(x_stat, k, axes=(-2, -1)).copy()
            y = np.rot90(y, k).copy()

            if (u_idx is not None) and (v_idx is not None):
                u = x_dyn[:, u_idx].copy()
                v = x_dyn[:, v_idx].copy()
                # In normalized space:
                # 90°:  u'=-v -> 1-v,   v'= u
                # 180°: u'=-u -> 1-u,   v'=-v -> 1-v
                # 270°: u'= v,          v'=-u -> 1-u
                if   k == 1:
                    u2, v2 = (1.0 - v),  u
                elif k == 2:
                    u2, v2 = (1.0 - u), (1.0 - v)
                elif k == 3:
                    u2, v2 =  v, (1.0 - u)
                else:
                    u2, v2 =  u,  v
                x_dyn[:, u_idx] = u2
                x_dyn[:, v_idx] = v2

        # ----- Channel dropout (Stage 3) -----
        # Zero out non-fire dynamic channels with prob p. fire_t (assumed index 0
        # in expected_dyn_order) is NEVER dropped — it is the dominant signal.
        if self.channel_dropout_p > 0.0 and x_dyn.shape[1] > 1:
            T, C, H, W = x_dyn.shape
            # Protect every name in channel_dropout_exclude (v4: fire_t,u,v,
            # viirs_i4). Map names -> indices via expected_dyn; if no expected
            # order is known, fall back to protecting channel 0 (fire_t).
            protect = set()
            if self.expected_dyn is not None:
                for nm in self.channel_dropout_exclude:
                    try:
                        protect.add(self.expected_dyn.index(nm))
                    except ValueError:
                        pass
            else:
                protect.add(0)
            drop_mask = np.random.rand(C) < self.channel_dropout_p
            for idx in protect:
                if 0 <= idx < C:
                    drop_mask[idx] = False
            if drop_mask.any():
                # zero entire channel across all timesteps + spatial extent
                x_dyn[:, drop_mask, :, :] = 0.0

        # ----- Gaussian noise on dynamic inputs (Stage 3) -----
        # Adds N(0, sigma) in normalized [0,1] space, then clips. Applied to ALL
        # dynamic channels including fire_t — a little label noise on the prior
        # fire mask is beneficial (keeps the model from pure persistence).
        if self.gaussian_noise_std > 0.0:
            noise = np.random.normal(
                0.0, self.gaussian_noise_std, size=x_dyn.shape
            ).astype(np.float32)
            x_dyn = np.clip(x_dyn + noise, 0.0, 1.0)

        # ----- Temporal dropout (Stage 3) -----
        # Randomly zero out non-last timesteps with prob p. The LAST frame is
        # always kept because (a) the model uses it for skip features via
        # d1_last/d2_last, and (b) it's the most predictive frame.
        if self.temporal_dropout_p > 0.0 and x_dyn.shape[0] > 1:
            T = x_dyn.shape[0]
            # Only consider frames 0 .. T-2 for dropout.
            drop_mask = np.random.rand(T - 1) < self.temporal_dropout_p
            for t in range(T - 1):
                if drop_mask[t]:
                    x_dyn[t, :, :, :] = 0.0

        return x_dyn, x_stat, y

    # ---------- delta helpers ----------
    def _fire_channel_index(self) -> int:
        """Index of fire_t in expected_dyn_order (or 0 by convention)."""
        if self.expected_dyn:
            try:
                return self.expected_dyn.index("fire_t")
            except ValueError:
                return 0
        return 0

    # ---------- main ----------
    def __getitem__(self, idx: int):
        path = self.files[idx]
        x_dyn, x_stat_raw, y, dyn_names, stat_names = self._read_npz(path)

        # Normalize & reorder dynamics by NAME (into expected order if given)
        x_dyn = self._normalize_dynamic_by_name(x_dyn, dyn_names)
        x_dyn = self._align_seq(x_dyn)  # [T,Cd,H,W]

        # Build a dict name->array for statics using names in file (or fallback guess)
        stat_map: Dict[str, np.ndarray] = {}
        if stat_names:
            for i, n in enumerate(stat_names):
                stat_map[n] = x_stat_raw[i]
        else:
            # Fallback to the original ETL order (no slope/aspect here yet)
            keys_guess = ["elev", "ndvi", "bi", "erc", "pdsi", "chili",
                          "impervious", "water", "population", "fuel1", "fuel2", "fuel3"]
            for i, n in enumerate(keys_guess[: x_stat_raw.shape[0]]):
                stat_map[n] = x_stat_raw[i]

        # Compute slope/aspect from elevation (and cache)
        if self.compute_slope_aspect:
            elev = stat_map.get("elev", x_stat_raw[0])
            slope, ac, asn = self._get_slope_aspect(path, elev)
            stat_map["slope"] = slope
            stat_map["aspect_cos"] = ac
            stat_map["aspect_sin"] = asn

        # Normalize statics by NAME, then assemble in expected order (preferred)
        if self.expected_stat:
            stat_map = self._normalize_stat_by_name(stat_map) if self.normalize else {k: v.astype(np.float32) for k, v in stat_map.items()}
            stat_list: List[np.ndarray] = []
            for name in self.expected_stat:
                arr = stat_map.get(name)
                if arr is None:
                    # if missing, insert zeros with same H,W as an existing band
                    H, W = next(iter(stat_map.values())).shape
                    arr = np.zeros((H, W), dtype=np.float32)
                stat_list.append(arr.astype(np.float32))
            x_stat = np.stack(stat_list, axis=0)
        else:
            # normalize in raw order and return
            stat_map = self._normalize_stat_by_name(stat_map) if self.normalize else {k: v.astype(np.float32) for k, v in stat_map.items()}
            order = list(stat_map.keys())
            x_stat = np.stack([stat_map[k] for k in order], axis=0)

        # Augment (vector-aware). Runs on BASE channels only — derived features
        # are appended AFTER augmentation so they always reflect the final
        # augmented inputs and match what the inference path will compute.
        if self.augment:
            x_dyn, x_stat, y = self._augment(x_dyn, x_stat, y)

        # Stage-4: derive delta target from the (augmented) last fire frame.
        # We do this BEFORE appending derived features so index bookkeeping
        # stays simple; derived features don't affect y/y_persist.
        fire_idx = self._fire_channel_index()
        fire_last = (x_dyn[-1, fire_idx] >= 0.5).astype(np.float32)   # [H,W]
        y_persist = fire_last                                          # [H,W]
        y_delta = np.clip(y - y_persist, 0.0, 1.0).astype(np.float32)  # [H,W]

        # Stage-4: append derived dynamic features (shared w/ inference).
        if self.derived_features_enable:
            dyn_order_now = list(self.expected_dyn) if self.expected_dyn else list(dyn_names)
            x_dyn, _new_order = append_derived_features(
                x_dyn,
                dyn_order=dyn_order_now,
                include=self.derived_features_include,
                days_since_fire_cap=self.days_since_fire_cap,
            )

        # to tensors
        x_dyn_t    = torch.from_numpy(x_dyn).float()                  # [T,Cd(+k),H,W]
        x_stat_t   = torch.from_numpy(x_stat).float()                 # [Cs,H,W]
        y_t        = torch.from_numpy(y).unsqueeze(0).float()         # [1,H,W]

        if self.target_mode == "delta":
            y_delta_t   = torch.from_numpy(y_delta).unsqueeze(0).float()    # [1,H,W]
            y_persist_t = torch.from_numpy(y_persist).unsqueeze(0).float()  # [1,H,W]
            return x_dyn_t, x_stat_t, y_delta_t, y_persist_t

        return x_dyn_t, x_stat_t, y_t
