#!/usr/bin/env python3
"""Visual + directional validation of a trained checkpoint on a preset event.

Why this exists
---------------
AP, CSI and IoU are all DIRECTION-BLIND. A model that grows the fire by the
right amount in the wrong direction scores respectably on every one of them and
is operationally worthless — evacuate the wrong neighbourhood. That is exactly
how v3 failed on Palisades: it pushed east along the urban edge instead of
southwest with the Santa Ana wind, and no metric in the training loop noticed.

So before wiring any checkpoint into tilesvc, look at it:

  * predicted heatmap against the observed perimeter
  * wind quiver over the top
  * cos(alignment) between predicted growth and the driving wind

It builds inputs through the SAME modules tilesvc serves with
(`dynamic_builder`, `static_catalog`, `grid`), so this is a mirror of
production rather than a parallel implementation. It deliberately does NOT go
through `app.py`, whose `_validate_model_contract` would reject a checkpoint
whose seq_len differs from the deployed config — which is the case we are
trying to evaluate.

Offline-capable: `.cache/runtime_cache/palisades/` holds HRRR grids and FIRMS
snapshots for the event window, so no network is needed for that preset.

Usage
-----
    python -m ignis_ml.scripts.validate_checkpoint \
        --ckpt "$IGNIS_DATA_ROOT/models_gpu/convlstm_unet_control_delta_Cd13_Cs15_H64_T3.pt" \
        --event palisades --threshold 0.85 --out /tmp/palisades_control.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: Physical -> [0,1] ranges. MUST match NpzTileDataset._dyn_ranges (training)
#: and app.py::_DYN_RANGES (serving). `build_dynamic_for_tile` returns RAW
#: physical units; normalization happens downstream, so a validator that feeds
#: the builder's output straight to the model is scoring garbage.
DYN_RANGES: Dict[str, Tuple[float, float]] = {
    "fire_t": (0.0, 1.0),
    "u": (-15.0, 15.0),
    "v": (-15.0, 15.0),
    "gust": (0.0, 25.0),
    "tempC": (-10.0, 45.0),
    "q": (0.0, 0.02),
    "precip": (0.0, 50.0),
}
FIRE_BOOST = 5.0        # app.py and NpzTileDataset both apply this to fire_t


def normalize_dynamic(dyn_phys: np.ndarray, order: List[str]) -> np.ndarray:
    """Replicate app.py::_prepare_dynamic_for_model's pre-derived-feature step."""
    x = np.empty_like(dyn_phys, dtype=np.float32)
    for i, name in enumerate(order):
        lo, hi = DYN_RANGES.get(name, (0.0, 1.0))
        x[:, i] = np.clip((dyn_phys[:, i] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    if order and order[0] == "fire_t":
        x[:, 0] = np.clip(x[:, 0] * FIRE_BOOST, 0.0, 1.0)
    return x


EVENTS: Dict[str, dict] = {
    "palisades": {"name": "Palisades Fire", "lat": 34.078, "lon": -118.555,
                  "ref_time": "2025-01-07T18:30:00Z", "ignition": True,
                  "expect": "southwest — offshore Santa Ana toward the ocean"},
    "eaton":     {"name": "Eaton Fire", "lat": 34.1897, "lon": -118.1300,
                  "ref_time": "2025-01-07T22:30:00Z", "ignition": True,
                  "expect": "south/southwest into the Altadena foothills"},
    "camp":      {"name": "Camp Fire", "lat": 39.7596, "lon": -121.6219,
                  "ref_time": "2018-11-08T14:30:00Z", "ignition": True,
                  "expect": "southwest — Jarbo Gap downslope wind toward Paradise"},
    "dixie":     {"name": "Dixie Fire", "lat": 39.8760, "lon": -121.3870,
                  "ref_time": "2021-07-14T17:00:00Z", "ignition": True, "expect": ""},
    "caldor":    {"name": "Caldor Fire", "lat": 38.5900, "lon": -120.5400,
                  "ref_time": "2021-08-14T18:00:00Z", "ignition": True, "expect": ""},
}


# ---------------------------------------------------------------------------
def centroid(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask)
    return idx.mean(axis=0)[::-1] if idx.size else np.array([np.nan, np.nan])  # (x, y)


def directional_alignment(growth: np.ndarray, prior: np.ndarray,
                          u_mean: float, v_mean: float) -> float:
    """cos angle between predicted-growth displacement and the wind vector.

    +1 = growth exactly downwind, -1 = exactly upwind. The v3 Palisades failure
    would have shown up here as a strongly negative or orthogonal value while
    IoU looked acceptable.
    """
    c_prior, c_grow = centroid(prior), centroid(growth)
    if np.isnan(c_prior).any() or np.isnan(c_grow).any():
        return float("nan")
    d = c_grow - c_prior
    w = np.array([u_mean, -v_mean])          # +v is north; image y grows south
    nd, nw = np.linalg.norm(d), np.linalg.norm(w)
    return float(d @ w / (nd * nw)) if nd > 1e-6 and nw > 1e-6 else float("nan")


def weighted_alignment(prob: np.ndarray, prior: np.ndarray,
                       u_mean: float, v_mean: float) -> float:
    """Threshold-FREE directional alignment, weighted by probability mass.

    A single thresholded cos is unreliable: on Palisades the control checkpoint
    gives +0.766 at 0.85 (2 pixels) and -0.488 at 0.50 (92 pixels). The peak
    leans downwind while the mass leans upwind, so the answer depends entirely
    on where you cut. This uses the probability-weighted centroid of the
    NON-prior area, which uses the whole field and has no cut point.
    """
    p = prob.copy()
    p[prior] = 0.0                     # growth only, exclude the existing fire

    # Drop negligible mass before weighting. A concentrated model leaves a
    # diffuse near-zero background over the whole tile; on control60 that was
    # ~4,000 cells at p~0.001 against ~40 real prediction pixels, which dragged
    # the centroid to the tile centre and produced a spurious cos of -0.095
    # while every threshold from 0.2 to 0.9 said -0.67 or worse. Weighting must
    # describe the prediction, not the background.
    floor = max(0.05, 0.05 * float(p.max()))
    p[p < floor] = 0.0

    tot = p.sum()
    if tot <= 1e-9:
        return float("nan")
    yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]]
    c_grow = np.array([(p * xx).sum() / tot, (p * yy).sum() / tot])
    c_prior = centroid(prior)
    if np.isnan(c_prior).any():
        return float("nan")
    d = c_grow - c_prior
    w = np.array([u_mean, -v_mean])
    nd, nw = np.linalg.norm(d), np.linalg.norm(w)
    return float(d @ w / (nd * nw)) if nd > 1e-6 and nw > 1e-6 else float("nan")


def alignment_sweep(prob: np.ndarray, prior: np.ndarray,
                    u_mean: float, v_mean: float,
                    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    ) -> List[Tuple[float, int, float]]:
    """(threshold, n_pixels, cos) so threshold-dependence is visible, not hidden."""
    out = []
    for t in thresholds:
        g = (prob >= t) & ~prior
        out.append((t, int(g.sum()), directional_alignment(g, prior, u_mean, v_mean)))
    return out


def compass(u: float, v: float) -> str:
    if abs(u) < 1e-9 and abs(v) < 1e-9:
        return "calm"
    ang = (np.degrees(np.arctan2(u, v)) + 360) % 360      # direction wind blows TOWARD
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((ang + 11.25) % 360 // 22.5)]


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visual/directional checkpoint validation")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--event", default="palisades", choices=sorted(EVENTS))
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: best_threshold stored in the checkpoint")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="default: seq_len stored in the checkpoint")
    ap.add_argument("--out", type=Path, default=Path("/tmp/checkpoint_validation.png"))
    ap.add_argument("--no-plot", action="store_true",
                    help="Report the numbers only. For headless runs with no plotting stack.")
    args = ap.parse_args(argv)

    import torch
    os.chdir(_REPO)

    # Point FIRMS at the cached snapshots for this event. Without it the builder
    # falls back to the live NRT API, which only covers ~5 days — so a 2025 event
    # returns ZERO fire pixels and the model predicts on fuel/weather alone.
    snap = _REPO / ".cache" / "runtime_cache" / args.event / "firms_snapshots"
    if snap.is_dir() and not os.environ.get("FIRMS_SNAPSHOT_DIR"):
        os.environ["FIRMS_SNAPSHOT_DIR"] = str(snap)
        os.environ["FIRMS_SNAPSHOT_REQUIRED"] = "1"
        print(f"FIRMS snapshots: {snap}")
    elif not snap.is_dir():
        print(f"no FIRMS snapshot cache at {snap} — relying on ignition seed only")

    # Gridded NOAA/HRRR weather. WITHOUT this the builder silently falls back to
    # the Open-Meteo archive, which for Palisades 2025-01-07 returns ~0.8-7 m/s
    # winds. The cached HRRR for the same hour has u=-4.8, v=-5.0, gusts to
    # 24 m/s — i.e. the actual Santa Ana. Validating on the fallback tests the
    # model against a calm day and says nothing about wind-driven spread.
    # (This is the same gap docs/v4-retraining-gameplan.md Phase 6 flags as
    #  "weatherQuality.source: noaa_hrrr, not open_meteo_fallback".)
    noaa = _REPO / ".cache" / "runtime_cache" / args.event / "noaa_grid_cache"
    if noaa.is_dir() and not os.environ.get("NOAA_GRID_CACHE_DIR"):
        os.environ["NOAA_GRID_CACHE_DIR"] = str(noaa)
        os.environ["NOAA_GRIB_ENABLED"] = "1"
        print(f"NOAA grid cache : {noaa}")
    elif not noaa.is_dir():
        print(f"⚠ no NOAA grid cache at {noaa} — weather will fall back to the "
              f"Open-Meteo archive, which understates extreme-wind events")

    from services.tilesvc.grid import lonlat_to_tile, tile_affine, tile_bounds_lonlat
    from services.tilesvc.dynamic_builder import build_dynamic_for_tile
    from services.tilesvc.static_catalog import load_static_tensor_for_model
    from ignis_ml.src.data.features import append_derived_features
    from ignis_ml.src.models.convlstm_unet import ConvLSTMUNet

    ev = EVENTS[args.event]
    ref_time = dt.datetime.fromisoformat(ev["ref_time"].replace("Z", "+00:00"))

    # ---- checkpoint drives the geometry; never assume the deployed config ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    Cd, Cs = int(ck["cd"]), int(ck["cs"])
    hidden = int(ck.get("hidden", 64))
    seq_len = args.seq_len or int(ck.get("seq_len", 6))
    thr = args.threshold if args.threshold is not None else float(ck.get("best_threshold", 0.5))
    dyn_order = list(ck.get("dyn_order") or
                     ["fire_t", "u", "v", "gust", "tempC", "q", "precip"])
    stat_order = list(ck.get("stat_order") or [])

    print(f"event      : {ev['name']}  ({ev['lat']}, {ev['lon']})  {ev['ref_time']}")
    print(f"checkpoint : {args.ckpt.name}")
    print(f"  arch={ck.get('arch_version')} Cd={Cd} Cs={Cs} hidden={hidden} "
          f"T={seq_len} target={ck.get('target_mode')}")
    print(f"  val_ap={ck.get('val_ap')} val_csi={ck.get('val_csi')} threshold={thr}")
    if "split_stats" in ck:
        print(f"  split={ck['split_stats'].get('split')}")
    if ev["expect"]:
        print(f"expected spread: {ev['expect']}")

    # ---- inputs, via the real serving builders -----------------------------
    print("\nbuilding inputs (tilesvc modules)...")
    x_dyn = build_dynamic_for_tile(
        ev["lat"], ev["lon"], T_seq=seq_len, hours_step=24,
        ignition=ev["ignition"], ref_time=ref_time, channel_order=dyn_order[:7],
    )
    x_dyn = np.asarray(x_dyn, dtype=np.float32)          # [T, 7, 64, 64]
    tile = lonlat_to_tile(ev["lon"], ev["lat"])
    stat, stat_summary = load_static_tensor_for_model(tile, stat_order)
    stat = np.asarray(stat, dtype=np.float32)
    print(f"  x_dyn {x_dyn.shape}  x_stat {stat.shape}")

    # Wind read from the RAW builder output, which is already in m/s.
    u_ms = float(x_dyn[:, 1].mean())
    v_ms = float(x_dyn[:, 2].mean())
    print(f"  mean wind: u={u_ms:+.2f} v={v_ms:+.2f} m/s -> blowing toward "
          f"{compass(u_ms, v_ms)}")
    if max(abs(u_ms), abs(v_ms)) > 60:
        print("  ⚠ implausible wind magnitude — the builder output is probably "
              "already normalized, or channel order is wrong")

    # Normalize BEFORE derived features, exactly as app.py does. Order matters:
    # the derived channels (wind_speed, wind_dir_cos/sin, ...) are computed from
    # NORMALIZED u,v, so appending them to raw values silently corrupts them.
    x_dyn = normalize_dynamic(x_dyn, dyn_order[:7])
    print(f"  normalized to [0,1] (+ fire_t x{FIRE_BOOST:.0f} boost)")

    if x_dyn.shape[1] < Cd:
        x_dyn, _ = append_derived_features(x_dyn, dyn_order=dyn_order[:7],
                                           include=None, days_since_fire_cap=None)
        print(f"  + derived -> {x_dyn.shape}")
    if x_dyn.shape[1] != Cd:
        print(f"  ERROR: got Cd={x_dyn.shape[1]}, checkpoint wants {Cd}")
        return 1

    # ---- inference ---------------------------------------------------------
    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=hidden, drop=0.0, drop_decoder=0.0)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_dyn[None]).float(),
                       torch.from_numpy(stat[None]).float())
        prob = torch.sigmoid(logits)[0, 0].numpy()

    fire_prior = x_dyn[-1, 0] >= 0.5
    growth = prob >= thr
    cos_al = directional_alignment(growth, fire_prior, u_ms, v_ms)

    cos_w = weighted_alignment(prob, fire_prior, u_ms, v_ms)
    sweep = alignment_sweep(prob, fire_prior, u_ms, v_ms)

    print(f"\nprediction : min={prob.min():.4f} mean={prob.mean():.4f} max={prob.max():.4f}")
    print(f"  prior fire pixels : {int(fire_prior.sum())}")
    print(f"  predicted growth  : {int(growth.sum())} @ threshold {thr}")

    print(f"\n  directional alignment vs threshold")
    print(f"  {'thresh':>7}{'pixels':>8}{'cos':>8}")
    for t, n, c in sweep:
        flag = "" if not np.isfinite(c) else ("  downwind" if c > 0.3 else
                                              "  UPWIND" if c < -0.3 else "  cross")
        print(f"  {t:>7.2f}{n:>8d}{c:>8.3f}{flag}")

    # The threshold-free number is the one to trust: a thresholded cos computed
    # from a handful of pixels is noise, and its sign can flip with the cut.
    print(f"\n  probability-weighted cos (threshold-free): {cos_w:+.3f}")
    if np.isfinite(cos_w):
        verdict = ("DOWNWIND — probability mass leans with the wind" if cos_w > 0.3 else
                   "CROSSWIND — mass roughly perpendicular to the wind" if cos_w > -0.3 else
                   "UPWIND — probability mass OPPOSES the wind; the v3 failure mode")
        print(f"  -> {verdict}")

    finite = [c for _, n, c in sweep if np.isfinite(c) and n >= 10]
    if finite and (max(finite) > 0.3 > min(finite) or min(finite) < -0.3 < max(finite)):
        print("  ⚠ sign flips across thresholds — the peak and the mass disagree.")
        print("    Trust the weighted value; a single thresholded cos is not evidence.")

    # ---- figure ------------------------------------------------------------
    # The numbers above are the evidence; the figure is a convenience. Skip it
    # when matplotlib is absent so this runs headless — the serving image does
    # not carry a plotting stack, and a Render one-off job is the only place
    # with credentials for the static COGs.
    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib unavailable — skipping figure; the numbers above stand)")
        return 0

    w, s, e, n = tile_bounds_lonlat(tile)
    extent = (w, e, s, n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(fire_prior, extent=extent, origin="upper",
                   interpolation="nearest", cmap="Reds")
    axes[0].set_title(f"prior fire (last of T={seq_len})")

    im = axes[1].imshow(prob, extent=extent, origin="upper",
                        interpolation="nearest", cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("predicted delta probability")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(prob, extent=extent, origin="upper",
                   interpolation="nearest", cmap="inferno", vmin=0, vmax=1)
    axes[2].contour(np.flipud(fire_prior.astype(float)), levels=[0.5],
                    colors="cyan", linewidths=1.2, extent=extent)
    # single arrow for mean wind, anchored at the tile centre
    cx, cy = (w + e) / 2, (s + n) / 2
    scale = (e - w) * 0.35 / max(np.hypot(u_ms, v_ms), 1e-6)
    axes[2].arrow(cx, cy, u_ms * scale, v_ms * scale, color="white",
                  width=(e - w) * 0.004, head_width=(e - w) * 0.02, alpha=0.9)
    axes[2].set_title(f"prediction + prior (cyan) + wind\ncos={cos_al:+.3f}")

    for a in axes:
        a.plot(ev["lon"], ev["lat"], "w*", ms=14, mec="k", mew=0.6)
        a.set_xlabel("lon"); a.set_ylabel("lat")
    fig.suptitle(f"{ev['name']} — {args.ckpt.name}  (threshold {thr})", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"\nfigure -> {args.out}")

    summary = {
        "event": args.event, "checkpoint": args.ckpt.name,
        "threshold": thr, "seq_len": seq_len,
        "val_ap": ck.get("val_ap"), "val_csi": ck.get("val_csi"),
        "wind_u_ms": float(u_ms), "wind_v_ms": float(v_ms),
        "wind_toward": compass(u_ms, v_ms),
        "prior_fire_px": int(fire_prior.sum()),
        "growth_px": int(growth.sum()),
        "cos_alignment_at_threshold": None if not np.isfinite(cos_al) else cos_al,
        "cos_alignment_weighted": None if not np.isfinite(cos_w) else cos_w,
        "alignment_sweep": [
            {"threshold": t, "pixels": n,
             "cos": None if not np.isfinite(c) else round(c, 4)}
            for t, n, c in sweep
        ],
        "prob_max": float(prob.max()),
        "prob_mean": float(prob.mean()),
    }
    js = args.out.with_suffix(".json")
    js.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
