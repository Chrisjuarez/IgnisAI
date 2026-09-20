"""Authoritative WildfireSpreadTS (WSTS) channel specification.

Transcribed from the reference implementation so the research track never
guesses at band order:

    SebastianGer/WildfireSpreadTS
      src/dataloader/FireSpreadDataset.py
        :: map_channel_index_to_features()
        :: get_static_and_dynamic_feature_ids()

Getting this wrong is silent and catastrophic — the model would run, produce
plausible-looking output, and be wrong. Everything in research/wsts/ imports
band order from here rather than hardcoding it.

Two representations matter
--------------------------
BASE (23 bands)  — what lives in the GeoTIFF / HDF5 on disk.
MODEL (40 chans) — what the network actually receives, after the dataset's
                   preprocessing expands landcover to one-hot and appends a
                   binary active-fire mask.

The MODEL layout is produced by `preprocess_and_augment`:

    x[:, :16]            base features 0..15          -> model 0..15
    one_hot(landcover)   base feature 16, 17 classes  -> model 16..32
    x[:, 17:]            base features 17..22         -> model 33..38
    binary_af_mask       (base 22 > 0)                -> model 39

so 16 + 17 + 6 + 1 = 40.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# BASE: 23 bands, GeoTIFF order
# ---------------------------------------------------------------------------
BASE_FEATURES: Tuple[str, ...] = (
    "VIIRS band M11",                       # 0
    "VIIRS band I2",                        # 1
    "VIIRS band I1",                        # 2
    "NDVI",                                 # 3
    "EVI2",                                 # 4
    "Total precipitation",                  # 5
    "Wind speed",                           # 6
    "Wind direction",                       # 7   degrees -> sin() applied
    "Minimum temperature",                  # 8
    "Maximum temperature",                  # 9
    "Energy release component",             # 10
    "Specific humidity",                    # 11
    "Slope",                                # 12  static
    "Aspect",                               # 13  static, degrees -> sin()
    "Elevation",                            # 14  static
    "Palmer drought severity index (PDSI)",  # 15  NOTE: WSTS treats as DYNAMIC
    "Landcover class",                      # 16  static, one-hot expanded
    "Forecast: Total precipitation",        # 17
    "Forecast: Wind speed",                 # 18
    "Forecast: Wind direction",             # 19  degrees -> sin()
    "Forecast: Temperature",                # 20
    "Forecast: Specific humidity",          # 21
    "Active fire",                          # 22  detection time hhmm -> hh
)
N_BASE = len(BASE_FEATURES)                 # 23

LANDCOVER_CLASSES: Tuple[str, ...] = (
    "Evergreen Needleleaf Forests", "Evergreen Broadleaf Forests",
    "Deciduous Needleleaf Forests", "Deciduous Broadleaf Forests",
    "Mixed Forests", "Closed Shrublands", "Open Shrublands",
    "Woody Savannas", "Savannas", "Grasslands", "Permanent Wetlands",
    "Croplands", "Urban and Built-up Lands",
    "Cropland/Natural Vegetation Mosaics", "Permanent Snow and Ice",
    "Barren", "Water Bodies",
)                                            # MODIS IGBP, 1-indexed on disk
N_LANDCOVER = len(LANDCOVER_CLASSES)         # 17

BAND_LANDCOVER = 16
BAND_ACTIVE_FIRE = 22
#: Bands in degrees; the dataset applies sin(deg2rad(x)) so 0deg and 360deg are close.
DEGREE_BANDS: Tuple[int, ...] = (7, 13, 19)


def model_channel_names() -> List[str]:
    """The 40 channel names in the order the network sees them."""
    return (
        list(BASE_FEATURES[:16])
        + [f"Land cover: {c}" for c in LANDCOVER_CLASSES]
        + list(BASE_FEATURES[17:])
        + ["Active fire (binary)"]
    )


N_MODEL = 40

#: Verbatim from FireSpreadDataset.get_static_and_dynamic_feature_ids().
STATIC_MODEL_IDS: Tuple[int, ...] = tuple([12, 13, 14] + list(range(16, 33)))
DYNAMIC_MODEL_IDS: Tuple[int, ...] = tuple(
    list(range(12)) + [15] + list(range(33, 40)))


# ---------------------------------------------------------------------------
# Mapping IgnisAI's existing sources onto WSTS bands
# ---------------------------------------------------------------------------
#: status:
#:   "have"    — IgnisAI already has this layer (S3 static rasters or the
#:               tilesvc runtime pipeline); needs resampling to 375 m only.
#:   "derive"  — computable from something IgnisAI has.
#:   "missing" — must be newly sourced for the WSTS track.
SOURCE_MAP: Dict[int, Dict[str, str]] = {
    0:  {"status": "missing", "source": "VIIRS VNP09/VJ109 band M11 (2.25um SWIR)",
         "note": "Earthdata. Not in any current IgnisAI pipeline."},
    1:  {"status": "missing", "source": "VIIRS band I2 (0.865um NIR)",
         "note": "Earthdata."},
    2:  {"status": "missing", "source": "VIIRS band I1 (0.640um red)",
         "note": "Earthdata."},
    3:  {"status": "have", "source": "s3://.../source-data/ndvi/noaa_star_smn_ndvi_*_500m.tif",
         "note": "Have at 500 m; resample to 375 m. WSTS NDVI is per-timestep, "
                 "IgnisAI's is a fire-season composite — a real fidelity gap."},
    4:  {"status": "derive", "source": "EVI2 = 2.5*(NIR-Red)/(NIR+2.4*Red+1)",
         "note": "Needs VIIRS I1/I2, so blocked on bands 1-2."},
    5:  {"status": "have", "source": "GridMET pr / HRRR runtime cache",
         "note": "tilesvc already fetches precipitation."},
    6:  {"status": "have", "source": "HRRR/GridMET wind speed",
         "note": "IgnisAI stores u,v — convert back to speed/direction."},
    7:  {"status": "have", "source": "HRRR/GridMET wind direction",
         "note": "IgnisAI stores u,v. WSTS wants DEGREES (sin applied later), "
                 "so do NOT pass u,v here."},
    8:  {"status": "have", "source": "GridMET tmmn", "note": "Kelvin in NDWS; check units."},
    9:  {"status": "have", "source": "GridMET tmmx", "note": ""},
    10: {"status": "have", "source": "s3://.../gridmet/gridmet_erc_fireseason_mean_2024_500m.tif",
         "note": "Have a fire-season MEAN; WSTS wants per-day ERC."},
    11: {"status": "have", "source": "GridMET sph / HRRR humidity", "note": ""},
    12: {"status": "derive", "source": "slope from SRTM DEM",
         "note": "dataset.py::_compute_slope_aspect already does this."},
    13: {"status": "derive", "source": "aspect from SRTM DEM",
         "note": "WSTS wants DEGREES, not the aspect_cos/aspect_sin pair "
                 "IgnisAI uses. Convert back."},
    14: {"status": "have", "source": "s3://.../topography/srtm_dem_western_conus_500m.tif",
         "note": ""},
    15: {"status": "have", "source": "s3://.../gridmet/gridmet_pdsi_fireseason_mean_2024_500m.tif",
         "note": "IgnisAI treats PDSI as static; WSTS treats it as dynamic."},
    16: {"status": "derive", "source": "s3://.../nlcd/nlcd_lndcov_western_conus_2024_500m.tif",
         "note": "NLCD classes must be remapped to MODIS IGBP 17-class scheme. "
                 "Lossy and needs a documented crosswalk."},
    17: {"status": "have", "source": "HRRR forecast precipitation",
         "note": "tilesvc fetches HRRR forecasts already."},
    18: {"status": "have", "source": "HRRR forecast wind speed", "note": ""},
    19: {"status": "have", "source": "HRRR forecast wind direction", "note": "degrees"},
    20: {"status": "have", "source": "HRRR forecast temperature", "note": ""},
    21: {"status": "have", "source": "HRRR forecast specific humidity", "note": ""},
    22: {"status": "have", "source": "FIRMS/VIIRS active fire",
         "note": "WSTS encodes DETECTION TIME (hhmm -> hh), not a binary mask. "
                 "IgnisAI's fire_t is binary — the time channel must be rebuilt."},
}


def gap_report() -> Dict[str, List[str]]:
    """Group WSTS bands by how much work IgnisAI needs to supply them."""
    out: Dict[str, List[str]] = {"have": [], "derive": [], "missing": []}
    for i, meta in sorted(SOURCE_MAP.items()):
        out[meta["status"]].append(f"[{i:2d}] {BASE_FEATURES[i]}")
    return out


def validate() -> None:
    """Self-check the spec. Cheap insurance against a bad edit."""
    assert len(BASE_FEATURES) == 23, len(BASE_FEATURES)
    assert len(LANDCOVER_CLASSES) == 17
    names = model_channel_names()
    assert len(names) == N_MODEL, f"{len(names)} != {N_MODEL}"
    assert names[0] == "VIIRS band M11"
    assert names[15] == "Palmer drought severity index (PDSI)"
    assert names[16].startswith("Land cover:")
    assert names[32].startswith("Land cover:")
    assert names[33] == "Forecast: Total precipitation"
    assert names[38] == "Active fire"
    assert names[39] == "Active fire (binary)"
    # static/dynamic partition must tile 0..39 exactly once
    both = sorted(list(STATIC_MODEL_IDS) + list(DYNAMIC_MODEL_IDS))
    assert both == list(range(N_MODEL)), "static/dynamic ids do not partition 0..39"
    assert set(SOURCE_MAP) == set(range(N_BASE)), "SOURCE_MAP must cover all 23 bands"


if __name__ == "__main__":
    validate()
    print(f"WSTS spec OK — {N_BASE} base bands -> {N_MODEL} model channels")
    print(f"  static: {len(STATIC_MODEL_IDS)}  dynamic: {len(DYNAMIC_MODEL_IDS)}")
    rep = gap_report()
    for status in ("have", "derive", "missing"):
        print(f"\n{status.upper()} ({len(rep[status])}):")
        for line in rep[status]:
            print(f"   {line}")
