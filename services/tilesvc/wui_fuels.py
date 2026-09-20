"""Wildland-urban interface fuels.

FBFM40 classes 91-93 as non-burnable: urban, developed, agriculture. That is
correct for the question the fuel models were built to answer - how a surface
fire moves through wildland vegetation - and wrong for the fires that generate
insurance claims. Palisades destroyed thousands of structures in terrain this
raster calls NB1. Standard fuel models say the town cannot burn. It burned.

So a physics model that stops at the city limit is not conservative, it is
wrong in the direction that matters: it under-predicts exactly where the
exposed assets are.

This assigns burnable fuel to developed cells from structure density, using
impervious surface as the proxy the pipeline already carries. The treatment is
deliberately explicit and opt-in - inventing fuel where a published model says
there is none is a real modelling claim, and it should never happen silently.

Approach and its limits. Structure-to-structure spread is driven by ember cast,
radiant heat between buildings, and construction materials - none of which
Rothermel represents. Mapping density onto a surface fuel model is a coarse
stand-in for a mechanism it does not contain. It is defensible as a first
approximation because WUI conflagrations do propagate roughly as a front whose
rate rises with fuel continuity, and structure density is a usable proxy for
continuity. It is NOT defensible as a structure-ignition model, and nothing
here should be read as a per-building probability.
"""
from __future__ import annotations

import numpy as np

from .fuel_models import lookup

#: FBFM40 developed classes. Water (98) and barren (99) stay non-burnable -
#: those are genuinely non-flammable, not merely unmodelled.
DEVELOPED_CODES = frozenset({91, 92, 93})

#: Substitutes by structure density. Sparse development behaves like the
#: grass-shrub it is interspersed with; dense development carries more like a
#: heavy shrub bed, which is the closest surface analogue to a continuous
#: run of structures with ornamental vegetation between them.
SPARSE_SUBSTITUTE = 121     # GS1 low load grass-shrub
MODERATE_SUBSTITUTE = 122   # GS2 moderate load grass-shrub
DENSE_SUBSTITUTE = 142      # SH2 moderate load shrub

#: Impervious fraction bounds. Below the low bound a cell is effectively
#: wildland with a building in it; above the high bound it is continuous urban
#: fabric - a city core, where a surface-fire analogue stops being meaningful
#: and the dominant mechanism is ember cast this model does not represent.
SPARSE_MAX = 0.35
MODERATE_MAX = 0.65
CORE_URBAN_MIN = 0.90


def apply_wui_fuels(
    fuel_codes: np.ndarray,
    impervious: np.ndarray,
    *,
    enable: bool = True,
) -> np.ndarray:
    """Give developed cells a burnable fuel model based on structure density.

    impervious is a 0-1 fraction on the same grid. Returns a new array; the
    input is not modified, so the unmodified wildland answer stays available
    for comparison.
    """
    codes = np.asarray(fuel_codes, dtype=np.int16).copy()
    if not enable:
        return codes

    imp = np.clip(np.asarray(impervious, dtype=np.float32), 0.0, 1.0)
    developed = np.isin(codes, list(DEVELOPED_CODES))

    codes[developed & (imp < SPARSE_MAX)] = SPARSE_SUBSTITUTE
    codes[developed & (imp >= SPARSE_MAX) & (imp < MODERATE_MAX)] = MODERATE_SUBSTITUTE
    codes[developed & (imp >= MODERATE_MAX) & (imp < CORE_URBAN_MIN)] = DENSE_SUBSTITUTE
    # Continuous urban core keeps its non-burnable class: a surface-fire model
    # has nothing useful to say there, and guessing would be worse than
    # declining to answer.
    return codes


def wui_summary(original: np.ndarray, adjusted: np.ndarray) -> dict:
    """What the treatment changed, so a response can declare it."""
    orig = np.asarray(original)
    adj = np.asarray(adjusted)
    changed = orig != adj
    burnable_before = np.array([[lookup(int(c)) is not None for c in row] for row in orig])
    burnable_after = np.array([[lookup(int(c)) is not None for c in row] for row in adj])
    return {
        "applied": bool(changed.any()),
        "cells_reassigned": int(changed.sum()),
        "burnable_fraction_before": float(burnable_before.mean()),
        "burnable_fraction_after": float(burnable_after.mean()),
        "basis": "impervious_surface_density",
        "caveat": (
            "Developed cells are given a surface fuel analogue from structure "
            "density. Ember cast and structure-to-structure ignition are not "
            "modelled; this is not a per-building ignition probability."
        ),
    }
