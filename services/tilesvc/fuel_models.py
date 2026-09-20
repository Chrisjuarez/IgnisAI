"""Scott & Burgan (2005) FBFM40 fuel model parameters.

The numbers Rothermel's equations need for each fuel class, straight from the
published tables (RMRS-GTR-153). Loads are given in tons/acre and depths in
feet, as published, and converted once at the boundary - transcribing
pre-converted numbers is how these tables usually go wrong.

Codes not listed are treated as non-burnable, which is the safe direction: a
missing fuel model produces no spread rather than an invented one.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional

#: tons/acre -> lb/ft^2
TONS_ACRE_TO_LB_FT2 = 2000.0 / 43560.0

#: FBFM40 reserves 91-99 for non-burnable: urban, snow, agriculture, water,
#: barren. Fire does not carry in them and the model must not pretend it does.
NON_BURNABLE = frozenset({91, 92, 93, 98, 99})


class FuelModel(NamedTuple):
    code: int
    name: str
    w_1h: float          # tons/acre
    w_10h: float
    w_100h: float
    w_herb: float        # live herbaceous
    w_woody: float       # live woody
    depth_ft: float      # fuel bed depth
    mx_dead: float       # dead fuel moisture of extinction, fraction
    sav_1h: float        # ft^-1
    sav_herb: float
    sav_woody: float
    dynamic: bool        # herbaceous load transfers dead<->live with curing


def _m(code, name, w1, w10, w100, wh, ww, depth, mx, s1, sh, sw, dyn=False):
    return FuelModel(code, name, w1, w10, w100, wh, ww, depth, mx / 100.0, s1, sh, sw, dyn)


#: The models present in the western-CONUS raster. Grass, grass-shrub, shrub,
#: timber-understory, timber-litter and slash.
FUEL_MODELS: Dict[int, FuelModel] = {m.code: m for m in [
    # --- GR: grass -----------------------------------------------------------
    _m(101, "GR1 short sparse dry climate grass", 0.10, 0.0, 0.0, 0.30, 0.0, 0.4, 15, 2200, 2000, 0, True),
    _m(102, "GR2 low load dry climate grass",     0.10, 0.0, 0.0, 1.00, 0.0, 1.0, 15, 2000, 1800, 0, True),
    _m(103, "GR3 low load very coarse grass",     0.10, 0.40, 0.0, 1.50, 0.0, 2.0, 30, 1500, 1300, 0, True),
    _m(104, "GR4 moderate load dry climate grass", 0.25, 0.0, 0.0, 1.90, 0.0, 2.0, 15, 2000, 1800, 0, True),
    _m(105, "GR5 low load humid climate grass",   0.40, 0.0, 0.0, 2.50, 0.0, 1.5, 40, 1800, 1600, 0, True),
    _m(106, "GR6 moderate load humid grass",      0.10, 0.0, 0.0, 3.40, 0.0, 1.5, 40, 2200, 2000, 0, True),
    _m(107, "GR7 high load dry climate grass",    1.00, 0.0, 0.0, 5.40, 0.0, 3.0, 15, 2000, 1800, 0, True),
    _m(108, "GR8 high load very coarse grass",    0.50, 1.00, 0.0, 7.30, 0.0, 4.0, 30, 1500, 1300, 0, True),
    _m(109, "GR9 very high load humid grass",     1.00, 1.00, 0.0, 9.00, 0.0, 5.0, 40, 1800, 1600, 0, True),
    # --- GS: grass-shrub -----------------------------------------------------
    _m(121, "GS1 low load dry climate grass-shrub", 0.20, 0.0, 0.0, 0.50, 0.65, 0.9, 15, 2000, 1800, 1800, True),
    _m(122, "GS2 moderate load dry grass-shrub",    0.50, 0.50, 0.0, 0.60, 1.00, 1.5, 15, 2000, 1800, 1800, True),
    _m(123, "GS3 moderate load humid grass-shrub",  0.30, 0.25, 0.0, 1.45, 1.25, 1.8, 40, 1800, 1600, 1600, True),
    _m(124, "GS4 high load humid grass-shrub",      1.90, 0.30, 0.10, 3.40, 7.10, 2.1, 40, 1800, 1600, 1600, True),
    # --- SH: shrub -----------------------------------------------------------
    _m(141, "SH1 low load dry climate shrub",     0.25, 0.25, 0.0, 0.15, 1.30, 1.0, 15, 2000, 1800, 1600, True),
    _m(142, "SH2 moderate load dry climate shrub", 1.35, 2.40, 0.75, 0.0, 3.85, 1.0, 15, 2000, 0, 1600),
    _m(143, "SH3 moderate load humid climate shrub", 0.45, 3.00, 0.0, 0.0, 6.20, 2.4, 40, 1600, 0, 1400),
    _m(144, "SH4 low load humid climate timber-shrub", 0.85, 1.15, 0.20, 0.0, 2.55, 3.0, 30, 2000, 0, 1600),
    _m(145, "SH5 high load dry climate shrub",    3.60, 2.10, 0.0, 0.0, 2.90, 6.0, 15, 750, 0, 1600),
    _m(146, "SH6 low load humid climate shrub",   2.90, 1.45, 0.0, 0.0, 1.40, 2.0, 30, 750, 0, 1600),
    _m(147, "SH7 very high load dry climate shrub", 3.50, 5.30, 2.20, 0.0, 3.40, 6.0, 15, 750, 0, 1600),
    _m(148, "SH8 high load humid climate shrub",  2.05, 3.40, 0.85, 0.0, 4.35, 3.0, 40, 750, 0, 1600),
    _m(149, "SH9 very high load humid shrub",     4.50, 2.45, 0.0, 1.55, 7.00, 4.4, 40, 750, 1800, 1500, True),
    # --- TU: timber-understory ----------------------------------------------
    _m(161, "TU1 light load dry climate timber-grass-shrub", 0.20, 0.90, 1.50, 0.20, 0.90, 0.6, 20, 2000, 1800, 1600, True),
    _m(162, "TU2 moderate load humid climate timber-shrub",  0.95, 1.80, 1.25, 0.0, 0.20, 1.0, 30, 2000, 0, 1600),
    _m(163, "TU3 moderate load humid climate timber-grass-shrub", 1.10, 0.15, 0.25, 0.65, 1.10, 1.3, 30, 1800, 1600, 1400, True),
    _m(164, "TU4 dwarf conifer with understory",  4.50, 0.0, 0.0, 0.0, 2.00, 0.5, 12, 2300, 0, 2000),
    _m(165, "TU5 very high load dry climate timber-shrub", 4.00, 4.00, 3.00, 0.0, 3.00, 1.0, 25, 1500, 0, 750),
    # --- TL: timber litter ---------------------------------------------------
    _m(181, "TL1 low load compact conifer litter", 1.00, 2.20, 3.60, 0.0, 0.0, 0.2, 30, 2000, 0, 0),
    _m(182, "TL2 low load broadleaf litter",       1.40, 2.30, 2.20, 0.0, 0.0, 0.2, 25, 2000, 0, 0),
    _m(183, "TL3 moderate load confier litter",    0.50, 2.20, 2.80, 0.0, 0.0, 0.3, 20, 2000, 0, 0),
    _m(184, "TL4 small downed logs",               0.50, 1.50, 4.20, 0.0, 0.0, 0.4, 25, 2000, 0, 0),
    _m(185, "TL5 high load conifer litter",        1.15, 2.50, 4.40, 0.0, 0.0, 0.6, 25, 2000, 0, 1600),
    _m(186, "TL6 moderate load broadleaf litter",  2.40, 1.20, 1.20, 0.0, 0.0, 0.3, 25, 2000, 0, 0),
    _m(187, "TL7 large downed logs",               0.30, 1.40, 8.10, 0.0, 0.0, 0.4, 25, 2000, 0, 0),
    _m(188, "TL8 long-needle litter",              5.80, 1.40, 1.10, 0.0, 0.0, 0.3, 35, 1800, 0, 0),
    _m(189, "TL9 very high load broadleaf litter", 6.65, 3.30, 4.15, 0.0, 0.0, 0.6, 35, 1800, 0, 1600),
    # --- SB: slash-blowdown --------------------------------------------------
    _m(201, "SB1 low load activity fuel",          1.50, 3.00, 11.00, 0.0, 0.0, 1.0, 25, 2000, 0, 0),
    _m(202, "SB2 moderate load activity fuel",     4.50, 4.25, 4.00, 0.0, 0.0, 1.0, 25, 2000, 0, 0),
    _m(203, "SB3 high load activity fuel",         5.50, 2.75, 3.00, 0.0, 0.0, 1.2, 25, 2000, 0, 0),
    _m(204, "SB4 high load blowdown",              5.25, 3.50, 5.25, 0.0, 0.0, 2.7, 25, 2000, 0, 0),
]}


def lookup(code: int) -> Optional[FuelModel]:
    """The fuel model for an FBFM40 code, or None where fire does not carry."""
    if int(code) in NON_BURNABLE:
        return None
    return FUEL_MODELS.get(int(code))
