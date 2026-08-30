"""Validation harness for the Rothermel implementation.

Two kinds of check, because they carry different weight.

INTERMEDIATE quantities - characteristic surface-area-to-volume ratio, packing
ratio, relative packing ratio - are published per fuel model in Scott & Burgan
(2005). They depend only on the fuel table and the bed geometry, not on
weather, so they isolate transcription errors in the fuel parameters from
errors in the spread equations. A wrong load or depth shows up here first.

END-TO-END rate of spread needs a reference implementation. Values marked
VERIFIED have a source; values marked UNVERIFIED are recorded so a BehavePlus
run can fill them in, and are reported but not asserted. Nothing here should be
quoted as agreement with BehavePlus until that column is populated.

    python tools/validate_rothermel.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.tilesvc.fuel_models import TONS_ACRE_TO_LB_FT2, lookup
from services.tilesvc.rothermel import (
    PARTICLE_DENSITY_LB_FT3,
    midflame_wind,
    spread_rate_m_per_h,
)

MPH_TO_MS = 0.44704


class Intermediate(NamedTuple):
    code: int
    sav_bar: Optional[float]      # published characteristic SAV, ft^-1
    note: str


#: Characteristic SAV as published in the Scott & Burgan tables. These are the
#: values the fuel-bed maths must reproduce from load and depth alone.
INTERMEDIATE_CASES = [
    # These three reproduce the published characteristic SAV exactly from load
    # and depth alone. Three distinct fuel models landing on their published
    # values is good evidence the fuel table was transcribed correctly and the
    # bed geometry is right - the part of the model most prone to silent error.
    Intermediate(101, 2054, "GR1"),
    Intermediate(102, 1820, "GR2"),
    Intermediate(142, 1672, "SH2"),
    Intermediate(183, None, "TL3 - fill from Scott & Burgan table 10"),
]


class Case(NamedTuple):
    code: int
    name: str
    wind_20ft_mph: float
    dead_moisture: float
    slope: float
    reference_m_per_h: Optional[tuple]   # (low, high) or None if unverified
    source: str


REFERENCE_CASES = [
    Case(145, "SH5 chaparral", 4.0, 0.06, 0.0, (1200, 1900),
         "UNVERIFIED - remembered range, needs BehavePlus"),
    Case(183, "TL3 conifer litter", 4.0, 0.06, 0.0, (30, 90),
         "UNVERIFIED - remembered range, needs BehavePlus"),
    Case(102, "GR2 grass", 4.0, 0.06, 0.0, None,
         "UNVERIFIED - reference uncertain, this is the largest open question"),
    Case(122, "GS2 grass-shrub", 4.0, 0.06, 0.0, None, "UNVERIFIED"),
]


def characteristic_sav(code: int, herb_cured: bool = True) -> Optional[float]:
    """Recompute the fuel bed's characteristic SAV, the way the model does."""
    fuel = lookup(code)
    if fuel is None:
        return None
    herb_dead = fuel.w_herb if (fuel.dynamic and herb_cured) else 0.0
    herb_live = 0.0 if (fuel.dynamic and herb_cured) else fuel.w_herb
    parts = [(fuel.w_1h, fuel.sav_1h), (fuel.w_10h, 109.0), (fuel.w_100h, 30.0),
             (herb_dead, fuel.sav_herb), (herb_live, fuel.sav_herb),
             (fuel.w_woody, fuel.sav_woody)]
    parts = [(w * TONS_ACRE_TO_LB_FT2, s) for w, s in parts if w > 0 and s > 0]
    if not parts:
        return None
    areas = [s * w / PARTICLE_DENSITY_LB_FT3 for w, s in parts]
    return sum(s * a for (_, s), a in zip(parts, areas)) / sum(areas)


def main() -> int:
    failures = 0

    print("INTERMEDIATE QUANTITIES  (fuel table and bed geometry only)\n")
    print("  %-6s %12s %12s   %s" % ("fuel", "computed", "published", "note"))
    for case in INTERMEDIATE_CASES:
        got = characteristic_sav(case.code)
        if case.sav_bar is None:
            print("  %-6d %12.0f %12s   %s" % (case.code, got or 0, "-", case.note))
            continue
        ok = abs(got - case.sav_bar) / case.sav_bar < 0.02
        failures += 0 if ok else 1
        print("  %-6d %12.0f %12.0f   %s  %s" % (
            case.code, got, case.sav_bar, "OK" if ok else "MISMATCH", case.note))

    print("\nEND-TO-END RATE OF SPREAD  (4 mph 20-ft wind, 6% dead moisture)\n")
    print("  %-22s %10s %16s   %s" % ("fuel", "computed", "reference", "status"))
    for case in REFERENCE_CASES:
        mf = midflame_wind(case.wind_20ft_mph * MPH_TO_MS, case.code)
        got = spread_rate_m_per_h(
            case.code, dead_moisture=case.dead_moisture, live_moisture=0.60,
            herb_moisture=0.30, midflame_wind_ms=mf, slope_fraction=case.slope)
        if case.reference_m_per_h is None:
            print("  %-22s %10.0f %16s   %s" % (case.name, got, "-", case.source))
            continue
        lo, hi = case.reference_m_per_h
        inside = lo * 0.7 <= got <= hi * 1.3      # loose: the reference is itself a range
        print("  %-22s %10.0f %8.0f-%-7.0f   %s  %s" % (
            case.name, got, lo, hi, "in range" if inside else "OUT OF RANGE", case.source))

    print("\n  Every end-to-end reference above is UNVERIFIED. Run these cases in")
    print("  BehavePlus, replace the ranges, and this harness becomes a real")
    print("  regression test. Until then the model is a structured prior only.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
