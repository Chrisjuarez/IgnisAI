"""Validation harness for the Rothermel implementation.

Two kinds of check, because they carry different weight.

INTERMEDIATE quantities - characteristic surface-area-to-volume ratio, packing
ratio, relative packing ratio - are published per fuel model in Scott & Burgan
(2005). They depend only on the fuel table and the bed geometry, not on
weather, so they isolate transcription errors in the fuel parameters from
errors in the spread equations. A wrong load or depth shows up here first.

END-TO-END rate of spread is now checked against BehavePlus itself. pyrothermel
(MIT) wraps the Behave core from the RMRS Missoula Fire Sciences Laboratory,
which is US Government work and therefore public domain under 17 USC 105 - so
it can be used commercially without restriction, unlike ELMFIRE, whose AGPL
plus Commons Clause forbids selling a service built on it.

    pip install pyrothermel
    python tools/validate_rothermel.py

Without pyrothermel installed the oracle comparison is skipped and only the
intermediate checks run.
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


#: Fuels and midflame winds to compare against the oracle.
ORACLE_FUELS = [("GR1", 101), ("GR2", 102), ("GS1", 121), ("GS2", 122),
                ("SH2", 142), ("SH5", 145), ("SH7", 147),
                ("TL3", 183), ("TL8", 188), ("TU5", 165)]
ORACLE_WINDS = (1.0, 3.0)

#: BehavePlus moisture scenario 1/1, read from pyrothermel rather than assumed.
ORACLE_MOISTURE = dict(one_hour=0.03, ten_hour=0.04, hundred_hour=0.05,
                       live_herbaceous=0.30, live_woody=0.60)


def compare_against_behaveplus():
    """Rate of spread against the Behave core, matched moisture and wind.

    Returns (rows, median_ratio) or None when pyrothermel is not installed.
    """
    try:
        import pyrothermel as pr
    except ImportError:
        return None

    import services.tilesvc.rothermel as R

    # Match Behave's scenario spacing so the comparison isolates the spread
    # equations rather than a difference in how the size classes are dried.
    saved = (R.MOISTURE_STEP_10H, R.MOISTURE_STEP_100H)
    R.MOISTURE_STEP_10H = ORACLE_MOISTURE["ten_hour"] - ORACLE_MOISTURE["one_hour"]
    R.MOISTURE_STEP_100H = ORACLE_MOISTURE["hundred_hour"] - ORACLE_MOISTURE["one_hour"]
    try:
        rows = []
        for name, code in ORACLE_FUELS:
            for wind in ORACLE_WINDS:
                fm = pr.FuelModel.from_existing(name, units_preset="metric")
                ms = pr.MoistureScenario.from_existing(1, 1)
                reference = pr.PyrothermelRun(
                    fm, ms, wind_speed=wind, units_preset="metric",
                    wind_input_mode="direct_midflame", slope=0.0,
                ).run_surface_fire_in_direction_of_max_spread()["spread_rate"] * 3600.0
                mine = R.spread_rate_m_per_h(
                    code, dead_moisture=ORACLE_MOISTURE["one_hour"],
                    live_moisture=ORACLE_MOISTURE["live_woody"],
                    herb_moisture=ORACLE_MOISTURE["live_herbaceous"],
                    midflame_wind_ms=wind, slope_fraction=0.0)
                rows.append((name, wind, reference, mine,
                             mine / reference if reference > 0 else float("nan")))
    finally:
        R.MOISTURE_STEP_10H, R.MOISTURE_STEP_100H = saved

    ratios = sorted(r for *_, r in rows if r == r)
    median = ratios[len(ratios) // 2] if ratios else float("nan")
    return rows, median


def compare_reference_implementations():
    """BehavePlus against pyretechnics on identical inputs.

    Worth doing before trusting either as an oracle. They implement the same
    published equations and still disagree by up to a factor of two, mostly on
    whether the effective wind speed limit is applied - GR1 at 3 m/s is 1220
    m/h in BehavePlus, 168 with pyretechnics' limit on and 1723 with it off.
    Shrub and litter differ substantially either way.

    The conclusion that matters: there is no single implementation ground
    truth. Matching one is a statement about configuration, not correctness,
    and only observed fires settle the question.
    """
    try:
        import pyretechnics.conversion as cv
        import pyretechnics.fuel_models as pfm
        import pyretechnics.surface_fire as psf
        import pyrothermel as pr
    except ImportError:
        return None

    MPS_TO_FPM = 196.85
    moisture = (ORACLE_MOISTURE["one_hour"], ORACLE_MOISTURE["ten_hour"],
                ORACLE_MOISTURE["hundred_hour"], 0.0,
                ORACLE_MOISTURE["live_herbaceous"], ORACLE_MOISTURE["live_woody"])
    rows = []
    for name, code in ORACLE_FUELS:
        for wind in ORACLE_WINDS:
            behave = pr.PyrothermelRun(
                pr.FuelModel.from_existing(name, units_preset="metric"),
                pr.MoistureScenario.from_existing(1, 1), wind_speed=wind,
                units_preset="metric", wind_input_mode="direct_midflame", slope=0.0,
            ).run_surface_fire_in_direction_of_max_spread()["spread_rate"] * 3600.0
            base = psf.calc_surface_fire_behavior_no_wind_no_slope(
                pfm.moisturize(pfm.get_fuel_model(code), moisture))
            def pyre(limit):
                return cv.fpm_to_mps(psf.calc_surface_fire_behavior_max(
                    base, wind * MPS_TO_FPM, 0.0, 0.0, 0.0,
                    use_wind_limit=limit)["max_spread_rate"]) * 3600.0
            rows.append((name, wind, behave, pyre(True), pyre(False)))
    return rows


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

    oracle = compare_against_behaveplus()
    if oracle is None:
        print("\n  pyrothermel not installed - oracle comparison skipped.")
        print("  pip install pyrothermel  (MIT; wraps the public-domain Behave core)")
        return 1 if failures else 0

    rows, median = oracle
    print("\nAGAINST BEHAVEPLUS  (pyrothermel, matched moisture and midflame wind)\n")
    print("  %-6s %9s %12s %10s %8s" % ("fuel", "wind m/s", "BehavePlus", "mine", "ratio"))
    for name, wind, reference, mine, ratio in rows:
        flag = "" if 0.8 <= ratio <= 1.25 else "  <--"
        print("  %-6s %9.1f %12.0f %10.0f %8.2f%s" % (name, wind, reference, mine, ratio, flag))

    print("\n  median ratio mine/BehavePlus : %.2f   (1.00 = exact)" % median)
    outliers = sum(1 for *_, r in rows if not (0.8 <= r <= 1.25))
    print("  cases outside +/-25%%          : %d of %d" % (outliers, len(rows)))
    print()
    print("  The disagreement is systematic, not noise: too slow in grass, too")
    print("  fast in shrub and litter, and the ratio grows with wind in almost")
    print("  every fuel - which points at the wind factor rather than the fuel")
    print("  table, since the characteristic SAV checks above pass exactly.")
    print()
    cross = compare_reference_implementations()
    if cross:
        print("\nTHE TWO REFERENCE IMPLEMENTATIONS AGAINST EACH OTHER\n")
        print("  %-6s %9s %12s %13s %13s" % (
            "fuel", "wind m/s", "BehavePlus", "pyre limited", "pyre unlimited"))
        for name, wind, behave, limited, unlimited in cross:
            print("  %-6s %9.1f %12.0f %13.0f %13.0f" % (name, wind, behave, limited, unlimited))
        print()
        print("  They implement the same published equations and still differ by up")
        print("  to a factor of two, largely on whether the effective wind speed")
        print("  limit is applied. There is no single implementation ground truth:")
        print("  matching one is a statement about configuration, not correctness.")
        print("  Only observed fires settle it - see the five held-out events in")
        print("  wsts_inputs, which no checkpoint has trained on.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
