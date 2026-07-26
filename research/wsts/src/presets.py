"""The five out-of-distribution evaluation events.

Kept in sync with `ignis_ml/scripts/eval_historical.py::PRESETS` so the WSTS
research track and the production eval harness score the *same* events at the
same reference times. If these drift, cross-track comparisons are meaningless.

`regime` is the variable of interest: the research question is whether a model
trained on WSTS (dominated by summer/fall vegetated-terrain fires) transfers to
January Santa Ana wildland-urban-interface events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    lat: float
    lon: float
    ref_time: str          # ISO8601 UTC — first prediction step is ref_time + 24h
    regime: str            # "santa_ana" | "summer_fall"
    wui: bool              # significant wildland-urban interface component
    notes: str = ""


PRESETS: Tuple[Preset, ...] = (
    Preset("palisades", "Palisades Fire", 34.078, -118.555,
           "2025-01-07T18:30:00Z", "santa_ana", True,
           "January coastal Santa Ana. The v3 failure case: prediction ran east "
           "along the urban edge instead of southwest with the wind."),
    Preset("eaton", "Eaton Fire", 34.1897, -118.1300,
           "2025-01-07T22:30:00Z", "santa_ana", True,
           "Same wind event as Palisades; Altadena foothills."),
    Preset("camp", "Camp/Paradise Fire", 39.7596, -121.6219,
           "2018-11-08T14:30:00Z", "santa_ana", True,
           "Jarbo Gap wind-driven, November. Not Santa Ana proper (northern "
           "CA downslope wind) but the same extreme-wind WUI regime."),
    Preset("dixie", "Dixie Fire", 39.8760, -121.3870,
           "2021-07-14T17:00:00Z", "summer_fall", False,
           "In-distribution control: July, vegetated terrain, and 2021 is a "
           "WSTS training year. Expect the SOTA model to do WELL here."),
    Preset("caldor", "Caldor Fire", 38.5900, -120.5400,
           "2021-08-14T18:00:00Z", "summer_fall", False,
           "Second in-distribution control. August 2021, Sierra Nevada."),
)

#: The comparison that carries the paper: extreme-wind/WUI events vs the
#: in-distribution controls. If SOTA scores well on Dixie/Caldor and poorly on
#: Palisades/Eaton, that is the generalization gap, cleanly isolated — and the
#: controls rule out "our data pipeline is broken" as the explanation.
OOD_KEYS = tuple(p.key for p in PRESETS if p.regime == "santa_ana")
CONTROL_KEYS = tuple(p.key for p in PRESETS if p.regime == "summer_fall")


def by_key(key: str) -> Preset:
    for p in PRESETS:
        if p.key == key:
            return p
    raise KeyError(f"unknown preset {key!r}; have {[p.key for p in PRESETS]}")


if __name__ == "__main__":
    print(f"{'key':<12}{'regime':<14}{'wui':<6}{'ref_time'}")
    for p in PRESETS:
        print(f"{p.key:<12}{p.regime:<14}{str(p.wui):<6}{p.ref_time}")
    print(f"\nOOD (extreme wind/WUI): {OOD_KEYS}")
    print(f"Controls (in-distribution): {CONTROL_KEYS}")
