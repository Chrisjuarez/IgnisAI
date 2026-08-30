"""Rothermel (1972) surface fire spread rate.

The model underneath BehavePlus, FARSITE, FlamMap and ELMFIRE. It answers one
question - how fast does a surface fire spread through this fuel, at this
moisture, under this wind, on this slope - from published fuel properties
rather than from anything learned.

That is the reason to have it. A learned model gives a number with no account
of itself; this gives a number an underwriter or a fire behaviour analyst can
interrogate, and it fails in ways that are explicable.

Equations follow Andrews 2018 (RMRS-GTR-371), which restates Rothermel with the
Albini corrections. Calculations stay in Rothermel's native imperial units and
convert only at the boundary: the published coefficients are fitted to those
units, and rewriting them metric is a well-known source of silent error.

Reference for the reader checking this against the source:
    R = I_R * xi * (1 + phi_w + phi_s) / (rho_b * epsilon * Q_ig)

MODULE NOTE - validation status. The equations follow the published form and
the behaviour is right qualitatively: grass outruns litter, chaparral outruns
both, spread rises with wind and slope and falls with moisture, and fuel above
its moisture of extinction does not burn. What has NOT been done is a
quantitative check against BehavePlus or ELMFIRE on matched inputs. Spot
comparisons put the heavier shrub and litter models perhaps two to three times
above remembered published figures, which is well inside the range a wind
adjustment factor or a fuel moisture scenario can account for - but it is not
verified, and until it is, nothing here should be quoted as a spread rate to a
third party. Treat the output as a physically-structured prior, not as a
calibrated forecast.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .fuel_models import TONS_ACRE_TO_LB_FT2, FuelModel, lookup

#: Rothermel's constants for wood.
HEAT_CONTENT_BTU_LB = 8000.0
PARTICLE_DENSITY_LB_FT3 = 32.0
TOTAL_MINERAL_CONTENT = 0.0555
EFFECTIVE_MINERAL_CONTENT = 0.010

FT_PER_MIN_TO_M_PER_H = 0.3048 * 60.0
M_PER_S_TO_FT_PER_MIN = 196.85


def _weighted(values, weights) -> float:
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total > 0 else 0.0


#: Heavier dead fuels hold more water than the fine fuels that carry the fire.
#: These offsets are the usual spacing between the 1/10/100-hour classes in the
#: standard fuel moisture scenarios.
MOISTURE_STEP_10H = 0.02
MOISTURE_STEP_100H = 0.04

#: Below this the fuel is at its moisture of extinction and the fire is out.
ETA_M_EXTINCT = 1e-9

#: Effective wind speed limit, ft/min per unit reaction intensity. Rothermel's
#: wind factor is unbounded; Andrews (2013) showed it must be capped near
#: 0.9 * I_R or the model returns impossible spread rates in strong wind.
EFFECTIVE_WIND_LIMIT_COEFF = 0.9

#: Grass is fully cured at or below 30% herbaceous moisture and fully green at
#: 120%, transferring linearly between (Scott & Burgan 2005).
CURED_AT = 0.30
GREEN_AT = 1.20


def curing_fraction(herb_moisture: float) -> float:
    """How much of a dynamic model's herbaceous load has become dead fuel."""
    if herb_moisture <= CURED_AT:
        return 1.0
    if herb_moisture >= GREEN_AT:
        return 0.0
    return float((GREEN_AT - herb_moisture) / (GREEN_AT - CURED_AT))


def spread_rate_m_per_h(
    fuel_code: int,
    *,
    dead_moisture: float,
    live_moisture: float = 1.0,
    herb_moisture: Optional[float] = None,
    midflame_wind_ms: float = 0.0,
    slope_fraction: float = 0.0,
) -> float:
    """Head-fire rate of spread, metres per hour.

    dead_moisture and live_moisture are fractions of oven-dry weight (0.06 is a
    dry 6% fine dead fuel; live chaparral in drought sits near 0.6).
    slope_fraction is rise over run, not degrees.
    """
    fuel = lookup(fuel_code)
    if fuel is None:
        return 0.0                       # non-burnable: fire does not carry

    # --- loads and surface areas, by particle class -------------------------
    #
    # Two details decide whether the numbers come out anywhere near published
    # fire behaviour, and leaving either out is visible immediately:
    #
    # Curing. In the dynamic models the herbaceous load starts live, but cured
    # grass in fire season is dead fuel at dead-fuel moisture. Without the
    # transfer, grass models are damped by live moisture and come out SLOWER
    # than conifer litter, which is the reverse of reality.
    #
    # Size-class moisture. 10-hour and 100-hour fuels do not dry to the same
    # value as fine 1-hour fuels. Applying the fine moisture to everything
    # makes the heavy shrub models - the ones with large coarse loads - far too
    # fast.
    cured = curing_fraction(herb_moisture if herb_moisture is not None else live_moisture)
    herb_live = fuel.w_herb * (1.0 - cured) if fuel.dynamic else fuel.w_herb
    herb_dead = fuel.w_herb * cured if fuel.dynamic else 0.0

    dead_parts = [
        (fuel.w_1h, fuel.sav_1h, dead_moisture),
        (fuel.w_10h, 109.0, dead_moisture + MOISTURE_STEP_10H),
        (fuel.w_100h, 30.0, dead_moisture + MOISTURE_STEP_100H),
        (herb_dead, fuel.sav_herb, dead_moisture),
    ]
    dead = [(w * TONS_ACRE_TO_LB_FT2, sv, m) for w, sv, m in dead_parts if w > 0 and sv > 0]
    live = [(w * TONS_ACRE_TO_LB_FT2, sv, live_moisture) for w, sv in
            ((herb_live, fuel.sav_herb), (fuel.w_woody, fuel.sav_woody)) if w > 0 and sv > 0]

    if not dead and not live:
        return 0.0

    def category(parts):
        if not parts:
            return 0.0, 0.0, 0.0, 0.0
        areas = [sv * w / PARTICLE_DENSITY_LB_FT3 for w, sv, _ in parts]
        load = sum(w for w, _, _ in parts)
        sav = _weighted([sv for _, sv, _ in parts], areas)
        # Characteristic moisture is area-weighted, so the fine fuels that
        # actually carry the fire dominate it.
        moisture = _weighted([m for _, _, m in parts], areas)
        return load, sav, sum(areas), moisture

    w_dead, sav_dead, area_dead, m_dead = category(dead)
    w_live, sav_live, area_live, m_live = category(live)
    area_total = area_dead + area_live
    if area_total <= 0:
        return 0.0

    f_dead, f_live = area_dead / area_total, area_live / area_total
    sav_bar = f_dead * sav_dead + f_live * sav_live
    if sav_bar <= 0:
        return 0.0

    # --- fuel bed ------------------------------------------------------------
    depth = max(fuel.depth_ft, 0.1)
    rho_b = (w_dead + w_live) / depth                      # bulk density
    beta = rho_b / PARTICLE_DENSITY_LB_FT3                 # packing ratio
    beta_op = 3.348 * sav_bar ** -0.8189
    beta_ratio = beta / beta_op if beta_op > 0 else 0.0

    # --- reaction intensity --------------------------------------------------
    gamma_max = sav_bar ** 1.5 / (495.0 + 0.0594 * sav_bar ** 1.5)
    a_exp = 133.0 * sav_bar ** -0.7913
    gamma = gamma_max * (beta_ratio ** a_exp) * math.exp(a_exp * (1.0 - beta_ratio))

    # Live fuel raises the effective moisture of extinction only when there is
    # dead fuel to carry the fire; a pure live bed uses the dead value.
    mx_live = fuel.mx_dead
    if w_live > 0 and w_dead > 0:
        fine_dead = sum(w * math.exp(-138.0 / sv) for w, sv, _ in dead)
        fine_live = sum(w * math.exp(-500.0 / sv) for w, sv, _ in live if sv > 0)
        if fine_live > 0:
            ratio = fine_dead / fine_live
            m_fine_dead = m_dead
            mx_live = max(
                2.9 * ratio * (1.0 - m_fine_dead / fuel.mx_dead) - 0.226,
                fuel.mx_dead,
            )

    def damping(moisture: float, mx: float) -> float:
        r = min(moisture / mx, 1.0) if mx > 0 else 1.0
        return max(0.0, 1.0 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3)

    eta_m = (f_dead * damping(m_dead, fuel.mx_dead)
             + f_live * damping(m_live, mx_live))
    eta_s = min(0.174 * EFFECTIVE_MINERAL_CONTENT ** -0.19, 1.0)
    w_net = (w_dead + w_live) * (1.0 - TOTAL_MINERAL_CONTENT)

    reaction_intensity = gamma * w_net * HEAT_CONTENT_BTU_LB * eta_m * eta_s
    # At the moisture of extinction the damping polynomial evaluates to zero
    # only to within rounding, and a fire that is out should report exactly no
    # spread rather than 1e-13 m/h.
    if eta_m <= ETA_M_EXTINCT or reaction_intensity <= 0:
        return 0.0

    # --- propagating flux, wind and slope ------------------------------------
    xi = (math.exp((0.792 + 0.681 * math.sqrt(sav_bar)) * (beta + 0.1))
          / (192.0 + 0.2595 * sav_bar))

    wind_ft_min = max(0.0, midflame_wind_ms) * M_PER_S_TO_FT_PER_MIN
    c = 7.47 * math.exp(-0.133 * sav_bar ** 0.55)
    b = 0.02526 * sav_bar ** 0.54
    e = 0.715 * math.exp(-3.59e-4 * sav_bar)

    # Effective wind speed limit (Andrews 2013; Andrews 2018 sec. 5). Rothermel's
    # wind factor has no upper bound, but the model was fitted well below the
    # winds a Santa Ana produces and runs away above roughly 0.9 * I_R. Left
    # unbounded it returns spread rates that are physically impossible - which
    # is exactly where this implementation was overshooting.
    wind_limit_ft_min = EFFECTIVE_WIND_LIMIT_COEFF * reaction_intensity
    wind_capped = min(wind_ft_min, wind_limit_ft_min)
    wind_limited = wind_ft_min > wind_limit_ft_min

    phi_w = c * (wind_capped ** b) * (beta_ratio ** -e) if wind_capped > 0 and beta_ratio > 0 else 0.0
    phi_s = 5.275 * beta ** -0.3 * max(0.0, slope_fraction) ** 2 if beta > 0 else 0.0

    # --- heat sink -----------------------------------------------------------
    epsilon = math.exp(-138.0 / sav_bar)
    q_ig = 250.0 + 1116.0 * m_dead

    r_ft_min = reaction_intensity * xi * (1.0 + phi_w + phi_s) / (rho_b * epsilon * q_ig)
    return max(0.0, r_ft_min) * FT_PER_MIN_TO_M_PER_H


def length_to_breadth(midflame_wind_ms: float) -> float:
    """Anderson (1983): how elongated the ellipse gets with wind."""
    mph = max(0.0, midflame_wind_ms) * 2.23694
    return float(min(1.0 + 0.25 * mph, 8.0))


#: Open wind is not what the fire feels. Rothermel takes midflame wind, and
#: 20-ft/10-m wind must be reduced for the fuel bed and any canopy above it.
#: These are the standard unsheltered factors by fuel type.
WIND_ADJUSTMENT = {"GR": 0.36, "GS": 0.32, "SH": 0.30, "TU": 0.22, "TL": 0.20, "SB": 0.25}


def midflame_wind(open_wind_ms: float, fuel_code: int) -> float:
    fuel = lookup(fuel_code)
    if fuel is None:
        return 0.0
    return float(open_wind_ms) * WIND_ADJUSTMENT.get(fuel.name[:2], 0.30)


def spread_rate_grid(
    fuel_codes: np.ndarray,
    *,
    dead_moisture: np.ndarray | float,
    live_moisture: np.ndarray | float = 1.0,
    wind_ms: np.ndarray | float,
    slope_fraction: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Rate of spread for every cell, metres per hour.

    Evaluated once per distinct fuel code rather than per cell: the equations
    are expensive and a 64x64 tile holds only a handful of fuel models.
    """
    codes = np.asarray(fuel_codes, dtype=np.int32)
    out = np.zeros(codes.shape, dtype=np.float32)

    def at(value, mask):
        return float(np.mean(np.asarray(value)[mask])) if np.ndim(value) else float(value)

    for code in np.unique(codes):
        mask = codes == code
        if lookup(int(code)) is None:
            continue
        out[mask] = spread_rate_m_per_h(
            int(code),
            dead_moisture=at(dead_moisture, mask),
            live_moisture=at(live_moisture, mask),
            midflame_wind_ms=midflame_wind(at(wind_ms, mask), int(code)),
            slope_fraction=at(slope_fraction, mask),
        )
    return out
