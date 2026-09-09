"""Hargreaves potential evapotranspiration from monthly climatology (FAO-56 formulation)."""

from __future__ import annotations

import numpy as np

MID_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def extraterrestrial_radiation_mm(lat_deg: np.ndarray, day_of_year: int) -> np.ndarray:
    """Daily extraterrestrial radiation Ra (mm/day water equivalent), FAO-56 eqs 21-25."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra_mj = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    return 0.408 * ra_mj


def hargreaves_pet_mm_yr(tmin: np.ndarray, tmax: np.ndarray, lat_rows: np.ndarray) -> np.ndarray:
    """Annual Hargreaves PET (mm/yr) from monthly tmin/tmax (12, ny, nx) and row latitudes (ny,)."""
    return hargreaves_pet_monthly_mm(tmin, tmax, lat_rows).sum(axis=0)


SNOW_C = -1.0  # monthly-mean temperature at/below which all precipitation is snow
RAIN_C = 3.0  # ... at/above which all precipitation is rain (linear ramp between)


def hargreaves_pet_monthly_mm(tmin: np.ndarray, tmax: np.ndarray, lat_rows: np.ndarray) -> np.ndarray:
    """Monthly Hargreaves PET totals (12, ny, nx) in mm from monthly tmin/tmax and row latitudes."""
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    tmean = 0.5 * (tmin + tmax)
    trange = np.sqrt(np.clip(tmax - tmin, 0, None))
    out = np.empty_like(tmin)
    for m in range(12):
        ra = extraterrestrial_radiation_mm(lat_rows, int(MID_MONTH_DOY[m]))[:, None]
        out[m] = np.clip(0.0023 * ra * (tmean[m] + 17.8) * trange[m], 0, None) * DAYS_IN_MONTH[m]
    return out


def seasonality_index(monthly: np.ndarray) -> np.ndarray:
    """Walsh & Lawler (1981) seasonality index of a (12, ...) monthly stack: 0 uniform … 1.83 single month."""
    monthly = np.asarray(monthly, dtype=float)
    annual = monthly.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(monthly - annual / 12).sum(axis=0) / annual


def snowfall_fraction(prec: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Precipitation-weighted fraction falling as snow, from (12, ...) monthly precip and mean temperature.

    Each month's snow share ramps linearly from 1 at ``SNOW_C`` to 0 at ``RAIN_C``
    (a smooth stand-in for the within-month spread of daily temperatures).
    """
    prec = np.asarray(prec, dtype=float)
    share = np.clip((RAIN_C - np.asarray(temp, dtype=float)) / (RAIN_C - SNOW_C), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (prec * share).sum(axis=0) / prec.sum(axis=0)


MIN_P_MM_YR = 1.0  # WorldClim reports annual precipitation as integer mm; below 1 mm is unresolved


def aridity_index(pet_mm_yr: np.ndarray, prec_mm_yr: np.ndarray) -> np.ndarray:
    """Aridity = PET / P with a precipitation floor at the data's resolution.

    Hyper-arid cells (Sahara, Atacama) round to 0 mm/yr in WorldClim, and cell means
    over mostly-zero pixels can land at ~1e-4 mm/yr, which makes a raw PET/P ratio
    infinite or absurd (2.4e6 was observed globally). Flooring P at
    ``MIN_P_MM_YR`` keeps the index finite and monotone while preserving the
    ordering that makes it useful: the driest cells still score highest.
    """
    prec = np.asarray(prec_mm_yr, dtype=float)
    return np.asarray(pet_mm_yr, dtype=float) / np.maximum(prec, MIN_P_MM_YR)
