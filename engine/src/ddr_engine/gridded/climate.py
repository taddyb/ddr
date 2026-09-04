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
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    tmean = 0.5 * (tmin + tmax)
    trange = np.sqrt(np.clip(tmax - tmin, 0, None))
    total = np.zeros(tmin.shape[1:], dtype=float)
    for m in range(12):
        ra = extraterrestrial_radiation_mm(lat_rows, int(MID_MONTH_DOY[m]))[:, None]
        pet_day = 0.0023 * ra * (tmean[m] + 17.8) * trange[m]
        total += np.clip(pet_day, 0, None) * DAYS_IN_MONTH[m]
    return total
