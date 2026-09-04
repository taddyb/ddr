"""Terrain derivatives on geographic (lat/lon) DEM tiles."""

from __future__ import annotations

import numpy as np

R_EARTH_M = 6371000.0


def slope_degrees(elv: np.ndarray, lat_rows: np.ndarray, res_deg: float) -> np.ndarray:
    """Slope (degrees) of a north-up DEM on a regular lat/lon grid.

    ``lat_rows`` is the latitude of each row (any order); the east-west pixel
    size shrinks with cos(lat) while north-south is constant.
    """
    dy = R_EARTH_M * np.radians(res_deg)
    dx = dy * np.cos(np.radians(np.asarray(lat_rows, dtype=float)))[:, None]
    gy, gx = np.gradient(np.asarray(elv, dtype=float))
    return np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
