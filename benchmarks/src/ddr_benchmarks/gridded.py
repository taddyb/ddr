"""CONUS gridded (DDM30) benchmark helpers.

Pure functions for benchmarking DDR routing on the ISIMIP DDM30 0.5-degree
network: point-to-cell assignment, downstream network closure, topological
accumulation (drainage areas / summed-Q' baseline), and drainage-area-matched
gauge snapping.

Positions throughout refer to indices into the topologically sorted ``order``
array of a DDM30 adjacency zarr (see ``ddr_engine.gridded``); ``dn`` arrays
map each position to its downstream position, or -1 for terminals.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NCOLS = 720
LAT0, LON0 = -55.75, -179.75  # DDM30 cell centers of row 0 / col 0
DA_TOLERANCE = (0.7, 1.4)  # acceptable cell-drained-area / gauge-DA ratio


def assign_cells(lats: NDArray, lons: NDArray) -> NDArray[np.int64]:
    """Assign points to flat DDM30 cell ids (``row * 720 + col``)."""
    rows = np.round((np.asarray(lats) - LAT0) / 0.5).astype(np.int64)
    cols = np.round((np.asarray(lons) - LON0) / 0.5).astype(np.int64)
    return rows * NCOLS + cols


def cell_areas_km2(lat_centers: NDArray) -> NDArray[np.float64]:
    """Area of 0.5-degree cells centered at the given latitudes."""
    r_earth = 6371.0
    dlam = np.radians(0.5)
    lat = np.asarray(lat_centers, dtype=np.float64)
    return r_earth**2 * dlam * (np.sin(np.radians(lat + 0.25)) - np.sin(np.radians(lat - 0.25)))


def downstream_closure(forced_pos: NDArray, dn: NDArray) -> NDArray[np.int64]:
    """Positions of the forced cells plus all their downstream chains.

    The result is ascending, which preserves topological order because the
    global ``order`` array is topologically sorted.
    """
    keep = np.zeros(len(dn), dtype=bool)
    for p in np.asarray(forced_pos):
        p = int(p)
        while p >= 0 and not keep[p]:
            keep[p] = True
            p = int(dn[p])
    return np.where(keep)[0]


def topo_accumulate(values: NDArray, dn: NDArray) -> NDArray:
    """Accumulate values down-network: each node receives everything upstream.

    Parameters
    ----------
    values : NDArray
        Shape ``(N,)`` or ``(T, N)`` - per-node values (e.g. cell areas, or a
        Q' time series). The last axis must be the node axis.
    dn : NDArray
        Shape ``(N,)`` downstream position per node (-1 for terminals), with
        ``dn[i] > i`` for every non-terminal (topological ordering).
    """
    dn = np.asarray(dn)
    nonterm = dn >= 0
    assert (dn[nonterm] > np.where(nonterm)[0]).all(), "dn is not topologically ordered"
    acc = np.array(values, dtype=np.float64, copy=True)
    for i in range(dn.shape[0]):
        if dn[i] >= 0:
            acc[..., dn[i]] += acc[..., i]
    return acc


def snap_gauges(
    gauges: pd.DataFrame,
    upstream_area: dict[int, float],
    tolerance: tuple[float, float] = DA_TOLERANCE,
) -> pd.DataFrame:
    """Snap gauges to the drainage-area-matched cell in their 3x3 neighborhood.

    For each gauge, every in-network cell within one cell of the nearest
    center is considered and the one minimizing ``|log(cell_area / gauge_DA)|``
    is chosen. Gauges whose best match falls outside ``tolerance`` are
    dropped, and when several gauges land on one cell only the best match is
    kept.

    Parameters
    ----------
    gauges : pd.DataFrame
        Must have ``STAID``, ``LAT_GAGE``, ``LNG_GAGE``, ``DRAIN_SQKM``.
    upstream_area : dict[int, float]
        Drained area (km^2) per in-network flat cell id.

    Returns
    -------
    pd.DataFrame
        Input rows that matched, with ``cell``, ``da_ratio``, ``log_err``.
    """
    nearest = assign_cells(gauges.LAT_GAGE.values, gauges.LNG_GAGE.values)
    cells, ratios, errs = [], [], []
    for cell0, da in zip(nearest, gauges.DRAIN_SQKM.values, strict=True):
        best, best_err = -1, np.inf
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = int(cell0 + dr * NCOLS + dc)
                area = upstream_area.get(cell)
                if area is not None:
                    err = abs(np.log(area / da))
                    if err < best_err:
                        best, best_err = cell, err
        cells.append(best)
        errs.append(best_err)
        ratios.append(upstream_area[best] / da if best >= 0 else np.nan)

    out = gauges.copy()
    out["cell"], out["da_ratio"], out["log_err"] = cells, ratios, errs
    out = out[(out.da_ratio > tolerance[0]) & (out.da_ratio < tolerance[1])]
    return out.sort_values("log_err").drop_duplicates("cell").sort_index()
