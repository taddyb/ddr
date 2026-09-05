"""Redistribute per-catchment lateral inflow (Q') onto DDM30 cells by area.

Q' is a volumetric flux per MERIT unit catchment (m3/s), so moving it to cells is
mass-conserving *aggregation*, not interpolation of an intensive field: each
catchment's discharge is split across the cells it overlaps in proportion to the
overlapping area, and the pieces are summed per cell. This is exact for catchments
inside one cell and correct for catchments straddling a cell edge, unlike assigning
a whole catchment to the cell holding its flowline midpoint.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .attributes import cell_polygons


def area_weights(catchments: gpd.GeoDataFrame, cell_ids: np.typing.ArrayLike) -> pd.DataFrame:
    """Fraction of each catchment's area falling in each DDM30 cell.

    Returns a long DataFrame ``[COMID, cell, weight]`` whose weights sum to 1 for
    every catchment that overlaps the grid at all (a catchment partly outside the
    supplied cells is renormalised, so its whole discharge still reaches the grid).
    Catchments overlapping no cell are dropped.
    """
    cells = cell_polygons(cell_ids)
    if catchments.crs is not None and cells.crs is not None:
        catchments = catchments.to_crs(cells.crs)
    else:  # MERIT catchment shapefiles ship without a .prj; they are plain lon/lat
        catchments = catchments.set_crs(cells.crs, allow_override=True)
    pieces = gpd.overlay(catchments[["COMID", "geometry"]], cells[["cell", "geometry"]], how="intersection")
    if pieces.empty:
        return pd.DataFrame(columns=["COMID", "cell", "weight"])
    pieces["weight"] = pieces.geometry.area
    out = pieces.groupby(["COMID", "cell"], as_index=False)["weight"].sum()
    out["weight"] /= out.groupby("COMID")["weight"].transform("sum")
    return out


def apply_weights(
    qr: np.ndarray,
    comids: np.ndarray,
    weights: pd.DataFrame,
    cells: np.ndarray,
) -> np.ndarray:
    """Map ``qr`` (T, n_comid) onto (T, n_cells) with the area weights; NaN counts as zero."""
    comid_pos = pd.Series(np.arange(len(comids)), index=np.asarray(comids))
    cell_pos = pd.Series(np.arange(len(cells)), index=np.asarray(cells))
    w = weights[weights["COMID"].isin(comid_pos.index) & weights["cell"].isin(cell_pos.index)]
    rows = comid_pos.loc[w["COMID"].to_numpy()].to_numpy()
    cols = cell_pos.loc[w["cell"].to_numpy()].to_numpy()
    out = np.zeros((qr.shape[0], len(cells)), dtype=np.float64)
    contrib = np.nan_to_num(qr[:, rows]) * w["weight"].to_numpy()
    np.add.at(out, (slice(None), cols), contrib)
    return out
