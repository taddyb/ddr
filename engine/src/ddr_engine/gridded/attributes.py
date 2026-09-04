"""Per-cell attributes: DDM30 0.5-degree cell polygons + extractrs zonal extraction.

Cell polygons use the flat cell-id scheme (id = row * 720 + col, row 0 at
-55.75 degrees; mirrored in ddr_benchmarks.gridded). Rasters are aggregated to
cells with extractrs coverage-weighted zonal means — never resampled from
basin-averaged products.
"""

from __future__ import annotations

from pathlib import Path

import extractrs  # noqa: F401 - registers the .extrs xarray accessor
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from shapely.geometry import box

NCOLS = 720
CELL_DEG = 0.5
LAT0 = -55.75  # centre of row 0 (southernmost)
LON0 = -179.75  # centre of col 0


def cell_polygons(cell_ids: np.typing.ArrayLike) -> gpd.GeoDataFrame:
    """0.5-degree cell polygons (EPSG:4326) for flat DDM30 cell ids, order preserved."""
    ids = np.atleast_1d(np.asarray(cell_ids, dtype=np.int64))
    lat_c = LAT0 + (ids // NCOLS) * CELL_DEG
    lon_c = LON0 + (ids % NCOLS) * CELL_DEG
    h = CELL_DEG / 2
    geoms = [box(x - h, y - h, x + h, y + h) for x, y in zip(lon_c, lat_c, strict=True)]
    return gpd.GeoDataFrame({"cell": ids}, geometry=geoms, crs="EPSG:4326")


def extract_cell_attributes(
    rasters: dict[str, Path | str],
    cell_ids: np.typing.ArrayLike,
    scales: dict[str, float] | None = None,
    valid_range: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Coverage-weighted zonal mean of each raster over each cell.

    Returns a DataFrame indexed by cell id (input order), one column per raster
    name; cells without raster coverage are NaN. ``scales`` multiplies a
    column after extraction (e.g. HiHydroSoil int storage -> 1e-4).
    ``valid_range`` masks raw raster values outside (lo, hi) before
    aggregation, so unflagged fill/overflow pixels cannot poison a cell mean.
    """
    cells = cell_polygons(cell_ids)
    out = pd.DataFrame(index=pd.Index(cells["cell"], name="cell"))
    for name, path in rasters.items():
        # band 1 is the data; geedim downloads carry a trailing FILL_MASK band
        da = rioxarray.open_rasterio(path, masked=True).isel(band=0, drop=True)
        if valid_range and name in valid_range:
            lo, hi = valid_range[name]
            da = da.where((da >= lo) & (da <= hi))
        # work in the raster's CRS: exact polygons, no raster resampling
        zones = cells.to_crs(da.rio.crs)
        xmin, ymin, xmax, ymax = zones.total_bounds
        try:
            da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
        except Exception:  # noqa: BLE001 - no overlap at all -> all-NaN column
            out[name] = np.nan
            continue
        covered = zones[zones.intersects(box(*da.rio.bounds()))]
        if covered.empty:
            out[name] = np.nan
            continue
        res = da.rename(name).extrs.zonal_stats(covered, stat="mean", id_col="cell")
        col = res[name].to_series() if hasattr(res, "data_vars") else res.to_series()
        out[name] = col.reindex(out.index)
        if scales and name in scales:
            out[name] *= scales[name]
    return out


def cell_area_km2(cell_ids: np.typing.ArrayLike) -> np.ndarray:
    """Spherical area (km²) of 0.5-degree cells; same formula as ddr_benchmarks.gridded."""
    ids = np.atleast_1d(np.asarray(cell_ids, dtype=np.int64))
    lat = LAT0 + (ids // NCOLS) * CELL_DEG
    r_earth = 6371.0
    half = np.radians(CELL_DEG / 2)
    return (
        r_earth**2 * np.radians(CELL_DEG) * (np.sin(np.radians(lat) + half) - np.sin(np.radians(lat) - half))
    )


def table_to_grid(df: pd.DataFrame, grid_shape: tuple[int, int]) -> xr.Dataset:
    """Scatter a cell-indexed table onto the (lat, lon) DDM30 grid; NaN where no cell."""
    nrows, ncols = grid_shape
    ids = np.asarray(df.index, dtype=np.int64)
    rows, cols = ids // ncols, ids % ncols
    data = {}
    for name in df.columns:
        grid = np.full(grid_shape, np.nan)
        grid[rows, cols] = df[name].to_numpy(dtype=float)
        data[name] = (("lat", "lon"), grid)
    lat = LAT0 + np.arange(nrows) * CELL_DEG
    lon = LON0 + np.arange(ncols) * CELL_DEG
    return xr.Dataset(data, coords={"lat": lat, "lon": lon})
