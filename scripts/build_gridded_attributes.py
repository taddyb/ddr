"""Build the DDM30 per-cell attribute store for ddr, ddrs and ISIMIP-style gridded routing.

Aggregates the native rasters acquired under /mnt/ssd1/data/gridded_attrs/ (see
wiki/gridded-attribute-sources.md) onto DDM30 0.5-degree cells with extractrs
coverage-weighted means, and writes the same table three ways:

  <out>/<stem>.nc        dim ``COMID`` = flat DDM30 cell id, one float variable per
                         attribute -- the layout of merit_global_attributes_v2.nc, so
                         ddr's AttributesReader (MERIT path) and ddrs load it unchanged
  <out>/<stem>.ic        icechunk repo, dim ``divide_id`` = cell id, for ddr's icechunk
                         attribute path (``read_ic``)
  <out>/<stem>_grid.nc   the same variables on the (lat, lon) 280 x 720 DDM30 grid,
                         i.e. the ISIMIP flowdir grid

``--global`` builds every cell of the ISIMIP grid (67,424) from the global rasters in
``gee_global/``; the default builds the CONUS subset from ``gee/``. Work proceeds in
30-degree blocks so a world raster is never held in memory.

Variable names and units follow merit_global_attributes_v2.nc where a counterpart
exists (meanelevation m, meanslope degrees, SoilGrids1km_* %, NDVI, snow_fraction,
snowfall_fraction, FW, seasonality_P/PET, catchsize km2, log10_uparea log10 km2,
meanP mm/yr, meanTa degC, ETPOT_Hargr mm/yr, aridity PET/P); HiHydroSoil variables
use the paper's names. Elevation and slope both come from GMTED2010 (MERIT Hydro
stops at 60N, which would drop 17,238 of the grid's cells).

Usage:
    uv run python scripts/build_gridded_attributes.py [--global]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import zarr
from ddr_benchmarks.gridded import topo_accumulate
from ddr_engine.gridded.attributes import (
    blocks,
    cell_area_km2,
    cell_polygons,
    cells_in_box,
    extract_cell_attributes,
    extract_from_dataarray,
    table_to_grid,
)
from ddr_engine.gridded.climate import (
    hargreaves_pet_monthly_mm,
    seasonality_index,
    snowfall_fraction,
)
from ddr_engine.gridded.terrain import slope_from_dataarray

CONUS_BBOX = (-125.0, 24.0, -66.0, 53.0)
GLOBAL_BBOX = (-180.0, -56.0, 180.0, 84.0)  # DDM30 rows span -55.75..83.75
BLOCK_DEG = 30.0  # processing window; bounds peak memory on global builds
DEM_BUFFER_DEG = 0.05  # halo so slope gradients are two-sided at block edges

DATA_ROOT = Path("/mnt/ssd1/data/gridded_attrs")
OUT_ROOT = Path("/mnt/ssd1/data/icechunk")  # alongside merit_global_attributes_v2.nc
ADJACENCY = Path("data/ddm30/ddm30_adjacency.zarr")
WORLDCLIM = DATA_ROOT / "worldclim"  # wc2.1_30s_{bio,tmin,tmax,prec}.zip, read via vsizip

# variable name -> (raster path relative to the mode's raster dir, scale to target units)
RASTERS: dict[str, tuple[str, float]] = {
    "meanelevation": ("gmted2010_mea.tif", 1.0),
    "NDVI": ("ndvi_mod13_mean.tif", 1e-4),
    "snow_fraction": ("snow_mod10_mean.tif", 0.01),
    "FW": ("glwd_v2_openwater_pct.tif", 0.01),
    "Ksat": ("hihydrosoil_ksat.tif", 1e-4),
    "ALPHA": ("hihydrosoil_alpha.tif", 1e-4),
    "N": ("hihydrosoil_N.tif", 1e-4),
    "ORMC": ("hihydrosoil_ormc.tif", 1e-4),
    "WCpF2": ("hihydrosoil_wcpf2.tif", 1e-4),
    "WCsat": ("hihydrosoil_wcsat.tif", 1e-4),
}
# soil rasters are global files, not per-mode
SOILGRIDS: dict[str, tuple[str, float]] = {
    "SoilGrids1km_clay": ("soilgrids/clay_0-5cm_mean_1000.tif", 0.1),
    "SoilGrids1km_sand": ("soilgrids/sand_0-5cm_mean_1000.tif", 0.1),
    "SoilGrids1km_silt": ("soilgrids/silt_0-5cm_mean_1000.tif", 0.1),
}
# raw-value bounds applied before aggregation (HiHydroSoil has unflagged int32-max pixels)
VALID_RANGE: dict[str, tuple[float, float]] = dict.fromkeys(
    ("Ksat", "ALPHA", "N", "ORMC", "WCpF2", "WCsat"), (0, 10000000.0)
)

log = logging.getLogger("build_gridded_attributes")


def grid_cells(adjacency: Path, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """DDM30 cell ids (topological order) whose centres fall inside bbox."""
    g = zarr.open_group(adjacency, mode="r")
    order, lat, lon = g["order"][:], g["lat"][:], g["lon"][:]
    xmin, ymin, xmax, ymax = bbox
    keep = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    return order[keep]


def network_log10_uparea(adjacency: Path, cells: np.ndarray) -> pd.Series:
    """log10 of the DDM30-network-accumulated area (km2) at each cell.

    Cumulative area function over the routing network: a headwater cell is its own
    catchsize, each downstream cell adds everything upstream.
    """
    g = zarr.open_group(adjacency, mode="r")
    order = g["order"][:]
    dn = np.full(len(order), -1, dtype=np.int64)
    dn[g["indices_1"][:]] = g["indices_0"][:]
    acc = topo_accumulate(cell_area_km2(order), dn)
    return pd.Series(np.log10(acc), index=pd.Index(order, name="cell")).reindex(cells)


def _window(path: str, bbox: tuple[float, float, float, float], buffer: float = 0.0) -> xr.DataArray:
    """Read one block window of a (possibly zipped) raster into memory."""
    xmin, ymin, xmax, ymax = bbox
    da = rioxarray.open_rasterio(path, masked=True).isel(band=0, drop=True)
    return da.rio.clip_box(xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer).load()


def _wc(zip_name: str, tif: str, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    return _window(f"zip://{WORLDCLIM / zip_name}!{tif}", bbox)


def block_climate(polys: gpd.GeoDataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """WorldClim-derived climate attributes for the cells of one block.

    meanP (mm/yr), meanTa (degC), ETPOT_Hargr (mm/yr, Hargreaves from monthly
    tmin/tmax), aridity = PET/P, seasonality_P and seasonality_PET (Walsh & Lawler
    index), snowfall_fraction (precipitation weighted by a temperature ramp).
    """
    mean_p = _wc("wc2.1_30s_bio.zip", "wc2.1_30s_bio_12.tif", bbox)
    mean_ta = _wc("wc2.1_30s_bio.zip", "wc2.1_30s_bio_1.tif", bbox)
    tmin = np.stack(
        [_wc("wc2.1_30s_tmin.zip", f"wc2.1_30s_tmin_{m:02d}.tif", bbox).values for m in range(1, 13)]
    )
    tmax = np.stack(
        [_wc("wc2.1_30s_tmax.zip", f"wc2.1_30s_tmax_{m:02d}.tif", bbox).values for m in range(1, 13)]
    )
    prec = np.stack(
        [_wc("wc2.1_30s_prec.zip", f"wc2.1_30s_prec_{m:02d}.tif", bbox).values for m in range(1, 13)]
    )
    pet_monthly = hargreaves_pet_monthly_mm(tmin, tmax, mean_ta.y.values)
    derived = {
        "ETPOT_Hargr": pet_monthly.sum(axis=0),
        "seasonality_P": seasonality_index(prec),
        "seasonality_PET": seasonality_index(pet_monthly),
        "snowfall_fraction": snowfall_fraction(prec, 0.5 * (tmin + tmax)),
    }
    df = pd.DataFrame(
        {
            "meanP": extract_from_dataarray(mean_p, polys),
            "meanTa": extract_from_dataarray(mean_ta, polys),
            **{k: extract_from_dataarray(mean_ta.copy(data=v), polys) for k, v in derived.items()},
        }
    )
    df["aridity"] = df["ETPOT_Hargr"] / df["meanP"]
    return df


def block_slope(
    polys: gpd.GeoDataFrame, bbox: tuple[float, float, float, float], dem_path: Path
) -> pd.Series:
    """Meanslope (degrees) per cell from the GMTED DEM window."""
    dem = _window(str(dem_path), bbox, buffer=DEM_BUFFER_DEG)
    return extract_from_dataarray(slope_from_dataarray(dem), polys)


def build(cells: np.ndarray, bbox: tuple[float, float, float, float], raster_dir: Path) -> pd.DataFrame:
    """Aggregate every raster + derived attribute onto ``cells``, one block at a time."""
    rasters = {name: raster_dir / rel for name, (rel, _) in RASTERS.items()}
    rasters |= {name: DATA_ROOT / rel for name, (rel, _) in SOILGRIDS.items()}
    scales = {name: sc for name, (_, sc) in (RASTERS | SOILGRIDS).items()}
    parts = []
    for i, blk in enumerate(blocks(bbox, BLOCK_DEG), start=1):
        blk_cells = cells_in_box(cells, blk)
        if len(blk_cells) == 0:
            continue
        polys = cell_polygons(blk_cells)
        df = extract_cell_attributes(rasters, blk_cells, scales=scales, valid_range=VALID_RANGE)
        df["meanslope"] = block_slope(polys, blk, rasters["meanelevation"]).reindex(df.index)
        df = df.join(block_climate(polys, blk))
        parts.append(df)
        log.info("block %d %s -> %d cells", i, blk, len(blk_cells))
    return pd.concat(parts).reindex(cells)


def write_icechunk(df: pd.DataFrame, path: Path, description: str) -> None:
    """Write the attribute table to a local icechunk repo with dim ``divide_id`` (= cell id)."""
    import icechunk as ic

    ds = xr.Dataset.from_dataframe(df.rename_axis("divide_id"))
    ds.attrs["description"] = description
    repo = ic.Repository.open_or_create(ic.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    ds.to_zarr(session.store, mode="w", consolidated=False, zarr_format=3)
    session.commit("gridded DDM30 attributes")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", type=Path, default=ADJACENCY)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--global",
        dest="global_mode",
        action="store_true",
        help="build every cell of the ISIMIP grid from the global rasters in gee_global/",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bbox = GLOBAL_BBOX if args.global_mode else CONUS_BBOX
    raster_dir = DATA_ROOT / ("gee_global" if args.global_mode else "gee")
    stem = "ddm30_global_attributes" if args.global_mode else "ddm30_conus_attributes"

    cells = grid_cells(args.adjacency, bbox)
    log.info("%d DDM30 cells; rasters from %s", len(cells), raster_dir)
    df = build(cells, bbox, raster_dir)
    df["catchsize"] = cell_area_km2(cells)
    df["log10_uparea"] = network_log10_uparea(args.adjacency, cells)
    for name in sorted(df.columns):
        c = df[name]
        log.info(
            "%-18s NaN %5.1f%%  min %10.4g  median %10.4g  max %10.4g",
            name,
            100 * c.isna().mean(),
            c.min(),
            c.median(),
            c.max(),
        )

    grid_shape = tuple(zarr.open_group(args.adjacency, mode="r").attrs["grid_shape"])
    description = "DDM30 0.5-degree cell attributes; cell id = row*720+col (row 0 = southernmost)"
    args.out.mkdir(parents=True, exist_ok=True)

    table = xr.Dataset.from_dataframe(df.rename_axis("COMID"))
    table.attrs["description"] = description
    table.to_netcdf(args.out / f"{stem}.nc")

    grid = table_to_grid(df, grid_shape)
    grid.attrs["description"] = description
    grid.to_netcdf(args.out / f"{stem}_grid.nc")

    write_icechunk(df, args.out / f"{stem}.ic", description)
    log.info("wrote %s{.nc,.ic,_grid.nc}", args.out / stem)


if __name__ == "__main__":
    main()
