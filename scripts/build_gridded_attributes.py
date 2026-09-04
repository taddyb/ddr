"""Build the DDM30 per-cell attribute store for ddr, ddrs and ISIMIP-style gridded routing.

Reads the native rasters acquired under /mnt/ssd1/data/gridded_attrs/ (see
wiki/gridded-attribute-sources.md), aggregates each onto DDM30 0.5-degree cells
with extractrs coverage-weighted means, and writes two files:

  <out>/ddm30_conus_attributes.nc       dim ``COMID`` = flat DDM30 cell id, one
                                        float variable per attribute -- the same
                                        layout as merit_global_attributes_v2.nc,
                                        so ddr's AttributesReader and ddrs load
                                        it unchanged (point ``attributes`` at it)
  <out>/ddm30_conus_attributes_grid.nc  the same variables on the (lat, lon)
                                        280 x 720 DDM30 grid (ISIMIP-style)

Variable names and units follow merit_global_attributes_v2.nc where a
counterpart exists (meanelevation m, SoilGrids1km_* %, NDVI 0-1, snow_fraction
0-1, FW 0-1, catchsize km2, meanslope degrees, log10_uparea log10 km2, meanP
mm/yr, meanTa degC, ETPOT_Hargr mm/yr, aridity PET/P); HiHydroSoil variables
use the paper's names. Terrain comes from the MERIT Hydro 3" tiles, climate
from WorldClim 2.1 30" (bio1, bio12, monthly tmin/tmax -> Hargreaves PET).

Usage:
    uv run python scripts/build_gridded_attributes.py [--bbox XMIN YMIN XMAX YMAX]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import zarr
from ddr_engine.gridded.attributes import (
    cell_area_km2,
    cell_polygons,
    extract_cell_attributes,
    extract_from_dataarray,
    table_to_grid,
)
from ddr_engine.gridded.climate import hargreaves_pet_mm_yr
from ddr_engine.gridded.terrain import slope_degrees
from shapely.geometry import box

CONUS_BBOX = (-125.0, 24.0, -66.0, 53.0)
DATA_ROOT = Path("/mnt/ssd1/data/gridded_attrs")
ADJACENCY = Path("data/ddm30/ddm30_adjacency.zarr")
MERIT_TILES = Path("/mnt/ssd1/data/merit_hydro/rasters")  # from scripts/acquire_merit_hydro.py
MERIT_RES_DEG = 1 / 1200
WORLDCLIM = DATA_ROOT / "worldclim"  # wc2.1_30s_{bio,tmin,tmax}.zip, read in place via GDAL vsizip

# variable name -> (raster relative to DATA_ROOT, multiplicative scale to target units)
RASTERS: dict[str, tuple[str, float]] = {
    "meanelevation": ("gee/gmted2010_mea.tif", 1.0),
    "SoilGrids1km_clay": ("soilgrids/clay_0-5cm_mean_1000.tif", 0.1),
    "SoilGrids1km_sand": ("soilgrids/sand_0-5cm_mean_1000.tif", 0.1),
    "SoilGrids1km_silt": ("soilgrids/silt_0-5cm_mean_1000.tif", 0.1),
    "NDVI": ("gee/ndvi_mod13_mean.tif", 1e-4),
    "snow_fraction": ("gee/snow_mod10_mean.tif", 0.01),
    "FW": ("gee/glwd_v2_openwater_pct.tif", 0.01),
    "Ksat": ("gee/hihydrosoil_ksat.tif", 1e-4),
    "ALPHA": ("gee/hihydrosoil_alpha.tif", 1e-4),
    "N": ("gee/hihydrosoil_N.tif", 1e-4),
    "ORMC": ("gee/hihydrosoil_ormc.tif", 1e-4),
    "WCpF2": ("gee/hihydrosoil_wcpf2.tif", 1e-4),
    "WCsat": ("gee/hihydrosoil_wcsat.tif", 1e-4),
}

# raw-value sanity bounds applied before aggregation (HiHydroSoil has unflagged int32-max overflow pixels)
VALID_RANGE: dict[str, tuple[float, float]] = dict.fromkeys(
    ("Ksat", "ALPHA", "N", "ORMC", "WCpF2", "WCsat"), (0, 10000000.0)
)

log = logging.getLogger("build_gridded_attributes")


def conus_cells(adjacency: Path, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """DDM30 cell ids (topological order) whose centres fall inside bbox."""
    g = zarr.open_group(adjacency, mode="r")
    order, lat, lon = g["order"][:], g["lat"][:], g["lon"][:]
    xmin, ymin, xmax, ymax = bbox
    keep = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    return order[keep]


def merit_terrain(cells: np.ndarray) -> pd.DataFrame:
    """Meanslope (degrees, mean of 3" slope) and log10_uparea (log10 km2 of max upa) per cell.

    Processed per 5x5-degree MERIT tile; tile edges align with 0.5-degree cell
    edges, so each cell (assigned by centroid) is fully inside one tile.
    """
    polys = cell_polygons(cells)
    centroids = polys.geometry.centroid
    parts = []
    for elv_path in sorted((MERIT_TILES / "elv").glob("*.tif")):
        elv = rioxarray.open_rasterio(elv_path, masked=True).isel(band=0, drop=True)
        tile_cells = polys[centroids.within(box(*elv.rio.bounds()))]
        if tile_cells.empty:
            continue
        slope = elv.copy(data=slope_degrees(elv.values, elv.y.values, MERIT_RES_DEG))
        upa_path = MERIT_TILES / "upa" / elv_path.name.replace("_elv_", "_upa_")
        upa = rioxarray.open_rasterio(upa_path, masked=True).isel(band=0, drop=True)
        parts.append(
            pd.DataFrame(
                {
                    "meanslope": extract_from_dataarray(slope, tile_cells, stat="mean"),
                    "log10_uparea": np.log10(extract_from_dataarray(upa, tile_cells, stat="max")),
                }
            )
        )
        log.info("terrain: %s -> %d cells", elv_path.stem[-7:], len(tile_cells))
    return pd.concat(parts).reindex(cells)


def _worldclim(zip_name: str, tif: str, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    path = f"zip://{WORLDCLIM / zip_name}!{tif}"
    da = rioxarray.open_rasterio(path, masked=True).isel(band=0, drop=True)
    return da.rio.clip_box(*bbox).load()


def worldclim_climate(cells: np.ndarray, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """MeanP (mm/yr), meanTa (degC), ETPOT_Hargr (mm/yr, Hargreaves from monthly tmin/tmax), aridity = PET/P."""
    polys = cell_polygons(cells)
    mean_p = _worldclim("wc2.1_30s_bio.zip", "wc2.1_30s_bio_12.tif", bbox)
    mean_ta = _worldclim("wc2.1_30s_bio.zip", "wc2.1_30s_bio_1.tif", bbox)
    tmin = np.stack(
        [_worldclim("wc2.1_30s_tmin.zip", f"wc2.1_30s_tmin_{m:02d}.tif", bbox).values for m in range(1, 13)]
    )
    tmax = np.stack(
        [_worldclim("wc2.1_30s_tmax.zip", f"wc2.1_30s_tmax_{m:02d}.tif", bbox).values for m in range(1, 13)]
    )
    pet = mean_ta.copy(data=hargreaves_pet_mm_yr(tmin, tmax, mean_ta.y.values))
    df = pd.DataFrame(
        {
            "meanP": extract_from_dataarray(mean_p, polys),
            "meanTa": extract_from_dataarray(mean_ta, polys),
            "ETPOT_Hargr": extract_from_dataarray(pet, polys),
        }
    ).reindex(cells)
    df["aridity"] = df["ETPOT_Hargr"] / df["meanP"]
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", type=Path, default=ADJACENCY)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DATA_ROOT)
    parser.add_argument("--bbox", nargs=4, type=float, default=list(CONUS_BBOX))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cells = conus_cells(args.adjacency, tuple(args.bbox))
    log.info("%d DDM30 cells in bbox", len(cells))
    rasters = {name: args.data_root / rel for name, (rel, _) in RASTERS.items()}
    scales = {name: scale for name, (_, scale) in RASTERS.items()}
    df = extract_cell_attributes(rasters, cells, scales=scales, valid_range=VALID_RANGE)
    df["catchsize"] = cell_area_km2(cells)
    df = df.join(merit_terrain(cells)).join(worldclim_climate(cells, tuple(args.bbox)))
    for name in df.columns:
        col = df[name]
        log.info(
            "%-18s NaN %5.1f%%  min %10.4g  median %10.4g  max %10.4g",
            name,
            100 * col.isna().mean(),
            col.min(),
            col.median(),
            col.max(),
        )

    grid_shape = tuple(zarr.open_group(args.adjacency, mode="r").attrs["grid_shape"])
    table = xr.Dataset.from_dataframe(df.rename_axis("COMID"))
    table.attrs["description"] = "DDM30 0.5-degree cell attributes; COMID = row*720+col flat cell id"
    table.to_netcdf(args.out / "ddm30_conus_attributes.nc")
    table_to_grid(df, grid_shape).to_netcdf(args.out / "ddm30_conus_attributes_grid.nc")
    log.info(
        "wrote %s and %s", args.out / "ddm30_conus_attributes.nc", args.out / "ddm30_conus_attributes_grid.nc"
    )


if __name__ == "__main__":
    main()
