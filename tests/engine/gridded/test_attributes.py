"""Tests for ddr_engine.gridded.attributes — cell polygons + extractrs zonal extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from ddr_engine.gridded.attributes import cell_polygons, extract_cell_attributes

JUNIATA_OUTLET = 138445  # row 192, col 205 -> center (40.25, -77.25)


class TestCellPolygons:
    def test_outlet_cell_bounds(self) -> None:
        gdf = cell_polygons([JUNIATA_OUTLET])
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
        assert list(gdf["cell"]) == [JUNIATA_OUTLET]
        assert gdf.geometry.iloc[0].bounds == (-77.5, 40.0, -77.0, 40.5)

    def test_half_degree_area(self) -> None:
        gdf = cell_polygons([JUNIATA_OUTLET, JUNIATA_OUTLET - 720])
        assert np.allclose(gdf.geometry.area, 0.25)  # 0.5 x 0.5 degrees

    def test_order_preserved(self) -> None:
        ids = [138444, 138445, 138443]
        assert list(cell_polygons(ids)["cell"]) == ids


def _write_tif(path: Path, values: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    """Write a small EPSG:4326 GeoTIFF with the given north-up bounds."""
    xmin, ymin, xmax, ymax = bounds
    ny, nx = values.shape
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": ymax - dy * (np.arange(ny) + 0.5),
            "x": xmin + dx * (np.arange(nx) + 0.5),
        },
    )
    da.rio.write_crs("EPSG:4326").rio.to_raster(path)


class TestExtractCellAttributes:
    @pytest.fixture()
    def two_cell_tif(self, tmp_path: Path) -> Path:
        # covers cells 138445 (-77.5..-77.0) and 138446 (-77.0..-76.5), lat 40.0..40.5
        values = np.zeros((10, 20), dtype="float32")
        values[:, :10] = 3.0
        values[:, 10:] = 7.0
        path = tmp_path / "toy.tif"
        _write_tif(path, values, (-77.5, 40.0, -76.5, 40.5))
        return path

    def test_per_cell_means(self, two_cell_tif: Path) -> None:
        df = extract_cell_attributes({"toy": two_cell_tif}, [138445, 138446])
        assert list(df.index) == [138445, 138446]
        assert df.loc[138445, "toy"] == pytest.approx(3.0)
        assert df.loc[138446, "toy"] == pytest.approx(7.0)

    def test_scale_factor_applied(self, two_cell_tif: Path) -> None:
        df = extract_cell_attributes({"toy": two_cell_tif}, [138445], scales={"toy": 1e-4})
        assert df.loc[138445, "toy"] == pytest.approx(3e-4)

    def test_cell_outside_raster_is_nan(self, two_cell_tif: Path) -> None:
        df = extract_cell_attributes({"toy": two_cell_tif}, [138445, 0])
        assert np.isnan(df.loc[0, "toy"])

    def test_returns_dataframe_with_named_index(self, two_cell_tif: Path) -> None:
        df = extract_cell_attributes({"toy": two_cell_tif}, [138445])
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "cell"


class TestProjectedRaster:
    def test_projected_raster_is_reprojected(self, tmp_path: Path) -> None:
        # constant field written in a projected CRS covering the Juniata outlet cell
        xmin, ymin, xmax, ymax = -78.0, 39.5, -76.5, 41.0
        da = xr.DataArray(
            np.full((30, 30), 42.0, dtype="float32"),
            dims=("y", "x"),
            coords={
                "y": ymax - (ymax - ymin) / 30 * (np.arange(30) + 0.5),
                "x": xmin + (xmax - xmin) / 30 * (np.arange(30) + 0.5),
            },
        ).rio.write_crs("EPSG:4326")
        path = tmp_path / "projected.tif"
        da.rio.reproject("ESRI:54052").rio.to_raster(path)  # Goode Homolosine, like SoilGrids
        df = extract_cell_attributes({"v": path}, [JUNIATA_OUTLET])
        assert df.loc[JUNIATA_OUTLET, "v"] == pytest.approx(42.0)


class TestCellAreaKm2:
    def test_equator_cell_area(self) -> None:
        from ddr_engine.gridded.attributes import cell_area_km2

        # row 112 -> lat centre 0.25 (cell spans 0..0.5 deg); ~55.6 km x ~55.6 km
        area = cell_area_km2([112 * 720])
        assert area[0] == pytest.approx(3092, rel=0.01)

    def test_area_shrinks_with_latitude(self) -> None:
        from ddr_engine.gridded.attributes import cell_area_km2

        a = cell_area_km2([112 * 720, JUNIATA_OUTLET])
        assert a[1] < a[0]
        assert a[1] == pytest.approx(3092 * np.cos(np.radians(40.25)), rel=0.01)


class TestTableToGrid:
    def test_scatter_to_grid(self) -> None:
        from ddr_engine.gridded.attributes import table_to_grid

        df = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.Index([0, 720 + 5], name="cell"))
        ds = table_to_grid(df, grid_shape=(280, 720))
        assert ds["v"].dims == ("lat", "lon")
        assert ds["v"].shape == (280, 720)
        assert ds["v"].values[0, 0] == 1.0
        assert ds["v"].values[1, 5] == 2.0
        assert np.isnan(ds["v"].values[2, 2])
        assert ds["lat"].values[0] == -55.75 and ds["lon"].values[0] == -179.75
        assert ds["lat"].values[-1] == pytest.approx(83.75)


class TestValidRange:
    def test_out_of_range_pixels_masked_before_mean(self, tmp_path: Path) -> None:
        values = np.full((10, 10), 3.0, dtype="float32")
        values[0, 0] = 1e9  # fill/overflow artifact
        path = tmp_path / "fill.tif"
        _write_tif(path, values, (-77.5, 40.0, -77.0, 40.5))
        poisoned = extract_cell_attributes({"v": path}, [JUNIATA_OUTLET])
        assert poisoned.loc[JUNIATA_OUTLET, "v"] > 1e6
        clean = extract_cell_attributes({"v": path}, [JUNIATA_OUTLET], valid_range={"v": (0, 100)})
        assert clean.loc[JUNIATA_OUTLET, "v"] == pytest.approx(3.0)
