"""Tests for ddr_engine.gridded.forcing — area-weighted redistribution of catchment Q' to cells."""

import geopandas as gpd
import numpy as np
import pytest
from ddr_engine.gridded.forcing import apply_weights, area_weights
from shapely.geometry import box

# cell 138445 spans (-77.5, 40.0)-(-77.0, 40.5); 138446 spans (-77.0, 40.0)-(-76.5, 40.5)
CELLS = [138445, 138446]


def _catchments(geoms: list, comids: list[int]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"COMID": comids}, geometry=geoms, crs="EPSG:4326")


class TestAreaWeights:
    def test_catchment_inside_one_cell_gets_weight_one(self) -> None:
        gdf = _catchments([box(-77.4, 40.1, -77.2, 40.3)], [1])
        w = area_weights(gdf, CELLS)
        assert len(w) == 1
        assert w.iloc[0]["cell"] == 138445
        assert w.iloc[0]["weight"] == pytest.approx(1.0)

    def test_straddling_catchment_splits_by_area(self) -> None:
        # square centred on the shared edge at -77.0: half in each cell
        gdf = _catchments([box(-77.1, 40.1, -76.9, 40.3)], [1])
        w = area_weights(gdf, CELLS).set_index("cell")["weight"]
        assert w.loc[138445] == pytest.approx(0.5)
        assert w.loc[138446] == pytest.approx(0.5)

    def test_weights_sum_to_one_per_catchment(self) -> None:
        gdf = _catchments([box(-77.4, 40.1, -77.2, 40.3), box(-77.2, 40.05, -76.8, 40.45)], [1, 2])
        w = area_weights(gdf, CELLS)
        sums = w.groupby("COMID")["weight"].sum()
        assert np.allclose(sums.to_numpy(), 1.0)

    def test_catchment_outside_all_cells_is_dropped(self) -> None:
        gdf = _catchments([box(0.1, 0.1, 0.2, 0.2)], [9])
        assert area_weights(gdf, CELLS).empty

    def test_partially_outside_catchment_renormalizes(self) -> None:
        """A catchment half outside the grid still routes all its water to covered cells."""
        gdf = _catchments([box(-77.6, 40.1, -77.4, 40.3)], [1])  # half west of the cell edge
        w = area_weights(gdf, CELLS)
        assert w["weight"].sum() == pytest.approx(1.0)


class TestApplyWeights:
    def test_conserves_volume(self) -> None:
        gdf = _catchments([box(-77.1, 40.1, -76.9, 40.3), box(-77.4, 40.1, -77.2, 40.3)], [1, 2])
        w = area_weights(gdf, CELLS)
        qr = np.array([[10.0, 4.0], [20.0, 8.0]])  # (T=2, n_comid=2)
        out = apply_weights(qr, np.array([1, 2]), w, np.array(CELLS))
        assert out.shape == (2, 2)
        assert np.allclose(out.sum(axis=1), qr.sum(axis=1))
        assert out[0, 0] == pytest.approx(10.0 * 0.5 + 4.0)  # cell 138445

    def test_nan_treated_as_zero(self) -> None:
        gdf = _catchments([box(-77.4, 40.1, -77.2, 40.3)], [1])
        w = area_weights(gdf, CELLS)
        out = apply_weights(np.array([[np.nan]]), np.array([1]), w, np.array(CELLS))
        assert out[0, 0] == 0.0

    def test_comid_without_weights_is_ignored(self) -> None:
        gdf = _catchments([box(-77.4, 40.1, -77.2, 40.3)], [1])
        w = area_weights(gdf, CELLS)
        out = apply_weights(np.array([[5.0, 99.0]]), np.array([1, 7]), w, np.array(CELLS))
        assert out[0, 0] == pytest.approx(5.0)
        assert out.sum() == pytest.approx(5.0)
