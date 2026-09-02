"""Integration tests: DDM30 netCDFs -> adjacency zarr -> gauge subsets."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from ddr_engine.core.zarr_io import coo_from_zarr
from ddr_engine.gridded import build_ddm30_adjacency, build_gauge_adjacencies, snap_to_cell

from .conftest import EXPECTED_VALID, SANDBOX_LAT, SANDBOX_LON

DDM30_DIR = Path(__file__).parents[3] / "data" / "ddm30"


@pytest.fixture(scope="module")
def built_zarr(sandbox_nc_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "ddm30_adjacency.zarr"
    return build_ddm30_adjacency(sandbox_nc_dir, out)


class TestBuildDdm30Adjacency:
    """End-to-end build on the sandbox netCDFs."""

    def test_roundtrip_autodetects_ddm30(self, built_zarr):
        coo, id_order = coo_from_zarr(built_zarr)
        assert coo.shape == (10, 10)
        assert coo.nnz == 5
        assert sorted(id_order) == sorted(EXPECTED_VALID)

    def test_attribute_arrays_aligned_to_order(self, built_zarr):
        root = zarr.open_group(store=built_zarr, mode="r")
        order = root["order"][:]
        n = len(order)
        for name in ["length_m", "slope", "lat", "lon", "basin"]:
            assert root[name].shape == (n,), name

    def test_slope_values_follow_order(self, built_zarr):
        """Slope was 0.001 on every sandbox cell for routing use."""
        root = zarr.open_group(store=built_zarr, mode="r")
        np.testing.assert_allclose(root["slope"][:], 0.001, rtol=1e-6)

    def test_latlon_match_cell_ids(self, built_zarr):
        root = zarr.open_group(store=built_zarr, mode="r")
        order = root["order"][:]
        ncols = root.attrs["grid_shape"][1]
        np.testing.assert_allclose(root["lat"][:], SANDBOX_LAT[order // ncols], rtol=1e-6)
        np.testing.assert_allclose(root["lon"][:], SANDBOX_LON[order % ncols], rtol=1e-6)

    def test_grid_shape_attr(self, built_zarr):
        root = zarr.open_group(store=built_zarr, mode="r")
        assert root.attrs["grid_shape"] == [4, 4]
        assert root.attrs["geodataset"] == "ddm30"

    def test_refuses_overwrite(self, sandbox_nc_dir, built_zarr):
        with pytest.raises(FileExistsError):
            build_ddm30_adjacency(sandbox_nc_dir, built_zarr)


class TestSnapToCell:
    """Tests for snapping gauge coordinates to grid cells."""

    def test_snaps_to_nearest_center(self):
        cell_id = snap_to_cell(41.30, -76.70, SANDBOX_LAT, SANDBOX_LON)
        assert cell_id == 10  # (row 2, col 2)

    def test_exact_center(self):
        assert snap_to_cell(40.25, -77.75, SANDBOX_LAT, SANDBOX_LON) == 0


class TestBuildGaugeAdjacencies:
    """Gauge subsets through the shared engine seams."""

    @pytest.fixture(scope="class")
    def gauge_zarr(self, sandbox_nc_dir, built_zarr, tmp_path_factory):
        out = tmp_path_factory.mktemp("gauges") / "ddm30_gauges.zarr"
        # Gauge at the confluence cell (id 10); one at an unmatched ocean point.
        gauges = {"01567000": (41.25, -76.75)}
        return build_gauge_adjacencies(sandbox_nc_dir, built_zarr, gauges, out)

    def test_subset_cells(self, gauge_zarr):
        root = zarr.open_group(store=gauge_zarr, mode="r")
        order = root["01567000"]["order"][:]
        assert set(order.tolist()) == {8, 9, 13, 10}

    def test_subset_edges(self, gauge_zarr):
        """Edges within the subset: 8->9, 9->10, 13->10."""
        root = zarr.open_group(store=gauge_zarr, mode="r")
        assert len(root["01567000"]["values"][:]) == 3

    def test_gage_metadata(self, gauge_zarr, built_zarr):
        root = zarr.open_group(store=gauge_zarr, mode="r")
        attrs = root["01567000"].attrs
        global_order = zarr.open_group(store=built_zarr, mode="r")["order"][:]
        assert attrs["gage_catchment"] == 10
        assert attrs["gage_idx"] == int(np.where(global_order == 10)[0][0])


@pytest.mark.skipif(not DDM30_DIR.exists(), reason="DDM30 data not downloaded")
class TestRealDdm30:
    """Smoke test against the real ISIMIP DDM30 product (data/ddm30/)."""

    @pytest.fixture(scope="class")
    def real_zarr(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("real") / "ddm30_adjacency.zarr"
        return build_ddm30_adjacency(DDM30_DIR, out)

    def test_published_network_size(self, real_zarr):
        """67,424 CRU land cells; 55,866 cells drain to a neighbor."""
        coo, id_order = coo_from_zarr(real_zarr)
        assert len(id_order) == 67424
        assert coo.nnz == 55866

    def test_lower_triangular(self, real_zarr):
        coo, _ = coo_from_zarr(real_zarr)
        assert np.all(coo.row >= coo.col)

    def test_juniata_gauge_subset(self, real_zarr, tmp_path_factory):
        """USGS 01567000 (Newport): the Juniata is 4 cells at 0.5 degrees."""
        out = tmp_path_factory.mktemp("real_gauges") / "ddm30_gauges.zarr"
        build_gauge_adjacencies(DDM30_DIR, real_zarr, {"01567000": (40.48, -77.13)}, out)
        root = zarr.open_group(store=out, mode="r")
        assert len(root["01567000"]["order"][:]) == 4
        assert len(root["01567000"]["values"][:]) == 3
