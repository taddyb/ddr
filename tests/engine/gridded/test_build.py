"""Tests for gridded/build.py - adjacency assembly and cell lengths."""

import numpy as np
import pytest
from ddr_engine.gridded import compute_cell_lengths, create_grid_adjacency

from .conftest import EXPECTED_ISOLATED, EXPECTED_VALID, NCOLS, SANDBOX_LAT, SANDBOX_LON


@pytest.fixture(scope="module")
def adjacency(sandbox_upstream):
    all_cell_ids = np.array(sorted(EXPECTED_VALID))
    return create_grid_adjacency(sandbox_upstream, all_cell_ids)


class TestCreateGridAdjacency:
    """Tests for create_grid_adjacency on the sandbox grid."""

    def test_matrix_covers_all_valid_cells(self, adjacency):
        matrix, id_order = adjacency
        assert matrix.shape == (10, 10)
        assert sorted(id_order) == sorted(EXPECTED_VALID)

    def test_edge_count(self, adjacency):
        matrix, _ = adjacency
        assert matrix.nnz == 5

    def test_lower_triangular(self, adjacency):
        matrix, _ = adjacency
        assert np.all(matrix.row >= matrix.col)

    def test_topological_order(self, adjacency):
        """Every upstream cell must appear before its downstream cell."""
        matrix, id_order = adjacency
        idx = {cid: i for i, cid in enumerate(id_order)}
        assert idx[8] < idx[9] < idx[10] < idx[6]
        assert idx[13] < idx[10]
        assert idx[3] < idx[0]

    def test_isolated_cells_appended(self, adjacency):
        """Cells with no edges sit at the end of the order with empty rows/cols."""
        matrix, id_order = adjacency
        assert set(id_order[-len(EXPECTED_ISOLATED) :]) == EXPECTED_ISOLATED
        connected = set(matrix.row.tolist()) | set(matrix.col.tolist())
        for cell_id in EXPECTED_ISOLATED:
            assert id_order.index(cell_id) not in connected

    def test_values_are_ones(self, adjacency):
        matrix, _ = adjacency
        assert matrix.data.dtype == np.uint8
        assert (matrix.data == 1).all()


class TestComputeCellLengths:
    """Tests for compute_cell_lengths (haversine, centre to centre)."""

    @pytest.fixture(scope="class")
    def lengths(self, sandbox_downstream, adjacency):
        _, id_order = adjacency
        return id_order, compute_cell_lengths(id_order, sandbox_downstream, SANDBOX_LAT, SANDBOX_LON, NCOLS)

    def test_east_west_step(self, lengths):
        """Cell 8 -> 9 is a 0.5-deg E step at 41.25N: ~55.6 km * cos(lat)."""
        id_order, length_m = lengths
        expected = 55_597.0 * np.cos(np.radians(41.25))
        assert length_m[id_order.index(8)] == pytest.approx(expected, rel=0.01)

    def test_north_south_step(self, lengths):
        """Cell 10 -> 6 is a 0.5-deg S step: ~55.6 km regardless of latitude."""
        id_order, length_m = lengths
        assert length_m[id_order.index(10)] == pytest.approx(55_597.0, rel=0.01)

    def test_diagonal_step(self, lengths):
        """Cell 13 -> 10 is a SE step: sqrt(NS^2 + EW^2) at the mean latitude."""
        id_order, length_m = lengths
        ns = 55_597.0
        ew = 55_597.0 * np.cos(np.radians(41.5))
        assert length_m[id_order.index(13)] == pytest.approx(np.hypot(ns, ew), rel=0.01)

    def test_terminal_gets_own_cell_scale(self, lengths):
        """Cells with no downstream get the 0.5-deg meridional length."""
        id_order, length_m = lengths
        for cell_id in [0, 6, 15]:
            assert length_m[id_order.index(cell_id)] == pytest.approx(55_597.0, rel=0.01)

    def test_all_finite_and_positive(self, lengths):
        _, length_m = lengths
        assert np.isfinite(length_m).all()
        assert (length_m > 0).all()
        assert length_m.dtype == np.float32
