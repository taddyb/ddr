"""Tests for gridded/graph.py - D8 flow-direction raster to connectivity."""

import numpy as np
from ddr_engine.gridded import FLOWDIR_OFFSETS, build_downstream_map

from .conftest import EXPECTED_EDGES, NCOLS


class TestFlowdirOffsets:
    """Tests for the DDM30 direction-code table."""

    def test_eight_directions(self):
        assert set(FLOWDIR_OFFSETS.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_east_is_positive_lon(self):
        assert FLOWDIR_OFFSETS[1] == (0, 1)

    def test_north_is_positive_lat_index(self):
        """lat coordinate ascends south->north, so N = +1 in row index."""
        assert FLOWDIR_OFFSETS[7] == (1, 0)

    def test_south_east_diagonal(self):
        assert FLOWDIR_OFFSETS[2] == (-1, 1)


class TestBuildDownstreamMap:
    """Tests for build_downstream_map on the sandbox grid."""

    def test_edges_match_sandbox(self, sandbox_downstream):
        assert set(sandbox_downstream.items()) == EXPECTED_EDGES

    def test_longitude_wraps(self, sandbox_downstream):
        """Cell 3 (easternmost column) flows E across the lon seam into cell 0."""
        assert sandbox_downstream[3] == 0

    def test_terminals_have_no_downstream(self, sandbox_downstream):
        for cell_id in [0, 6, 15]:
            assert cell_id not in sandbox_downstream

    def test_off_grid_flow_dropped(self, sandbox_downstream):
        """Cell 12 flows north off the grid; it gets no downstream edge."""
        assert 12 not in sandbox_downstream

    def test_flow_into_nan_dropped(self, sandbox_downstream):
        """Cell 5 flows into a NaN cell; it gets no downstream edge."""
        assert 5 not in sandbox_downstream

    def test_diagnostics(self, sandbox_flowdir):
        _, diag = build_downstream_map(sandbox_flowdir)
        assert diag["terminal_outlet"] == 2  # ids 6, 15
        assert diag["terminal_neg"] == 1  # id 0
        assert diag["points_off_grid"] == 1  # id 12
        assert diag["points_to_invalid"] == 1  # id 5

    def test_all_nan_raster_is_empty(self):
        downstream, diag = build_downstream_map(np.full((3, 3), np.nan, dtype=np.float32))
        assert downstream == {}
        assert all(v == 0 for v in diag.values())


class TestBuildUpstreamDict:
    """Tests for build_upstream_dict (inversion of the downstream map)."""

    def test_correct_keys(self, sandbox_upstream):
        """Only cells receiving flow are keys."""
        assert set(sandbox_upstream.keys()) == {0, 6, 9, 10}

    def test_confluence_cell(self, sandbox_upstream):
        """Cell 10 receives the main stem (9) and the SE tributary (13)."""
        assert sandbox_upstream[10] == [9, 13]

    def test_upstream_lists_sorted(self, sandbox_upstream):
        for ups in sandbox_upstream.values():
            assert ups == sorted(ups)

    def test_headwaters_not_keys(self, sandbox_upstream):
        for cell_id in [3, 5, 8, 12, 13, 15]:
            assert cell_id not in sandbox_upstream

    def test_feeds_merit_build_graph(self, sandbox_upstream):
        """The gridded upstream dict must plug into the existing graph builder."""
        from ddr_engine.merit.graph import build_graph

        graph, node_indices = build_graph(sandbox_upstream)
        assert graph.num_nodes() == 7  # cells participating in edges
        assert graph.num_edges() == 5
        assert set(node_indices.keys()) == {0, 3, 6, 8, 9, 10, 13}


class TestCellIdConvention:
    """The flat-index scheme must be stable: id = row * ncols + col."""

    def test_id_roundtrip(self):
        cell_id = 2 * NCOLS + 1
        assert (cell_id // NCOLS, cell_id % NCOLS) == (2, 1)
