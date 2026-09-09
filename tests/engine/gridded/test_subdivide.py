"""Tests for ddr_engine.gridded.subdivide — splitting 0.5-degree cells into ~6 km reaches."""

import numpy as np
import pytest
from ddr_engine.gridded.subdivide import (
    SUB_ID_STRIDE,
    parent_of,
    subdivide_network,
    subdivision_counts,
)

# 4-cell Juniata: 139163 -> 138443 -> 138444 -> 138445 (outlet), in topological order
ORDER = np.array([139163, 138443, 138444, 138445])
DN = np.array([1, 2, 3, -1])  # positions, -1 = terminal
LENGTH = np.array([69845.0, 42434.0, 42434.0, 42434.0])
SLOPE = np.array([0.005477, 0.00386, 0.002003, 0.001202])


class TestSubdivisionCounts:
    def test_rounds_to_nearest_target(self) -> None:
        k = subdivision_counts(np.array([42434.0, 69845.0, 6000.0]), target_m=6000.0)
        assert k.tolist() == [7, 12, 1]

    def test_never_below_one(self) -> None:
        assert subdivision_counts(np.array([100.0, 2000.0]), target_m=6000.0).tolist() == [1, 1]

    def test_sub_length_close_to_target(self) -> None:
        length = np.array([42434.0, 69845.0])
        k = subdivision_counts(length, target_m=6000.0)
        sub = length / k
        assert np.all(np.abs(sub - 6000.0) < 1000.0)


class TestSubdivideNetwork:
    def test_node_count_is_sum_of_counts(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        assert len(net.node_ids) == int(subdivision_counts(LENGTH, 6000.0).sum())

    def test_total_length_preserved_per_cell(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        for pos in range(len(ORDER)):
            assert net.length_m[net.parent_pos == pos].sum() == pytest.approx(LENGTH[pos])

    def test_slope_inherited(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        for pos in range(len(ORDER)):
            assert np.allclose(net.slope[net.parent_pos == pos], SLOPE[pos])

    def test_edges_are_strictly_lower_triangular(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        assert np.all(net.rows > net.cols)

    def test_each_node_has_at_most_one_downstream(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        _, counts = np.unique(net.cols, return_counts=True)
        assert counts.max() == 1

    def test_chain_within_cell_then_link_to_next_cell(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        k0 = int(subdivision_counts(LENGTH, 6000.0)[0])
        # inside cell 0: 0->1->...->k0-1
        for i in range(k0 - 1):
            assert (net.rows[net.cols == i][0], i) == (i + 1, i)
        # last node of cell 0 drains to the first node of cell 1
        assert net.rows[net.cols == k0 - 1][0] == k0

    def test_terminal_cell_last_node_has_no_downstream(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        assert (len(net.node_ids) - 1) not in net.cols.tolist()

    def test_edge_count(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        # every node drains onward except the single terminal
        assert len(net.rows) == len(net.node_ids) - 1

    def test_node_ids_decode_to_parent_cell(self) -> None:
        net = subdivide_network(ORDER, DN, LENGTH, SLOPE, target_m=6000.0)
        assert np.array_equal(parent_of(net.node_ids), ORDER[net.parent_pos])
        assert net.node_ids[0] == ORDER[0] * SUB_ID_STRIDE

    def test_single_reach_cells_reproduce_the_input(self) -> None:
        net = subdivide_network(ORDER, DN, np.full(4, 1000.0), SLOPE, target_m=6000.0)
        assert len(net.node_ids) == 4
        assert np.array_equal(net.rows, np.array([1, 2, 3]))
        assert np.array_equal(net.cols, np.array([0, 1, 2]))


class TestCourantMatchedCounts:
    def test_targets_celerity_times_dt(self) -> None:
        from ddr_engine.gridded.subdivide import courant_matched_counts

        # c = 1.7 m/s, dt = 3600 s -> ideal reach 6120 m; a 42,434 m cell wants 7
        k = courant_matched_counts(np.array([42434.0]), np.array([1.7]), dt_s=3600.0)
        assert k.tolist() == [7]

    def test_slow_cells_get_more_reaches(self) -> None:
        from ddr_engine.gridded.subdivide import courant_matched_counts

        k = courant_matched_counts(np.full(2, 42434.0), np.array([0.6, 2.4]), dt_s=3600.0)
        assert k[0] > k[1]

    def test_never_below_one(self) -> None:
        from ddr_engine.gridded.subdivide import courant_matched_counts

        assert courant_matched_counts(np.array([500.0]), np.array([2.0]), dt_s=3600.0).tolist() == [1]

    def test_unknown_celerity_falls_back_to_the_fixed_target(self) -> None:
        """Cells without a discharge estimate must still be split, not left at 42 km."""
        from ddr_engine.gridded.subdivide import courant_matched_counts

        k = courant_matched_counts(np.array([42434.0, 42434.0]), np.array([np.nan, 0.0]), dt_s=3600.0)
        assert k.tolist() == [7, 7]  # round(42434 / 6000)

    def test_reaches_never_shorter_than_the_floor(self) -> None:
        from ddr_engine.gridded.subdivide import MIN_REACH_M, courant_matched_counts

        # celerity 0.01 m/s would ask for 36 m reaches
        k = courant_matched_counts(np.array([42434.0]), np.array([0.01]), dt_s=3600.0)
        assert 42434.0 / k[0] >= MIN_REACH_M
