"""Tests for ddr_engine.gridded.validate — network, attribute and unit checks."""

import numpy as np
import pandas as pd
from ddr_engine.gridded.validate import (
    UNITS,
    check_attribute_consistency,
    check_network,
    check_units,
)


def _net(n: int = 6):
    """Chain-ish network: 0->2, 1->2, 2->4, 3->4, 4 terminal, 5 isolated (topological order)."""
    rows = np.array([2, 2, 4, 4])  # downstream (to)
    cols = np.array([0, 1, 2, 3])  # upstream (from)
    lat = np.full(n, 40.25)
    lon = np.array([-77.25, -76.75, -77.25, -76.75, -77.25, -100.25])
    basin = np.array([1, 1, 1, 1, 1, 2])
    return rows, cols, lat, lon, basin


class TestCheckNetwork:
    def test_clean_network_passes(self) -> None:
        rows, cols, lat, lon, basin = _net()
        results = check_network(rows, cols, lat, lon, basin, ncols=720)
        assert all(r.passed for r in results), [r for r in results if not r.passed]

    def test_two_downstreams_fails(self) -> None:
        rows, cols, lat, lon, basin = _net()
        rows = np.append(rows, 5)
        cols = np.append(cols, 0)  # cell 0 now drains to two cells
        results = check_network(rows, cols, lat, lon, basin, ncols=720)
        assert not next(r for r in results if r.name == "single downstream").passed

    def test_upper_triangular_edge_fails(self) -> None:
        rows, cols, lat, lon, basin = _net()
        rows[0] = 0
        cols[0] = 3  # edge from 3 to 0: not lower-triangular
        results = check_network(rows, cols, lat, lon, basin, ncols=720)
        assert not next(r for r in results if r.name == "lower triangular").passed

    def test_self_loop_fails(self) -> None:
        rows, cols, lat, lon, basin = _net()
        rows[0], cols[0] = 3, 3
        results = check_network(rows, cols, lat, lon, basin, ncols=720)
        assert not next(r for r in results if r.name == "no self loops").passed

    def test_cross_basin_edge_fails(self) -> None:
        rows, cols, lat, lon, basin = _net()
        basin[3] = 9
        results = check_network(rows, cols, lat, lon, basin, ncols=720)
        assert not next(r for r in results if r.name == "edges within basin").passed


class TestCheckUnits:
    def test_bounds_spec_covers_expected_variables(self) -> None:
        assert {"meanslope", "meanP", "snow_fraction", "catchsize", "log10_uparea"} <= set(UNITS)
        for spec in UNITS.values():
            assert spec.lo < spec.hi and spec.unit

    def test_in_range_passes(self) -> None:
        df = pd.DataFrame({"meanslope": [0.0, 5.0, 30.0], "NDVI": [0.1, 0.5, 0.8]})
        assert all(r.passed for r in check_units(df))

    def test_out_of_range_fails_and_names_variable(self) -> None:
        df = pd.DataFrame({"meanslope": [0.0, 95.0], "NDVI": [0.1, 0.5]})
        bad = [r for r in check_units(df) if not r.passed]
        assert len(bad) == 1 and "meanslope" in bad[0].name

    def test_unknown_column_is_reported(self) -> None:
        df = pd.DataFrame({"mystery_var": [1.0]})
        assert any("mystery_var" in r.name and not r.passed for r in check_units(df))


class TestAttributeConsistency:
    def _good(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SoilGrids1km_clay": [20.0, 30.0],
                "SoilGrids1km_sand": [40.0, 30.0],
                "SoilGrids1km_silt": [40.0, 40.0],
                "WCpF2": [0.30, 0.35],
                "WCsat": [0.45, 0.50],
                "meanP": [1000.0, 500.0],
                "ETPOT_Hargr": [1200.0, 1500.0],
                "aridity": [1.2, 3.0],
                "catchsize": [2000.0, 2000.0],
                "log10_uparea": [3.4, 4.0],
            }
        )

    def test_consistent_frame_passes(self) -> None:
        assert all(r.passed for r in check_attribute_consistency(self._good()))

    def test_texture_not_summing_to_100_fails(self) -> None:
        df = self._good()
        df.loc[0, "SoilGrids1km_clay"] = 5.0
        assert not next(r for r in check_attribute_consistency(df) if r.name == "texture sums to 100%").passed

    def test_field_capacity_above_saturation_fails(self) -> None:
        df = self._good()
        df.loc[1, "WCpF2"] = 0.9
        assert not next(r for r in check_attribute_consistency(df) if r.name == "WCpF2 <= WCsat").passed

    def test_aridity_identity_violation_fails(self) -> None:
        df = self._good()
        df.loc[0, "aridity"] = 99.0
        assert not next(r for r in check_attribute_consistency(df) if r.name == "aridity = PET/P").passed

    def test_uparea_below_own_cell_area_fails(self) -> None:
        df = self._good()
        df.loc[0, "log10_uparea"] = 1.0
        assert not next(
            r for r in check_attribute_consistency(df) if r.name == "log10_uparea >= cell area"
        ).passed
