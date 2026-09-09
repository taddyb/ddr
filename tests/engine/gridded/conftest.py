"""Fixtures for gridded (D8 raster) engine tests.

Sandbox Grid Reference
======================
A 4x4 grid of 0.5-degree cells. Cell IDs are flat indices ``id = row * ncols + col``
where row 0 is the SOUTHERNMOST latitude (lat coordinate ascending, DDM30 convention).
Rendered here with north at the top (row 3 first):

    row 3 |  ↑12   ↘13    ·    ◎15
    row 2 |  →8    →9    ↓10    ·
    row 1 |   ·    ↓5    ◎6     ·
    row 0 |  ×0     ·     ·    →3   (3 flows east, wrapping to 0)

    ·  = NaN (ocean / outside mask)
    ◎  = flow direction 0 (sink / ocean outlet)
    ×  = flow direction -1 (endorheic terminal, undocumented DDM30 netCDF code)

Edges (upstream -> downstream):
    8 -> 9 -> 10 -> 6      (main stem, E, E, then S into the outlet)
    13 -> 10               (SE diagonal tributary)
    3 -> 0                 (eastward flow wrapping across the lon seam)

Pathological cells (become no-downstream diagnostics, not edges):
    12 flows N off the grid          (points_off_grid)
    5  flows S into a NaN cell       (points_to_invalid)
    15 is a sink with no upstreams   (isolated)

Expected totals: 10 valid cells, 5 edges, terminals {0, 6}, isolated {5, 12, 15}.
"""

import numpy as np
import pytest

pytest.importorskip("ddr_engine")

NROWS, NCOLS = 4, 4

# Cell centers (0.5-degree spacing, lat ascending)
SANDBOX_LAT = np.array([40.25, 40.75, 41.25, 41.75], dtype=np.float32)
SANDBOX_LON = np.array([-77.75, -77.25, -76.75, -76.25], dtype=np.float32)

EXPECTED_EDGES = {(8, 9), (9, 10), (10, 6), (13, 10), (3, 0)}
EXPECTED_VALID = {0, 3, 5, 6, 8, 9, 10, 12, 13, 15}
EXPECTED_ISOLATED = {5, 12, 15}


@pytest.fixture(scope="session")
def sandbox_flowdir() -> np.ndarray:
    """Flow direction raster for the sandbox grid (see module docstring)."""
    fd = np.full((NROWS, NCOLS), np.nan, dtype=np.float32)
    fd[0, 0] = -1  # id 0: endorheic terminal
    fd[0, 3] = 1  # id 3: E, wraps to id 0
    fd[1, 1] = 3  # id 5: S into NaN (points_to_invalid)
    fd[1, 2] = 0  # id 6: outlet
    fd[2, 0] = 1  # id 8: E -> 9
    fd[2, 1] = 1  # id 9: E -> 10
    fd[2, 2] = 3  # id 10: S -> 6
    fd[3, 0] = 7  # id 12: N off the grid (points_off_grid)
    fd[3, 1] = 2  # id 13: SE -> 10
    fd[3, 3] = 0  # id 15: isolated sink
    return fd


@pytest.fixture(scope="session")
def sandbox_basins() -> np.ndarray:
    """Basin numbers: main stem = 100, wrap pair = 200, singletons 300/400/500."""
    bs = np.full((NROWS, NCOLS), np.nan, dtype=np.float32)
    for cell_id in [6, 8, 9, 10, 13]:
        bs[cell_id // NCOLS, cell_id % NCOLS] = 100
    for cell_id in [0, 3]:
        bs[cell_id // NCOLS, cell_id % NCOLS] = 200
    bs[1, 1] = 300  # id 5
    bs[3, 0] = 400  # id 12
    bs[3, 3] = 500  # id 15
    return bs


@pytest.fixture(scope="session")
def sandbox_slopes(sandbox_flowdir) -> np.ndarray:
    """Slopes: 0.001 everywhere a cell is valid."""
    sl = np.full((NROWS, NCOLS), np.nan, dtype=np.float32)
    sl[~np.isnan(sandbox_flowdir)] = 0.001
    return sl


@pytest.fixture(scope="session")
def sandbox_nc_dir(tmp_path_factory, sandbox_flowdir, sandbox_basins, sandbox_slopes):
    """Sandbox rasters written as DDM30-named netCDF files."""
    import xarray as xr

    d = tmp_path_factory.mktemp("ddm30_sandbox")
    for filename, var, values in [
        ("ddm30_flowdir_cru_neva.nc", "flowdirection", sandbox_flowdir),
        ("ddm30_basins_cru_neva.nc", "basinnumber", sandbox_basins),
        ("ddm30_slopes_cru_neva.nc", "slope", sandbox_slopes),
    ]:
        ds = xr.Dataset(
            {var: (("lat", "lon"), values)},
            coords={"lat": SANDBOX_LAT, "lon": SANDBOX_LON},
        )
        ds.to_netcdf(d / filename)
    return d


@pytest.fixture(scope="session")
def sandbox_downstream(sandbox_flowdir):
    from ddr_engine.gridded import build_downstream_map

    downstream, _ = build_downstream_map(sandbox_flowdir)
    return downstream


@pytest.fixture(scope="session")
def sandbox_upstream(sandbox_downstream):
    from ddr_engine.gridded import build_upstream_dict

    return build_upstream_dict(sandbox_downstream)
