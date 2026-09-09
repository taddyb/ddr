"""Build functions for adjacency matrices from D8 flow-direction rasters."""

from pathlib import Path

import numpy as np
import rustworkx as rx
import xarray as xr
import zarr
from numpy.typing import NDArray
from scipy import sparse

from ddr_engine.core.zarr_io import coo_to_zarr, coo_to_zarr_group
from ddr_engine.merit.graph import build_graph, subset_upstream
from ddr_engine.merit.io import create_subset_coo

from .graph import build_downstream_map, build_upstream_dict

EARTH_RADIUS_M = 6_371_000.0

# Standard ISIMIP DDM30 filenames and their variable names
DDM30_FILES = {
    "flowdir": ("ddm30_flowdir_cru_neva.nc", "flowdirection"),
    "basins": ("ddm30_basins_cru_neva.nc", "basinnumber"),
    "slopes": ("ddm30_slopes_cru_neva.nc", "slope"),
}


def haversine_m(lat1: NDArray, lon1: NDArray, lat2: NDArray, lon2: NDArray) -> NDArray:
    """Great-circle distance in metres (vectorized)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def create_grid_adjacency(
    upstream: dict[int, list[int]],
    all_cell_ids: NDArray[np.int64],
) -> tuple[sparse.coo_matrix, list[int]]:
    """
    Create a lower triangular adjacency matrix from gridded connectivity.

    Parameters
    ----------
    upstream : dict[int, list[int]]
        {downstream_cell_id: upstream_cell_ids} from ``build_upstream_dict``.
    all_cell_ids : NDArray
        Flat IDs of every valid (land-mask) cell; cells with no edges are
        appended to the order as isolated nodes.

    Returns
    -------
    tuple[sparse.coo_matrix, list[int]]
        tuple[0]: lower triangular COO adjacency matrix
        tuple[1]: topological ordering of cell IDs (isolated cells last)
    """
    graph, _ = build_graph(upstream)
    ts_order = rx.topological_sort(graph)  # D8 grids cannot cycle
    id_order = [graph.get_node_data(i) for i in ts_order]

    isolated = sorted(set(all_cell_ids.tolist()) - set(id_order))
    id_order = id_order + isolated

    idx_map = {cid: i for i, cid in enumerate(id_order)}
    row, col = [], []
    for node in ts_order:
        if graph.out_degree(node) == 0:
            continue
        successors = graph.successors(node)
        assert len(successors) == 1, f"Cell {graph.get_node_data(node)} has multiple successors"
        row.append(idx_map[successors[0]])
        col.append(idx_map[graph.get_node_data(node)])

    matrix = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)),
        shape=(len(id_order), len(id_order)),
        dtype=np.uint8,
    )
    assert np.all(matrix.row >= matrix.col), "Matrix is not lower triangular"
    return matrix, id_order


def compute_cell_lengths(
    id_order: list[int],
    downstream: dict[int, int],
    lat: NDArray,
    lon: NDArray,
    ncols: int,
) -> NDArray[np.float32]:
    """
    Flow length per cell, aligned to ``id_order``.

    Cells with a downstream neighbor get the great-circle distance between
    cell centres; terminal and isolated cells get the 0.5-degree meridional
    length as their own-cell scale.
    """
    order = np.array(id_order)
    cell_lat = lat[order // ncols].astype(np.float64)
    cell_lon = lon[order % ncols].astype(np.float64)

    idx_map = {cid: i for i, cid in enumerate(id_order)}
    dn_pos = np.full(len(id_order), -1, dtype=np.int64)
    for up_id, dn_id in downstream.items():
        dn_pos[idx_map[up_id]] = idx_map[dn_id]

    length_m = np.empty(len(id_order), dtype=np.float32)
    has_dn = dn_pos >= 0
    length_m[has_dn] = haversine_m(
        cell_lat[has_dn], cell_lon[has_dn], cell_lat[dn_pos[has_dn]], cell_lon[dn_pos[has_dn]]
    )
    length_m[~has_dn] = haversine_m(
        cell_lat[~has_dn] - 0.25, cell_lon[~has_dn], cell_lat[~has_dn] + 0.25, cell_lon[~has_dn]
    )
    return length_m


def _load_ddm30(nc_dir: Path) -> tuple[dict[str, np.ndarray], NDArray, NDArray]:
    """Load the three DDM30 rasters and grid coordinates from a directory."""
    rasters = {}
    for key, (filename, var) in DDM30_FILES.items():
        ds = xr.open_dataset(nc_dir / filename)
        rasters[key] = ds[var].values
        lat, lon = ds["lat"].values, ds["lon"].values
    return rasters, lat, lon


def snap_to_cell(lat_pt: float, lon_pt: float, lat: NDArray, lon: NDArray) -> int:
    """Snap a point to the flat index of the nearest cell centre."""
    r = int(np.abs(lat - lat_pt).argmin())
    c = int(np.abs(lon - lon_pt).argmin())
    return r * len(lon) + c


def build_ddm30_adjacency(nc_dir: Path, out_path: Path) -> Path:
    """
    Build the DDM30 adjacency matrix and per-cell attributes, saved to zarr.

    Parameters
    ----------
    nc_dir : Path
        Directory holding the three ISIMIP DDM30 netCDF files
        (``ddm30_{flowdir,basins,slopes}_cru_neva.nc``).
    out_path : Path
        Path to save the zarr group.

    Returns
    -------
    Path
        Path to the created zarr store, containing the COO adjacency plus
        ``length_m``, ``slope``, ``lat``, ``lon``, and ``basin`` arrays
        aligned to ``order``, and a ``grid_shape`` attribute.

    Raises
    ------
    FileExistsError
        If a zarr store already exists at out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"Cannot create zarr store {out_path}. One already exists")

    rasters, lat, lon = _load_ddm30(Path(nc_dir))
    flowdir = rasters["flowdir"]
    ncols = flowdir.shape[1]

    valid_rows, valid_cols = np.where(~np.isnan(flowdir))
    all_cell_ids = valid_rows * ncols + valid_cols
    print(f"Creating adjacency matrix for {len(all_cell_ids)} grid cells")

    downstream, diag = build_downstream_map(flowdir)
    print(f"Connectivity diagnostics: {diag}")

    matrix, id_order = create_grid_adjacency(build_upstream_dict(downstream), all_cell_ids)
    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    coo_to_zarr(matrix, id_order, out_path, "ddm30")

    order = np.array(id_order)
    r_idx, c_idx = order // ncols, order % ncols
    length_m = compute_cell_lengths(id_order, downstream, lat, lon, ncols)
    attributes = {
        "length_m": length_m,
        "slope": rasters["slopes"][r_idx, c_idx].astype(np.float32),
        "lat": lat[r_idx].astype(np.float32),
        "lon": lon[c_idx].astype(np.float32),
        "basin": rasters["basins"][r_idx, c_idx].astype(np.int32),
    }

    root = zarr.open_group(store=out_path, mode="r+")
    for name, arr in attributes.items():
        za = root.create_array(name=name, shape=arr.shape, dtype=arr.dtype)
        za[:] = arr
    root.attrs["grid_shape"] = [int(flowdir.shape[0]), int(flowdir.shape[1])]
    root.attrs["cell_id_scheme"] = "row * ncols + col (row 0 = southernmost lat)"

    print(f"DDM30 adjacency and attributes written to zarr at {out_path}")
    return out_path


def build_gauge_adjacencies(
    nc_dir: Path,
    adjacency_zarr_path: Path,
    gauges: dict[str, tuple[float, float]],
    out_path: Path,
) -> Path:
    """
    Build per-gauge upstream-subset adjacency matrices on the DDM30 grid.

    Parameters
    ----------
    nc_dir : Path
        Directory holding the three ISIMIP DDM30 netCDF files.
    adjacency_zarr_path : Path
        Path to the full DDM30 adjacency zarr store (for the global order).
    gauges : dict[str, tuple[float, float]]
        Mapping of gauge STAID to (lat, lon); each gauge is snapped to the
        nearest grid cell.
    out_path : Path
        Path to save the gauge zarr group.

    Returns
    -------
    Path
        Path to the created zarr store (one group per STAID, matching the
        MERIT gauge-zarr layout).

    Raises
    ------
    FileExistsError
        If a zarr store already exists at out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"Cannot create zarr store {out_path}. One already exists")

    rasters, lat, lon = _load_ddm30(Path(nc_dir))
    flowdir = rasters["flowdir"]

    downstream, _ = build_downstream_map(flowdir)
    graph, node_indices = build_graph(build_upstream_dict(downstream))

    global_order = zarr.open_group(store=adjacency_zarr_path, mode="r")["order"][:]
    mapping = {int(cid): idx for idx, cid in enumerate(global_order)}

    store = zarr.storage.LocalStore(root=out_path)
    root = zarr.create_group(store=store)

    for staid, (gauge_lat, gauge_lon) in gauges.items():
        origin = snap_to_cell(gauge_lat, gauge_lon, lat, lon)
        if origin not in mapping:
            print(f"Gauge {staid} snapped to cell {origin}, not a valid land cell. Skipping.")
            continue
        gauge_root = root.create_group(staid)
        subset_ids = subset_upstream(origin, graph, node_indices)
        coo, subset_list = create_subset_coo(subset_ids, mapping, graph, node_indices)
        coo_to_zarr_group(coo, subset_list, origin, gauge_root, mapping, "ddm30")

    print(f"DDM30 gauge adjacency matrices written to {out_path}")
    return out_path
