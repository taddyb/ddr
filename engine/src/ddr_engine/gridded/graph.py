"""Connectivity from D8 flow-direction rasters (DDM30 encoding).

Cell IDs are flat indices ``id = row * ncols + col`` where row 0 is the
southernmost latitude (the lat coordinate is ascending, DDM30 convention).
Longitude wraps: the grid is assumed to span the full 360 degrees, as the
ISIMIP DDM30 product does.
"""

import numpy as np

# DDM30 flow direction codes -> (dlat_idx, dlon_idx).
# lat ascends south -> north, so "north" is +1 in row index.
# 0 = sink / ocean outlet; -1 (undocumented, netCDF only) = endorheic terminal.
FLOWDIR_OFFSETS: dict[int, tuple[int, int]] = {
    1: (0, 1),  # E
    2: (-1, 1),  # SE
    3: (-1, 0),  # S
    4: (-1, -1),  # SW
    5: (0, -1),  # W
    6: (1, -1),  # NW
    7: (1, 0),  # N
    8: (1, 1),  # NE
}


def build_downstream_map(
    flowdir: np.ndarray,
) -> tuple[dict[int, int], dict[str, int]]:
    """Map each cell to its downstream neighbor from a D8 flow-direction raster.

    Parameters
    ----------
    flowdir : np.ndarray
        2D (lat, lon) raster of DDM30 direction codes; NaN marks cells
        outside the land mask.

    Returns
    -------
    tuple[dict[int, int], dict[str, int]]
        tuple[0]: {cell_id: downstream_cell_id} for every cell with a valid
        downstream neighbor.
        tuple[1]: diagnostics — counts of ``terminal_outlet`` (code 0),
        ``terminal_neg`` (code -1), ``points_off_grid`` (flows past the N/S
        edge), and ``points_to_invalid`` (flows into a NaN cell). Cells in
        the last two categories get no downstream edge.
    """
    nrows, ncols = flowdir.shape
    valid = ~np.isnan(flowdir)

    downstream: dict[int, int] = {}
    diag = {"terminal_outlet": 0, "terminal_neg": 0, "points_to_invalid": 0, "points_off_grid": 0}

    rows, cols = np.where(valid)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        code = int(flowdir[r, c])
        if code == 0:
            diag["terminal_outlet"] += 1
            continue
        if code == -1:
            diag["terminal_neg"] += 1
            continue
        dr, dc = FLOWDIR_OFFSETS[code]
        r2 = r + dr
        c2 = (c + dc) % ncols  # longitude wraps
        if not (0 <= r2 < nrows):
            diag["points_off_grid"] += 1
            continue
        if not valid[r2, c2]:
            diag["points_to_invalid"] += 1
            continue
        downstream[r * ncols + c] = r2 * ncols + c2

    return downstream, diag


def build_upstream_dict(downstream: dict[int, int]) -> dict[int, list[int]]:
    """Invert a downstream map into {downstream_id: sorted upstream_ids}.

    The result has the same shape as ``merit.graph.build_upstream_dict`` output
    and plugs directly into ``merit.graph.build_graph``.
    """
    upstream: dict[int, list[int]] = {}
    for up_id, dn_id in downstream.items():
        upstream.setdefault(dn_id, []).append(up_id)
    for dn_id in upstream:
        upstream[dn_id].sort()
    return upstream
