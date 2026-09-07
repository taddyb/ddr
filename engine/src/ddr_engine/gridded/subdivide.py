"""Split DDM30 cells into sub-reaches so Muskingum-Cunge runs near Courant 1.

A 0.5-degree cell is 33-79 km long, giving K = L/c of 5-13 h against the model's
hardcoded dt = 3600 s. The Cunge weighting X then saturates at its 0.5 cap and c1
goes negative for ~88% of cell-flow states. Splitting each cell into reaches of
about ``TARGET_M`` puts the Courant number near 1, where every Muskingum
coefficient is non-negative, without touching dt or any shared routing code.

Shorter is not better: below ~4 km the reaches become too short for a one-hour
step, the Courant number climbs past 1.5 and c3 goes negative instead.

The sub-reaches of a cell form a chain, and the last one drains to the first
sub-reach of the downstream cell, so the network stays dendritic and — because
the parent order is topological — strictly lower-triangular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_M = 6000.0  # Courant ~1 at dt=3600 s for celerity ~1.7 m/s
SUB_ID_STRIDE = 100  # node id = parent cell id * STRIDE + sub index (max k is ~13)


@dataclass(frozen=True)
class SubdividedNetwork:
    """A subdivided routing network, topologically ordered."""

    node_ids: np.ndarray  # parent cell id * SUB_ID_STRIDE + sub index
    parent_pos: np.ndarray  # index into the parent cell arrays
    rows: np.ndarray  # downstream node index of each edge
    cols: np.ndarray  # upstream node index of each edge
    length_m: np.ndarray
    slope: np.ndarray


def subdivision_counts(length_m: np.ndarray, target_m: float = TARGET_M) -> np.ndarray:
    """Number of sub-reaches per cell: ``round(L / target)``, at least one."""
    return np.maximum(1, np.rint(np.asarray(length_m, dtype=float) / target_m)).astype(np.int64)


def parent_of(node_ids: np.ndarray) -> np.ndarray:
    """Parent DDM30 cell id of each sub-reach node."""
    return np.asarray(node_ids) // SUB_ID_STRIDE


def subdivide_network(
    order: np.ndarray,
    dn: np.ndarray,
    length_m: np.ndarray,
    slope: np.ndarray,
    target_m: float = TARGET_M,
    counts: np.ndarray | None = None,
) -> SubdividedNetwork:
    """Expand each cell into a chain of sub-reaches.

    ``dn`` holds the downstream *position* of each cell (-1 for terminals). Reach
    counts come from ``counts`` when given, else from ``target_m``. Each sub-reach
    inherits its cell's slope; the cell's length is divided exactly, so total
    channel length is preserved.
    """
    order = np.asarray(order)
    dn = np.asarray(dn)
    k = subdivision_counts(length_m, target_m) if counts is None else np.asarray(counts, dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(k)[:-1]])  # first node index of each cell

    parent_pos = np.repeat(np.arange(len(order)), k)
    sub_idx = np.concatenate([np.arange(n) for n in k])
    node_ids = order[parent_pos] * SUB_ID_STRIDE + sub_idx
    lengths = (np.asarray(length_m, dtype=float) / k)[parent_pos]
    slopes = np.asarray(slope, dtype=float)[parent_pos]

    rows, cols = [], []
    for pos in range(len(order)):
        first, n = int(start[pos]), int(k[pos])
        for s in range(n - 1):  # chain inside the cell
            cols.append(first + s)
            rows.append(first + s + 1)
        if dn[pos] >= 0:  # last sub-reach drains to the next cell's first
            cols.append(first + n - 1)
            rows.append(int(start[dn[pos]]))
    return SubdividedNetwork(
        node_ids=node_ids,
        parent_pos=parent_pos,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        length_m=lengths,
        slope=slopes,
    )


def courant_matched_counts(
    length_m: np.ndarray, celerity_m_s: np.ndarray, dt_s: float = 3600.0
) -> np.ndarray:
    """Sub-reach count that puts each cell's Courant number nearest 1.

    A fixed target length only hits Courant 1 where celerity happens to match
    ``target/dt``; celerity spans roughly 0.6-2.4 m/s across CONUS, so matching per
    cell (ideal reach = c*dt) roughly triples the share of reaches with every
    Muskingum coefficient non-negative, at about double the node count. Cells with
    no celerity estimate are left undivided.
    """
    length = np.asarray(length_m, dtype=float)
    c = np.asarray(celerity_m_s, dtype=float)
    ideal = np.where(np.isfinite(c) & (c > 0), c * dt_s, np.inf)
    return np.maximum(1, np.rint(length / ideal)).astype(np.int64)
