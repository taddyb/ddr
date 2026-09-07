"""Build a sub-reach adjacency zarr from a DDM30 cell adjacency.

Each 0.5-degree cell becomes a chain of ~6 km reaches so Muskingum-Cunge runs near
Courant 1 at the model's hardcoded dt = 3600 s (see ddr_engine.gridded.subdivide).
The output has the same schema as the cell adjacency plus a ``parent_cell`` array
mapping every node to its DDM30 cell, so per-cell attributes and Q' expand by
lookup rather than needing their own subdivided stores.

Usage:
    uv run python scripts/build_subdivided_adjacency.py [--target-m 6000] [--bbox ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr
from ddr_engine.gridded.subdivide import (
    courant_matched_counts,
    parent_of,
    subdivide_network,
    subdivision_counts,
)

ADJACENCY = Path("data/ddm30/ddm30_adjacency.zarr")
OUT = Path("data/ddm30/ddm30_subreach_adjacency.zarr")
CONUS_BBOX = (-125.0, 24.0, -66.0, 53.0)

log = logging.getLogger("build_subdivided_adjacency")


def _celerity(attributes: Path, cells: np.ndarray, length_m: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Kinematic celerity per cell at long-term mean flow, using ddr's own geometry."""
    import xarray as xr

    sys.path.insert(0, str(Path(__file__).parent))
    from validate_gridded import coefficients, mean_discharge

    df = xr.open_dataset(attributes).to_dataframe()
    q_by_cell = dict(
        zip(
            df.index.to_numpy(),
            mean_discharge(df["meanP"].to_numpy(), 10 ** df["log10_uparea"].to_numpy()),
            strict=False,
        )
    )
    q = np.array([q_by_cell.get(int(c), np.nan) for c in cells])
    ok = np.isfinite(q) & (length_m > 0) & (slope > 0)
    out = np.full(len(cells), np.nan)
    *_, courant, _ = coefficients(length_m[ok], slope[ok], q[ok])
    out[ok] = courant * length_m[ok] / 3600.0
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", type=Path, default=ADJACENCY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--target-m", type=float, default=6000.0)
    parser.add_argument("--bbox", nargs=4, type=float, default=list(CONUS_BBOX))
    parser.add_argument("--global", dest="global_mode", action="store_true")
    parser.add_argument(
        "--courant-matched",
        type=Path,
        default=None,
        metavar="ATTRIBUTES_NC",
        help="size each cell's reaches from its own celerity at mean flow (needs meanP/log10_uparea)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    g = zarr.open_group(args.adjacency, mode="r")
    order, lat, lon = g["order"][:], g["lat"][:], g["lon"][:]
    length_m, slope, basin = g["length_m"][:], g["slope"][:], g["basin"][:]
    dn_global = np.full(len(order), -1, dtype=np.int64)
    dn_global[g["indices_1"][:]] = g["indices_0"][:]

    if args.global_mode:
        keep = np.ones(len(order), dtype=bool)
    else:
        xmin, ymin, xmax, ymax = args.bbox
        keep = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    # a subset must keep its internal edges only; cells draining outside become terminals
    pos_map = np.full(len(order), -1, dtype=np.int64)
    pos_map[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    dn = np.where(dn_global >= 0, pos_map[np.clip(dn_global, 0, None)], -1)[keep]

    if args.courant_matched:
        counts = courant_matched_counts(
            length_m[keep], _celerity(args.courant_matched, order[keep], length_m[keep], slope[keep])
        )
    else:
        counts = subdivision_counts(length_m[keep], args.target_m)
    net = subdivide_network(order[keep], dn, length_m[keep], slope[keep], counts=counts)
    log.info(
        "%d cells -> %d sub-reaches (%.1f per cell, %.0f-%.0f m); %d edges",
        int(keep.sum()),
        len(net.node_ids),
        len(net.node_ids) / int(keep.sum()),
        net.length_m.min(),
        net.length_m.max(),
        len(net.rows),
    )
    assert np.all(net.rows > net.cols), "adjacency must be strictly lower-triangular"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(args.out, mode="w")
    root["order"] = net.node_ids.astype(np.int64)
    root["parent_cell"] = parent_of(net.node_ids).astype(np.int32)
    root["indices_0"] = net.rows.astype(np.int32)
    root["indices_1"] = net.cols.astype(np.int32)
    root["values"] = np.ones(len(net.rows), dtype=np.uint8)
    root["length_m"] = net.length_m.astype(np.float32)
    root["slope"] = net.slope.astype(np.float32)
    root["lat"] = lat[keep][net.parent_pos].astype(np.float32)
    root["lon"] = lon[keep][net.parent_pos].astype(np.float32)
    root["basin"] = basin[keep][net.parent_pos].astype(np.int32)
    n = len(net.node_ids)
    root.attrs.update(
        {
            "format": "COO",
            "shape": [n, n],
            "geodataset": "ddm30",
            "subdivided": True,
            "target_reach_m": args.target_m,
            "cell_id_scheme": "node id = cell id * 100 + sub index (upstream-most first)",
            "data_types": {"indices_0": "int32", "indices_1": "int32", "values": "uint8"},
        }
    )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
