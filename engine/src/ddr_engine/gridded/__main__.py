"""CLI entrypoint for building DDM30 gridded adjacency matrices.

Usage:
    python -m ddr_engine.gridded <nc_dir> [--path PATH] [--gages GAGES]
"""

import argparse
import csv
from pathlib import Path

from . import build_ddm30_adjacency, build_gauge_adjacencies


def main(argv: list[str] | None = None) -> None:
    """The main function for the gridded (DDM30) engine"""
    parser = argparse.ArgumentParser(
        description="Create lower triangular adjacency matrices from the ISIMIP DDM30 grid."
    )
    parser.add_argument(
        "nc_dir",
        type=Path,
        help="Directory holding ddm30_{flowdir,basins,slopes}_cru_neva.nc.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/"),
        help="Path to save the zarr group. Defaults to 'data/'.",
    )
    parser.add_argument(
        "--gages",
        type=Path,
        default=None,
        help="CSV with STAID, LAT, LON columns for gauge subset matrices.",
    )
    args = parser.parse_args(argv)

    out_path = args.path / "ddm30_adjacency.zarr"
    build_ddm30_adjacency(args.nc_dir, out_path)

    if args.gages is not None:
        with open(args.gages) as f:
            gauges = {row["STAID"]: (float(row["LAT"]), float(row["LON"])) for row in csv.DictReader(f)}
        gages_out_path = args.path / "ddm30_gages_adjacency.zarr"
        build_gauge_adjacencies(args.nc_dir, out_path, gauges, gages_out_path)


if __name__ == "__main__":
    main()
