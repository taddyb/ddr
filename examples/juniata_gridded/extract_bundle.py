"""Maintainer script: cut a self-contained bundle for the gridded Juniata example.

Needs the full gridded stores (see docs/gridded_data.md). Users do not run this;
they receive the output committed under examples/juniata_gridded/data/.

    uv run python examples/juniata_gridded/extract_bundle.py --out examples/juniata_gridded/data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

JUNIATA_CELLS = (139163, 138443, 138444, 138445)
GAGE = "01567000"
SUBREACH = Path("data/ddm30/ddm30_subreach_adjacency.zarr")
ATTRS = Path("/mnt/ssd1/data/icechunk/ddm30_conus_attributes.nc")
QPRIME = Path("/mnt/ssd1/data/icechunk/ddm30_conus_uh_retrospective_regridded.ic")
OBS = Path("/mnt/ssd1/data/icechunk/usgs_daily_observations")
START, END = "1980-01-01", "2010-12-31"

log = logging.getLogger("extract_bundle")


def _write_local_ic(path: Path, ds: xr.Dataset, message: str) -> None:
    """Create a local icechunk store at path and commit ds into it."""
    import icechunk as ic

    repo = ic.Repository.create(ic.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    ds.to_zarr(session.store, consolidated=False, mode="w")
    session.commit(message)


def main() -> None:
    """Write the bundle."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("examples/juniata_gridded/data"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # --- sub-reach subgraph, reindexed to a compact 0..n-1 ordering ---
    g = zarr.open_group(str(SUBREACH), mode="r")
    parent_all = g["parent_cell"][:]
    keep = np.flatnonzero(np.isin(parent_all, JUNIATA_CELLS))
    local = {int(p): i for i, p in enumerate(keep)}
    m = np.isin(g["indices_0"][:], keep) & np.isin(g["indices_1"][:], keep)
    rows = np.array([local[int(x)] for x in g["indices_0"][:][m]], dtype=np.int32)
    cols = np.array([local[int(x)] for x in g["indices_1"][:][m]], dtype=np.int32)
    assert (rows > cols).all(), "subgraph must stay lower-triangular"

    n = len(keep)
    root = zarr.open_group(args.out / "juniata_subreach_adjacency.zarr", mode="w")
    root["order"] = g["order"][:][keep].astype(np.int64)
    root["parent_cell"] = parent_all[keep].astype(np.int32)
    root["indices_0"], root["indices_1"] = rows, cols
    root["values"] = np.ones(len(rows), dtype=np.uint8)
    for k in ("length_m", "slope", "lat", "lon"):
        root[k] = g[k][:][keep]
    root.attrs.update(
        {
            "format": "COO",
            "shape": [n, n],
            "geodataset": "ddm30",
            "subdivided": True,
            "cell_id_scheme": "node id = cell id * 1000 + sub index (upstream-most first)",
            "data_types": {"indices_0": "int32", "indices_1": "int32", "values": "uint8"},
        }
    )
    log.info("adjacency: %d sub-reaches, %d edges", n, len(rows))

    # --- attributes: the whole CONUS table, so normalisation matches a CONUS run ---
    xr.open_dataset(ATTRS).to_netcdf(args.out / "ddm30_conus_attributes.nc")

    # --- forcing and observations, trimmed to the four cells and the sample period ---
    from ddr.io.readers import read_ic

    q = read_ic(str(QPRIME))
    q_sub = q.sel(divide_id=list(JUNIATA_CELLS), time=slice(START, END)).load()
    _write_local_ic(args.out / "juniata_qprime.ic", q_sub, "gridded Juniata Q' subset")
    o = read_ic(str(OBS))
    o_sub = o.sel(gage_id=[GAGE], time=slice(START, END)).load()
    _write_local_ic(args.out / "juniata_obs.ic", o_sub, f"USGS obs subset ({GAGE})")

    pd.DataFrame(
        [
            {
                "STAID": GAGE,
                "STANAME": "JUNIATA RIVER AT NEWPORT, PA",
                "DRAIN_SQKM": 8657.0,
                "LAT_GAGE": 40.4785,
                "LNG_GAGE": -77.1294,
                "cell": 138445,
                "da_ratio": 9419.0 / 8657.0,
            }
        ]
    ).to_csv(args.out / "juniata_gage.csv", index=False)
    log.info("wrote bundle to %s", args.out)


if __name__ == "__main__":
    main()
