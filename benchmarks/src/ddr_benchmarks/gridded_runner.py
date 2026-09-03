"""CONUS-scale gridded (DDM30) routing benchmark runner.

Routes MERIT dHBV2 lateral inflows, aggregated to ISIMIP DDM30 0.5-degree
cells, through ddr's Muskingum-Cunge engine with fixed spatial parameters
(no KAN), and scores daily discharge against USGS observations at
drainage-area-matched large gauges. A summed-Q' (instantaneous
accumulation, no routing) baseline quantifies the value of routing.

This is a fixed-parameter benchmark: gridded KAN attributes do not exist
yet, so Manning's n and q_spatial are spatially uniform.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from .gridded import (
    assign_cells,
    cell_areas_km2,
    downstream_closure,
    snap_gauges,
    topo_accumulate,
)

NCOLS = 720


@dataclass
class GriddedPaths:
    """Data sources for the CONUS gridded benchmark."""

    ddm30_zarr: Path
    riv_shapefile: Path
    qr_icechunk: Path
    obs_icechunk: Path
    gages_csv: Path
    cache_dir: Path


def _open_icechunk(path: Path) -> "xr.Dataset":  # type: ignore[name-defined]  # noqa: F821
    import icechunk
    import xarray as xr

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open(storage)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def comid_cell_map(paths: GriddedPaths) -> pd.DataFrame:
    """COMID -> DDM30 cell id via flowline midpoints (cached parquet)."""
    cache = paths.cache_dir / "comid_cell.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from pyogrio import read_dataframe

    fp = read_dataframe(str(paths.riv_shapefile), columns=["COMID"])
    mid = fp.geometry.interpolate(0.5, normalized=True)
    df = pd.DataFrame(
        {
            "COMID": fp["COMID"].astype(np.int64),
            "cell": assign_cells(mid.y.values, mid.x.values),
        }
    )
    df.to_parquet(cache)
    return df


def _make_config(save_path: Path) -> "Config":  # type: ignore[name-defined]  # noqa: F821
    from ddr.validation.configs import Config

    return Config(
        name="gridded-conus-benchmark",
        mode="testing",
        geodataset="merit",
        device="cpu",
        data_sources={
            "geospatial_fabric_gpkg": "unused",
            "conus_adjacency": "unused",
        },
        params={
            "save_path": str(save_path),
            "parameter_ranges": {"n": [0.02, 0.2], "q_spatial": [0.0, 1.0]},
            "log_space_parameters": [],
            "defaults": {"p_spatial": 21},
        },
        kan={"input_var_names": [], "learnable_parameters": ["n", "q_spatial"]},
        experiment={"start_time": "1995/10/01", "end_time": "2000/09/30"},
    )


def run_conus_benchmark(
    paths: GriddedPaths,
    start: str = "1995-10-01",
    end: str = "1997-09-30",
    n_manning: float = 0.05,
    q_spatial: float = 0.4,
    min_da_km2: float = 5000.0,
    min_obs_coverage: float = 0.9,
) -> dict:
    """Run the CONUS gridded benchmark.

    Returns
    -------
    dict
        ``gauges``: per-gauge DataFrame (STAID, cell, da_ratio, nse_routed,
        nse_summed, kge_routed, kge_summed); ``neg_solve_rate``: fraction of
        negative triangular solves; ``n_cells``, ``runtime_s``.
    """
    root = zarr.open_group(store=paths.ddm30_zarr, mode="r")
    order = root["order"][:]
    rows_g, cols_g = root["indices_0"][:], root["indices_1"][:]
    pos = {int(cid): i for i, cid in enumerate(order)}
    dn = np.full(len(order), -1, dtype=np.int64)
    dn[cols_g] = rows_g

    # --- network: cells receiving MERIT Q' plus their downstream chains ---
    comid_cell = comid_cell_map(paths)
    forced_pos = np.array(sorted({pos[c] for c in comid_cell.cell.unique() if c in pos}))
    node_pos = downstream_closure(forced_pos, dn)
    n = len(node_pos)
    pos_to_c = {int(p): i for i, p in enumerate(node_pos)}

    mask = np.isin(rows_g, node_pos) & np.isin(cols_g, node_pos)
    r_c = np.array([pos_to_c[p] for p in rows_g[mask]])
    c_c = np.array([pos_to_c[p] for p in cols_g[mask]])
    adjacency = torch.sparse_coo_tensor(
        np.stack([r_c, c_c]), torch.ones(len(r_c)), size=(n, n)
    ).to_sparse_csr()
    dn_c = np.full(n, -1, dtype=np.int64)
    dn_c[c_c] = r_c

    # --- gauges: DA-matched snap against accumulated cell areas ---
    upstream_area_all = topo_accumulate(cell_areas_km2(root["lat"][:]), dn)
    upstream_area = {int(order[p]): float(upstream_area_all[p]) for p in node_pos}

    gauges = pd.read_csv(paths.gages_csv, dtype={"STAID": str})
    gauges["STAID"] = gauges.STAID.str.zfill(8)
    gauges = gauges[gauges.DRAIN_SQKM > min_da_km2]
    gauges = snap_gauges(gauges, upstream_area)

    obs_ds = _open_icechunk(paths.obs_icechunk)
    obs_ids = set(obs_ds.gage_id.values.astype(str))
    gauges = gauges[gauges.STAID.isin(obs_ids)]
    coverage = (
        obs_ds["streamflow"]
        .sel(gage_id=gauges.STAID.tolist(), time=slice(start, end))
        .notnull()
        .mean("time")
        .values
    )
    gauges = gauges[coverage > min_obs_coverage]
    outflow_idx = [np.array([pos_to_c[pos[c]]]) for c in gauges.cell]

    # --- lateral inflow: daily Qr summed per cell, repeated to hourly ---
    qp_daily = _aggregate_qprime(paths, comid_cell, node_pos, order, pos_to_c, pos, start, end)

    # --- route ---
    from ddr.geodatazoo.dataclasses import RoutingDataclass
    from ddr.routing.torch_mc import dmc

    rd = RoutingDataclass(
        adjacency_matrix=adjacency,
        length=torch.tensor(root["length_m"][:][node_pos], dtype=torch.float32),
        slope=torch.tensor(root["slope"][:][node_pos], dtype=torch.float32),
        side_slope=torch.empty(0),
        top_width=torch.empty(0),
        divide_ids=np.arange(n),
        outflow_idx=outflow_idx,
        gage_catchment=gauges.STAID.tolist(),
        flow_scale=None,
    )
    n_norm = (n_manning - 0.02) / (0.2 - 0.02)
    spatial_parameters = {
        "n": torch.full((n,), n_norm),
        "q_spatial": torch.full((n,), q_spatial),
    }
    t0 = time.time()
    model = dmc(cfg=_make_config(paths.cache_dir), device="cpu")
    with torch.no_grad():
        out = model(
            routing_dataclass=rd,
            streamflow=torch.tensor(np.repeat(qp_daily, 24, axis=0)),
            spatial_parameters=spatial_parameters,
        )
    runtime_s = time.time() - t0
    engine = model.routing_engine
    neg_solve_rate = engine.neg_solve_count / max(engine.neg_solve_total, 1)
    routed_daily = out["runoff"].numpy().reshape(len(gauges), -1, 24).mean(axis=2)

    # --- baseline and evaluation ---
    summed_all = topo_accumulate(qp_daily, dn_c)
    summed_daily = np.stack([summed_all[:, idx[0]] for idx in outflow_idx])
    obs = obs_ds["streamflow"].sel(gage_id=gauges.STAID.tolist(), time=slice(start, end)).values

    from ddr.validation.metrics import Metrics

    warmup = 30
    da_scale = (1.0 / gauges.da_ratio.values)[:, None]
    m_routed = Metrics(pred=routed_daily[:, warmup:] * da_scale, target=obs[:, warmup:])
    m_summed = Metrics(pred=summed_daily[:, warmup:] * da_scale, target=obs[:, warmup:])

    result = gauges[["STAID", "DRAIN_SQKM", "cell", "da_ratio"]].reset_index(drop=True)
    result["nse_routed"], result["nse_summed"] = m_routed.nse, m_summed.nse
    result["kge_routed"], result["kge_summed"] = m_routed.kge, m_summed.kge
    return {
        "gauges": result,
        "neg_solve_rate": neg_solve_rate,
        "n_cells": n,
        "runtime_s": runtime_s,
    }


def _aggregate_qprime(
    paths: GriddedPaths,
    comid_cell: pd.DataFrame,
    node_pos: np.ndarray,
    order: np.ndarray,
    pos_to_c: dict[int, int],
    pos: dict[int, int],
    start: str,
    end: str,
) -> np.ndarray:
    ds = _open_icechunk(paths.qr_icechunk)
    cell_to_node = {int(order[p]): pos_to_c[int(p)] for p in node_pos}
    contributing = comid_cell[comid_cell.cell.isin(cell_to_node)]
    contributing = contributing[contributing.COMID.isin(pd.Index(ds.divide_id.values))]
    qr = ds["Qr"].sel(divide_id=contributing.COMID.values, time=slice(start, end)).values
    qp = np.zeros((qr.shape[1], len(node_pos)), dtype=np.float32)
    node_idx = contributing.cell.map(cell_to_node).values
    np.add.at(qp.T, node_idx, np.nan_to_num(qr))
    return qp
