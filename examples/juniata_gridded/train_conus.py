"""Train the gridded (DDM30 sub-reach) routing model on CONUS gauges.

Scales the Juniata example to the full usable subset of ``gages_3000.csv``. A 0.5
degree cell is about 2,350 km2, so a basin smaller than that cannot be represented
by a single routing element: 2,416 of the 3,211 gauges are sub-cell and are dropped.
Gauges above ``--min-da-km2`` are snapped to cells by a 3x3 drainage-area match.

Training batches gauges, routes the union of their upstream sub-reaches, and
backpropagates an L1 loss on daily flow into the KAN.

Usage:
    uv run python examples/juniata_gridded/train_conus.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

SUBREACH = Path("data/ddm30/ddm30_subreach_adjacency.zarr")
ATTRS = Path("/mnt/ssd1/data/icechunk/ddm30_conus_attributes.nc")
QPRIME = Path("/mnt/ssd1/data/icechunk/ddm30_conus_uh_retrospective_regridded.ic")
OBS = Path("/mnt/ssd1/data/icechunk/usgs_daily_observations")
GAGES = Path("references/gage_info/gages_3000.csv")
KAN_INPUTS = [
    "SoilGrids1km_clay",
    "aridity",
    "meanelevation",
    "meanP",
    "NDVI",
    "meanslope",
    "log10_uparea",
    "SoilGrids1km_sand",
    "ETPOT_Hargr",
    "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]

log = logging.getLogger("train_conus")


def build_network(min_da_km2: float, min_cov: float, start: str, end: str) -> dict:
    """Sub-reach network, per-gauge upstream sets, attributes, forcing and observations."""
    from ddr_benchmarks.gridded import cell_areas_km2, snap_gauges, topo_accumulate

    from ddr.io.readers import read_ic

    g = zarr.open_group(str(SUBREACH), mode="r")
    parent = g["parent_cell"][:]
    node_ids = g["order"][:]
    rows, cols = g["indices_0"][:], g["indices_1"][:]
    n_all = len(node_ids)
    dn = np.full(n_all, -1, dtype=np.int64)
    dn[cols] = rows

    _, inv, cnt = np.unique(parent, return_inverse=True, return_counts=True)
    k_per_node = cnt[inv]
    uparea = topo_accumulate(cell_areas_km2(g["lat"][:]) / k_per_node, dn)

    # the cell's outlet is its most downstream sub-reach
    sub_idx = node_ids % 1000
    outlet_of_cell: dict[int, int] = {}
    for i in range(n_all):
        c = int(parent[i])
        if c not in outlet_of_cell or sub_idx[i] > sub_idx[outlet_of_cell[c]]:
            outlet_of_cell[c] = i
    ua_cell = {c: float(uparea[i]) for c, i in outlet_of_cell.items()}

    gauges = pd.read_csv(GAGES, dtype={"STAID": str})
    gauges["STAID"] = gauges.STAID.str.zfill(8)
    gauges = snap_gauges(gauges[gauges.DRAIN_SQKM > min_da_km2], ua_cell)
    obs_ds = read_ic(str(OBS))
    gauges = gauges[gauges.STAID.isin(set(obs_ds.gage_id.values.astype(str)))]
    cov = (
        obs_ds["streamflow"]
        .sel(gage_id=gauges.STAID.tolist(), time=slice(start, end))
        .notnull()
        .mean("time")
        .values
    )
    gauges = gauges[cov > min_cov].reset_index(drop=True)
    log.info("%d gauges after DA match and observation coverage", len(gauges))

    # upstream reach set per gauge, by BFS on the reverse adjacency
    up: list[list[int]] = [[] for _ in range(n_all)]
    for r, c in zip(rows, cols, strict=True):
        up[int(r)].append(int(c))
    upstream_sets = []
    for cell in gauges.cell:
        stack, seen = [outlet_of_cell[int(cell)]], set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(up[v])
        upstream_sets.append(np.array(sorted(seen), dtype=np.int64))
    log.info(
        "basins: median %d reaches, max %d",
        int(np.median([len(s) for s in upstream_sets])),
        max(len(s) for s in upstream_sets),
    )

    ds = xr.open_dataset(ATTRS)
    conus = ds[KAN_INPUTS].to_dataframe()
    stats = conus.describe().loc[["mean", "std"]]
    rows_attr = conus.reindex([int(p) for p in parent]).reset_index(drop=True)
    rows_attr["log10_uparea"] = np.log10(np.maximum(uparea, 1.0))
    attrs = ((rows_attr - stats.loc["mean"]) / stats.loc["std"].replace(0, 1)).fillna(0.0)

    qp = read_ic(str(QPRIME))
    qr = qp["Qr"].transpose("time", "divide_id")
    cell_q = np.nan_to_num(qr.values.astype(np.float32))
    q_cols = {int(c): j for j, c in enumerate(qr.divide_id.values)}
    times = pd.to_datetime(qr.time.values)
    node_qcol = np.array([q_cols.get(int(p), -1) for p in parent])

    return {
        "node_ids": node_ids,
        "parent": parent,
        "dn": dn,
        "rows": rows,
        "cols": cols,
        "uparea": uparea,
        "k_per_node": k_per_node,
        "outlet_of_cell": outlet_of_cell,
        "gauges": gauges,
        "upstream_sets": upstream_sets,
        "attrs": torch.tensor(attrs.to_numpy(), dtype=torch.float32),
        "length": g["length_m"][:],
        "slope": g["slope"][:],
        "cell_q": cell_q,
        "node_qcol": node_qcol,
        "times": times,
        "obs_ds": obs_ds,
    }


def make_config(save_path: Path):  # type: ignore[no-untyped-def] # returns ddr Config
    """Routing config with n, q_spatial and p_spatial learnable."""
    from ddr.validation.configs import Config

    return Config(
        name="ddm30-conus",
        mode="training",
        geodataset="merit",
        device="cpu",
        data_sources={"geospatial_fabric_gpkg": "unused", "conus_adjacency": "unused"},
        params={
            "save_path": str(save_path),
            "parameter_ranges": {"n": [0.02, 0.2], "q_spatial": [0.0, 1.0], "p_spatial": [1.0, 200.0]},
            "log_space_parameters": ["p_spatial"],
            "defaults": {"p_spatial": 21},
        },
        kan={"input_var_names": KAN_INPUTS, "learnable_parameters": LEARNABLE},
        experiment={"start_time": "1981/10/01", "end_time": "1995/09/30"},
    )


def to_physical(params: dict) -> dict:
    """KAN sigmoid outputs -> physical parameter values, using the config's bounds."""
    from ddr.routing.utils import denormalize

    bounds = {"n": [0.02, 0.2], "q_spatial": [0.0, 1.0], "p_spatial": [1.0, 200.0]}
    log_space = {"p_spatial"}
    return {
        k: denormalize(v.detach(), bounds[k], log_space=k in log_space).numpy().copy()
        for k, v in params.items()
    }


def batch_inputs(net: dict, idx: np.ndarray, t0: int, n_days: int) -> tuple:
    """Union subnetwork, forcing and observations for a batch of gauges."""
    nodes = np.unique(np.concatenate([net["upstream_sets"][i] for i in idx]))
    local = {int(p): i for i, p in enumerate(nodes)}
    m = np.isin(net["rows"], nodes) & np.isin(net["cols"], nodes)
    r = np.array([local[int(x)] for x in net["rows"][m]])
    c = np.array([local[int(x)] for x in net["cols"][m]])
    n = len(nodes)
    adjacency = torch.sparse_coo_tensor(np.stack([r, c]), torch.ones(len(r)), size=(n, n)).to_sparse_csr()
    qcol = net["node_qcol"][nodes]
    q = np.zeros((n_days, n), dtype=np.float32)
    has = qcol >= 0
    q[:, has] = net["cell_q"][t0 : t0 + n_days][:, qcol[has]] / net["k_per_node"][nodes][has]
    outflow = [np.array([local[net["outlet_of_cell"][int(net["gauges"].cell.iloc[i])]]]) for i in idx]
    return nodes, adjacency, q, outflow


def run(args: argparse.Namespace) -> None:
    """Train on CONUS gauges and dump parameters, predictions and history."""
    from ddr.geodatazoo.dataclasses import RoutingDataclass
    from ddr.nn.kan import kan
    from ddr.routing.torch_mc import dmc
    from ddr.validation.metrics import Metrics

    torch.manual_seed(args.seed)
    net = build_network(args.min_da_km2, 0.9, args.train_start, args.test_end)
    gauges = net["gauges"]
    times = net["times"]
    train_mask = (times >= pd.Timestamp(args.train_start)) & (times <= pd.Timestamp(args.train_end))
    train_i = np.flatnonzero(train_mask)

    cfg = make_config(args.out)
    model = dmc(cfg=cfg, device="cpu")
    nn_model = kan(
        input_var_names=KAN_INPUTS,
        learnable_parameters=LEARNABLE,
        hidden_size=21,
        num_hidden_layers=2,
        grid=50,
        k=2,
        seed=args.seed,
        device="cpu",
    )
    opt = torch.optim.Adam(nn_model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    obs_all = net["obs_ds"]["streamflow"].sel(gage_id=gauges.STAID.tolist())
    history, param_epochs = [], []

    for epoch in range(1, args.epochs + 1):
        if epoch == max(2, args.epochs // 2):
            for pg in opt.param_groups:
                pg["lr"] = args.lr / 2
        perm = rng.permutation(len(gauges))
        ep_loss, n_batch, t_ep = 0.0, 0, time.time()
        for b0 in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[b0 : b0 + args.batch_size]
            t0 = int(rng.integers(train_i[0], train_i[-1] - args.rho))
            nodes, adjacency, q, outflow = batch_inputs(net, idx, t0, args.rho)
            window = times[t0 : t0 + args.rho]
            target = torch.tensor(
                obs_all.isel(gage_id=idx).sel(time=slice(window[0], window[-1])).values, dtype=torch.float32
            )
            if target.shape[1] < args.rho:
                continue
            rd = RoutingDataclass(
                adjacency_matrix=adjacency,
                length=torch.tensor(net["length"][nodes], dtype=torch.float32),
                slope=torch.tensor(net["slope"][nodes], dtype=torch.float32),
                side_slope=torch.empty(0),
                top_width=torch.empty(0),
                divide_ids=np.arange(len(nodes)),
                outflow_idx=outflow,
                gage_catchment=gauges.STAID.iloc[idx].tolist(),
                flow_scale=None,
            )
            params = nn_model(inputs=net["attrs"][nodes])
            out = model(
                routing_dataclass=rd,
                streamflow=torch.tensor(np.repeat(q, 24, axis=0)),
                spatial_parameters=params,
            )
            pred = out["runoff"].reshape(len(idx), -1, 24).mean(dim=2)
            # apply the same drainage-area correction evaluation uses, so the KAN is not
            # asked to absorb an area mismatch (per-gauge ratio spans 0.82-1.21) into n
            pred = pred * torch.tensor((1.0 / gauges.da_ratio.values[idx])[:, None], dtype=torch.float32)
            mask = torch.isfinite(target[:, args.warmup :])
            if not mask.any():
                continue
            loss = torch.abs(pred[:, args.warmup :][mask] - target[:, args.warmup :][mask]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            n_batch += 1
        with torch.no_grad():
            p_all = nn_model(inputs=net["attrs"])
        phys = to_physical(p_all)
        param_epochs.append(phys)
        eng = model.routing_engine
        rec = dict(
            epoch=epoch,
            loss=ep_loss / max(n_batch, 1),
            batches=n_batch,
            minutes=(time.time() - t_ep) / 60,
            neg_solve=eng.neg_solve_count / max(eng.neg_solve_total, 1),
            **{f"{k}_med": float(np.median(v)) for k, v in phys.items()},
        )
        history.append(rec)
        log.info(
            "epoch %2d loss %.3f (%d batches, %.1f min) n %.4f q %.3f p %.3f neg %.4f%%",
            epoch,
            rec["loss"],
            n_batch,
            rec["minutes"],
            rec["n_med"],
            rec["q_spatial_med"],
            rec["p_spatial_med"],
            100 * rec["neg_solve"],
        )
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
        torch.save(nn_model.state_dict(), args.out / "kan.pt")

    # --- evaluation on the held-out period, in gauge batches ---
    test_i = np.flatnonzero((times >= pd.Timestamp(args.test_start)) & (times <= pd.Timestamp(args.test_end)))
    n_days = len(test_i)
    rows_out = []
    from ddr_benchmarks.gridded import topo_accumulate

    with torch.no_grad():
        for b0 in range(0, len(gauges), args.batch_size):
            idx = np.arange(b0, min(b0 + args.batch_size, len(gauges)))
            nodes, adjacency, q, outflow = batch_inputs(net, idx, int(test_i[0]), n_days)
            rd = RoutingDataclass(
                adjacency_matrix=adjacency,
                length=torch.tensor(net["length"][nodes], dtype=torch.float32),
                slope=torch.tensor(net["slope"][nodes], dtype=torch.float32),
                side_slope=torch.empty(0),
                top_width=torch.empty(0),
                divide_ids=np.arange(len(nodes)),
                outflow_idx=outflow,
                gage_catchment=gauges.STAID.iloc[idx].tolist(),
                flow_scale=None,
            )
            out = model(
                routing_dataclass=rd,
                streamflow=torch.tensor(np.repeat(q, 24, axis=0)),
                spatial_parameters=nn_model(inputs=net["attrs"][nodes]),
            )
            routed = out["runoff"].reshape(len(idx), -1, 24).mean(dim=2).numpy()
            dn_local = np.full(len(nodes), -1, dtype=np.int64)
            local = {int(p): i for i, p in enumerate(nodes)}
            for r, c in zip(net["rows"], net["cols"], strict=True):
                if int(r) in local and int(c) in local:
                    dn_local[local[int(c)]] = local[int(r)]
            summed_all = topo_accumulate(q, dn_local)
            summed = np.stack([summed_all[:, o[0]] for o in outflow])
            obs = (
                obs_all.isel(gage_id=idx)
                .sel(time=slice(times[test_i[0]], times[test_i[-1]]))
                .values[:, : routed.shape[1]]
            )
            w = 30
            da = (1.0 / gauges.da_ratio.values[idx])[:, None]
            m_r = Metrics(pred=routed[:, w:] * da, target=obs[:, w:])
            m_s = Metrics(pred=summed[:, w:] * da, target=obs[:, w:])
            for j, gi in enumerate(idx):
                rows_out.append(
                    {
                        "STAID": gauges.STAID.iloc[gi],
                        "DRAIN_SQKM": gauges.DRAIN_SQKM.iloc[gi],
                        "cell": int(gauges.cell.iloc[gi]),
                        "da_ratio": float(gauges.da_ratio.iloc[gi]),
                        "nse_routed": float(m_r.nse[j]),
                        "nse_summed": float(m_s.nse[j]),
                        "kge_routed": float(m_r.kge[j]),
                        "kge_summed": float(m_s.kge[j]),
                    }
                )
            log.info("eval %d/%d gauges", len(rows_out), len(gauges))

    res = pd.DataFrame(rows_out)
    res.to_csv(args.out / "gauge_metrics.csv", index=False)
    with torch.no_grad():
        final = to_physical(nn_model(inputs=net["attrs"]))
    xr.Dataset(
        {k: ("reach", v) for k, v in final.items()}
        | {
            "length_m": ("reach", net["length"]),
            "slope": ("reach", net["slope"]),
            "uparea_km2": ("reach", net["uparea"]),
            "parent_cell": ("reach", net["parent"]),
        },
        coords={"reach": np.arange(len(net["node_ids"]))},
    ).to_netcdf(args.out / "kan_parameters.nc")
    np.savez(
        args.out / "parameter_epochs.npz", **{k: np.stack([p[k] for p in param_epochs]) for k in LEARNABLE}
    )
    summary = {
        "gauges": len(res),
        "median_nse": float(res.nse_routed.median()),
        "median_kge": float(res.kge_routed.median()),
        "median_nse_summed": float(res.nse_summed.median()),
        "beats_summed": float((res.nse_routed > res.nse_summed).mean()),
        "nse_gt_05": int((res.nse_routed > 0.5).sum()),
        "epochs": args.epochs,
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "TEST median NSE %.3f (summed %.3f), beats summed %.0f%%, NSE>0.5 at %d/%d",
        summary["median_nse"],
        summary["median_nse_summed"],
        100 * summary["beats_summed"],
        summary["nse_gt_05"],
        summary["gauges"],
    )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rho", type=int, default=90)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-da-km2", type=float, default=2000.0)
    p.add_argument("--train-start", default="1981-10-01")
    p.add_argument("--train-end", default="1995-09-30")
    p.add_argument("--test-start", default="1995-10-01")
    p.add_argument("--test-end", default="1997-09-30")
    p.add_argument("--out", type=Path, default=Path("examples/juniata_gridded/runs/conus"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a)


if __name__ == "__main__":
    main()
