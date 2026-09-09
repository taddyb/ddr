"""Train the gridded (DDM30 sub-reach) routing model on the Juniata at USGS 01567000.

The gridded equivalent of the MERIT Juniata sample: same gauge, same train/test
split, same KAN inputs, but the network is the ISIMIP DDM30 0.5-degree grid split
into Courant~1 sub-reaches (27 reaches of ~7 km across 4 cells) instead of MERIT
flowlines.

Sub-reaches inherit their parent cell's attributes, except ``log10_uparea`` which is
recomputed per reach from the accumulated area, so the KAN sees within-cell variation.
Attributes are normalised with CONUS-wide statistics so learned parameters are
comparable to a CONUS run.

All inputs ship in ``examples/juniata_gridded/data`` (1.6 MB); nothing external is
needed. Writes per-epoch parameters, predictions and metrics to ``--out`` for the
eval plots. See README.md to run it and docs/gridded_data.md to rebuild the inputs.

Usage:
    uv run python examples/juniata_gridded/train_gridded.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

JUNIATA_CELLS = (139163, 138443, 138444, 138445)
GAGE = "01567000"
GAGE_DA_KM2 = 8657.0
BUNDLE = Path(__file__).parent / "data"  # self-contained inputs, see README.md
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

log = logging.getLogger("train_gridded")


def build_network(bundle: Path = BUNDLE) -> dict:
    """Juniata sub-reach network, attributes, forcing and observations."""
    from ddr_benchmarks.gridded import cell_areas_km2, topo_accumulate

    from ddr.io.readers import read_ic

    g = zarr.open_group(str(bundle / "juniata_subreach_adjacency.zarr"), mode="r")
    parent = g["parent_cell"][:]
    n = len(g["order"][:])
    pos = np.arange(n)  # the bundle is already reindexed to 0..n-1
    r, c = g["indices_0"][:].astype(np.int64), g["indices_1"][:].astype(np.int64)
    assert (r > c).all(), "sub-network must stay lower-triangular"
    adjacency = torch.sparse_coo_tensor(np.stack([r, c]), torch.ones(len(r)), size=(n, n)).to_sparse_csr()

    _, inv, cnt = np.unique(parent, return_inverse=True, return_counts=True)
    k_per_node = cnt[inv]
    area = cell_areas_km2(g["lat"][:]) / k_per_node
    dn_local = np.full(n, -1, dtype=np.int64)
    dn_local[c] = r
    uparea = topo_accumulate(area, dn_local)
    outlet = int(np.argmax(uparea))

    # attributes: parent cell's values, with per-reach accumulated area
    ds = xr.open_dataset(bundle / "ddm30_conus_attributes.nc")
    conus = ds[KAN_INPUTS].to_dataframe()
    rows = conus.loc[[int(p) for p in parent]].reset_index(drop=True)
    rows["log10_uparea"] = np.log10(uparea)
    stats = conus.describe().loc[["mean", "std"]]  # CONUS-wide normalisation
    normalized = ((rows - stats.loc["mean"]) / stats.loc["std"].replace(0, 1)).fillna(0.0)

    qp = read_ic(str(bundle / "juniata_qprime.ic"))
    qr = qp["Qr"].sel(divide_id=[int(p) for p in np.unique(parent)]).transpose("time", "divide_id")
    col = {int(cid): j for j, cid in enumerate(qr.divide_id.values)}
    qvals = np.nan_to_num(qr.values)
    q_prime = np.stack([qvals[:, col[int(p)]] / k_per_node[i] for i, p in enumerate(parent)], axis=1)
    times = pd.to_datetime(qr.time.values)

    obs_ds = read_ic(str(bundle / "juniata_obs.ic"))
    obs = obs_ds["streamflow"].sel(gage_id=GAGE)
    return {
        "n": n,
        "adjacency": adjacency,
        "parent": parent,
        "uparea": uparea,
        "outlet": outlet,
        "length": torch.tensor(g["length_m"][:][pos], dtype=torch.float32),
        "slope": torch.tensor(g["slope"][:][pos], dtype=torch.float32),
        "attrs": torch.tensor(normalized.to_numpy(), dtype=torch.float32),
        "raw_attrs": rows,
        "q_prime": q_prime,
        "times": times,
        "obs": obs,
        "dn_local": dn_local,
    }


def make_config(save_path: Path) -> object:
    """Minimal Config for the routing engine, with p_spatial learnable."""
    from ddr.validation.configs import Config

    return Config(
        name="juniata-gridded",
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


def _slice(net: dict, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    t = net["times"]
    m = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))
    obs = net["obs"].sel(time=slice(start, end)).values
    return net["q_prime"][m], obs


def run(epochs: int, rho: int, warmup: int, lr: float, seed: int, out: Path, bundle: Path) -> None:
    """Train, test, and dump everything the eval plots need."""
    from ddr.geodatazoo.dataclasses import RoutingDataclass
    from ddr.nn.kan import kan
    from ddr.routing.torch_mc import dmc
    from ddr.validation.metrics import Metrics

    torch.manual_seed(seed)
    net = build_network(bundle)
    n = net["n"]
    log.info("network: %d sub-reaches over %d cells, outlet reach %d", n, len(JUNIATA_CELLS), net["outlet"])

    rd = RoutingDataclass(
        adjacency_matrix=net["adjacency"],
        length=net["length"],
        slope=net["slope"],
        side_slope=torch.empty(0),
        top_width=torch.empty(0),
        divide_ids=np.arange(n),
        outflow_idx=[np.array([net["outlet"]])],
        gage_catchment=[GAGE],
        flow_scale=None,
    )
    cfg = make_config(out)
    model = dmc(cfg=cfg, device="cpu")
    nn_model = kan(
        input_var_names=KAN_INPUTS,
        learnable_parameters=LEARNABLE,
        hidden_size=21,
        num_hidden_layers=2,
        grid=50,
        k=2,
        seed=seed,
        device="cpu",
    )
    opt = torch.optim.Adam(nn_model.parameters(), lr=lr)

    qp_train, obs_train = _slice(net, "1981-10-01", "1995-09-30")
    qp_test, obs_test = _slice(net, "1995-10-01", "2010-09-30")
    rng = np.random.default_rng(seed)
    history, param_epochs = [], []

    for epoch in range(1, epochs + 1):
        i0 = int(rng.integers(0, len(qp_train) - rho))
        qp = qp_train[i0 : i0 + rho]
        target = torch.tensor(obs_train[i0 : i0 + rho], dtype=torch.float32)
        params = nn_model(inputs=net["attrs"])
        out_r = model(
            routing_dataclass=rd,
            streamflow=torch.tensor(np.repeat(qp, 24, axis=0), dtype=torch.float32),
            spatial_parameters=params,
        )
        pred = out_r["runoff"].reshape(1, -1, 24).mean(dim=2)[0]
        mask = torch.isfinite(target[warmup:])
        loss = torch.abs(pred[warmup:][mask] - target[warmup:][mask]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
        opt.step()
        eng = model.routing_engine
        with torch.no_grad():
            phys = {
                k: v.detach().numpy().copy()
                for k, v in zip(LEARNABLE, [eng.n, eng.q_spatial, eng.p_spatial], strict=False)
            }
        param_epochs.append(phys)
        rec = {
            "epoch": epoch,
            "loss": float(loss),
            "neg_solve": eng.neg_solve_count / max(eng.neg_solve_total, 1),
            "n_med": float(np.median(phys["n"])),
            "q_med": float(np.median(phys["q_spatial"])),
            "p_med": float(np.median(phys["p_spatial"])),
        }
        history.append(rec)
        log.info(
            "epoch %2d loss %.4f  n %.4f  q %.3f  p %.1f  neg %.4f%%",
            epoch,
            rec["loss"],
            rec["n_med"],
            rec["q_med"],
            rec["p_med"],
            100 * rec["neg_solve"],
        )

    # --- evaluation on the held-out period ---
    with torch.no_grad():
        params = nn_model(inputs=net["attrs"])
        out_r = model(
            routing_dataclass=rd,
            streamflow=torch.tensor(np.repeat(qp_test, 24, axis=0), dtype=torch.float32),
            spatial_parameters=params,
        )
        routed = out_r["runoff"].reshape(1, -1, 24).mean(dim=2).numpy()[0]
    from ddr_benchmarks.gridded import topo_accumulate

    summed = topo_accumulate(qp_test, net["dn_local"])[:, net["outlet"]]
    obs_t = obs_test
    m_r = Metrics(pred=routed[None, warmup:], target=obs_t[None, warmup:])
    m_s = Metrics(pred=summed[None, warmup:], target=obs_t[None, warmup:])
    eng = model.routing_engine
    result = {
        "routed": {"nse": float(m_r.nse[0]), "kge": float(m_r.kge[0]), "rmse": float(m_r.rmse[0])},
        "summed": {"nse": float(m_s.nse[0]), "kge": float(m_s.kge[0]), "rmse": float(m_s.rmse[0])},
        "neg_solve": eng.neg_solve_count / max(eng.neg_solve_total, 1),
        "n_reaches": n,
        "epochs": epochs,
    }
    log.info(
        "TEST routed NSE %.3f KGE %.3f | summed NSE %.3f KGE %.3f",
        result["routed"]["nse"],
        result["routed"]["kge"],
        result["summed"]["nse"],
        result["summed"]["kge"],
    )

    out.mkdir(parents=True, exist_ok=True)
    final = {
        k: v.detach().numpy() for k, v in zip(LEARNABLE, [eng.n, eng.q_spatial, eng.p_spatial], strict=False)
    }
    xr.Dataset(
        {k: ("reach", v) for k, v in final.items()}
        | {
            "length_m": ("reach", net["length"].numpy()),
            "slope": ("reach", net["slope"].numpy()),
            "uparea_km2": ("reach", net["uparea"]),
            "parent_cell": ("reach", net["parent"]),
        },
        coords={"reach": np.arange(n)},
    ).to_netcdf(out / "kan_parameters.nc")
    xr.Dataset(
        {"routed": ("time", routed), "summed": ("time", summed), "observed": ("time", obs_t)},
        coords={
            "time": net["times"][
                (net["times"] >= pd.Timestamp("1995-10-01")) & (net["times"] <= pd.Timestamp("2010-09-30"))
            ][: len(routed)]
        },
    ).to_netcdf(out / "predictions.nc")
    np.savez(out / "parameter_epochs.npz", **{k: np.stack([p[k] for p in param_epochs]) for k in LEARNABLE})
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(nn_model.state_dict(), out / "kan.pt")
    log.info("wrote %s", out)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--rho", type=int, default=90)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("examples/juniata_gridded/runs/latest"))
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.epochs, a.rho, a.warmup, a.lr, a.seed, a.out, a.bundle)


if __name__ == "__main__":
    main()
