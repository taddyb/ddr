"""Evaluation plots for a gridded Juniata training run.

Mirrors the ddrs eval-plot families (hydrograph, parameter maps, convergence,
metrics, channel geometry) for the DDM30 sub-reach network. Reads the outputs of
``train_gridded.py`` and writes PNGs next to them.

Usage:
    uv run python examples/juniata_gridded/eval_plots.py --run examples/juniata_gridded/runs/gridded_30ep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PARAMS = ["n", "q_spatial", "p_spatial"]
LABELS = {"n": "Manning's n", "q_spatial": "Leopold-Maddock q", "p_spatial": "Leopold-Maddock p"}


def hydrograph(run: Path) -> Path:
    """Routed vs summed-Q' vs observed at the gauge, full test period and a wet year."""
    ds = xr.open_dataset(run / "predictions.nc")
    metrics = json.loads((run / "metrics.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, (a, b) in zip(axes, [(None, None), ("1996-01-01", "1996-12-31")], strict=False):
        d = ds.sel(time=slice(a, b))
        ax.plot(d.time, d.observed, color="k", lw=1.1, label="observed (USGS 01567000)")
        ax.plot(d.time, d.routed, color="#0072B2", lw=1.0, label="routed (gridded sub-reach)")
        ax.plot(d.time, d.summed, color="#D55E00", lw=0.8, ls="--", alpha=0.8, label="summed Q' baseline")
        ax.set_ylabel("discharge (m$^3$/s)")
        ax.grid(alpha=0.3)
    axes[0].set_title(
        f"Juniata at 01567000, gridded DDM30 sub-reaches ({metrics['n_reaches']} reaches)\n"
        f"routed NSE {metrics['routed']['nse']:.3f} / KGE {metrics['routed']['kge']:.3f}   "
        f"summed NSE {metrics['summed']['nse']:.3f} / KGE {metrics['summed']['kge']:.3f}"
    )
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set_title("water year 1996 detail")
    fig.tight_layout()
    p = run / "hydrograph.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def parameter_fields(run: Path) -> Path:
    """Learned parameters along the network: profile, distribution, and vs drainage area."""
    ds = xr.open_dataset(run / "kan_parameters.nc")
    order = np.argsort(ds.uparea_km2.values)  # upstream to downstream
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    cells = ds.parent_cell.values
    palette = {c: plt.cm.tab10(i) for i, c in enumerate(np.unique(cells))}
    for row, key in enumerate(PARAMS):
        v = ds[key].values
        ax = axes[row, 0]
        ax.scatter(np.arange(len(v)), v[order], c=[palette[c] for c in cells[order]], s=28)
        ax.set_ylabel(LABELS[key])
        ax.set_xlabel("reach, upstream to downstream" if row == 2 else "")
        ax.grid(alpha=0.3)
        ax = axes[row, 1]
        ax.hist(v, bins=12, color="#0072B2", alpha=0.85)
        ax.set_xlabel(LABELS[key])
        ax.set_ylabel("reaches")
        ax.grid(alpha=0.3)
        ax = axes[row, 2]
        ax.scatter(ds.uparea_km2.values, v, c=[palette[c] for c in cells], s=28)
        ax.set_xscale("log")
        ax.set_xlabel("upstream drainage area (km$^2$)" if row == 2 else "")
        ax.set_ylabel(LABELS[key])
        ax.grid(alpha=0.3)
    axes[0, 0].set_title("parameter along the network")
    axes[0, 1].set_title("distribution")
    axes[0, 2].set_title("vs drainage area (colour = DDM30 cell)")
    fig.suptitle("Learned KAN parameters, gridded Juniata", y=1.0)
    fig.tight_layout()
    p = run / "parameter_fields.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def convergence(run: Path) -> Path:
    """Per-epoch parameter drift and training loss."""
    z = np.load(run / "parameter_epochs.npz")
    hist = pd.read_csv(run / "history.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, key in zip(axes[:3], PARAMS, strict=False):
        arr = z[key]
        for r in range(arr.shape[1]):
            ax.plot(np.arange(1, len(arr) + 1), arr[:, r], color="#0072B2", alpha=0.35, lw=0.9)
        ax.plot(np.arange(1, len(arr) + 1), np.median(arr, axis=1), color="k", lw=2, label="median")
        ax.set_xlabel("epoch")
        ax.set_ylabel(LABELS[key])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[3].plot(hist.epoch, hist.loss, color="#D55E00", marker="o", ms=3)
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("L1 loss (m$^3$/s)")
    axes[3].set_title("training loss (new random window each epoch)")
    axes[3].grid(alpha=0.3)
    fig.suptitle("Parameter convergence, gridded Juniata", y=1.03)
    fig.tight_layout()
    p = run / "convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def channel_geometry(run: Path) -> Path:
    """Baseflow width and depth implied by the learned n, p, q; the internal physics check."""
    import torch

    from ddr.geometry.trapezoidal import compute_trapezoidal_geometry

    ds = xr.open_dataset(run / "kan_parameters.nc")
    pred = xr.open_dataset(run / "predictions.nc")
    q_ref = float(np.nanmedian(pred.observed.values))
    n_r = len(ds.reach)
    geom = compute_trapezoidal_geometry(
        n=torch.tensor(ds["n"].values, dtype=torch.float32),
        p_spatial=torch.tensor(ds["p_spatial"].values, dtype=torch.float32),
        q_spatial=torch.tensor(ds["q_spatial"].values, dtype=torch.float32),
        discharge=torch.full((n_r,), q_ref),
        slope=torch.tensor(np.clip(ds.slope.values, 1e-6, None), dtype=torch.float32),
        depth_lb=0.01,
        bottom_width_lb=0.1,
    )
    w = geom["top_width"].numpy()
    d = geom["depth"].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].scatter(ds.uparea_km2.values, w, s=28, color="#0072B2")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("top width (m)")
    axes[1].scatter(ds.uparea_km2.values, d, s=28, color="#009E73")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("depth (m)")
    axes[2].scatter(ds.uparea_km2.values, w / d, s=28, color="#CC79A7")
    axes[2].set_xscale("log")
    axes[2].set_ylabel("width : depth")
    axes[2].axhspan(10, 60, color="grey", alpha=0.18, label="typical natural channels")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("upstream drainage area (km$^2$)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Channel geometry implied by learned parameters at Q = {q_ref:.0f} m$^3$/s", y=1.04)
    fig.tight_layout()
    p = run / "channel_geometry.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def metrics_panel(run: Path) -> Path:
    """Routed vs summed baseline across metrics, plus flow-duration curves."""
    from ddr.validation.metrics import Metrics

    ds = xr.open_dataset(run / "predictions.nc")
    obs, routed, summed = ds.observed.values, ds.routed.values, ds.summed.values
    m_r = Metrics(pred=routed[None, 30:], target=obs[None, 30:])
    m_s = Metrics(pred=summed[None, 30:], target=obs[None, 30:])
    names = ["nse", "kge", "rmse", "bias_percent", "flv", "fhv"]
    vals_r, vals_s = [], []
    for k in names:
        vals_r.append(float(np.atleast_1d(getattr(m_r, k))[0]) if hasattr(m_r, k) else np.nan)
        vals_s.append(float(np.atleast_1d(getattr(m_s, k))[0]) if hasattr(m_s, k) else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(names))
    axes[0].bar(x - 0.19, vals_r, 0.38, label="routed", color="#0072B2")
    axes[0].bar(x + 0.19, vals_s, 0.38, label="summed Q'", color="#D55E00")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("metrics, held-out 1995-2010")
    for series, lab, col in [
        (obs, "observed", "k"),
        (routed, "routed", "#0072B2"),
        (summed, "summed Q'", "#D55E00"),
    ]:
        s = np.sort(series[np.isfinite(series)])[::-1]
        axes[1].plot(np.arange(1, len(s) + 1) / len(s) * 100, s, label=lab, color=col, lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("exceedance probability (%)")
    axes[1].set_ylabel("discharge (m$^3$/s)")
    axes[1].set_title("flow duration curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = run / "metrics.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("examples/juniata_gridded/runs/gridded_30ep"))
    a = ap.parse_args()
    for fn in (hydrograph, parameter_fields, convergence, channel_geometry, metrics_panel):
        print("wrote", fn(a.run))


if __name__ == "__main__":
    main()
