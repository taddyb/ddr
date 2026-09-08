"""Evaluation plots for the CONUS gridded training run.

Mirrors the ddrs eval-plot families at CONUS scale: parameter maps over the DDM30
sub-reach network, parameter-versus-attribute relations, convergence, the metric
distribution against the summed-Q' baseline, and the channel-geometry physics check.

Usage:
    uv run python examples/juniata_gridded/eval_plots_conus.py --run examples/juniata_gridded/runs/conus_12ep
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
NCOLS, LAT0, LON0, CELL = 720, -55.75, -179.75, 0.5


def _lonlat(parent_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return LON0 + (parent_cell % NCOLS) * CELL, LAT0 + (parent_cell // NCOLS) * CELL


def parameter_maps(run: Path) -> Path:
    """Spatial map of each learned parameter, averaged to its DDM30 cell."""
    ds = xr.open_dataset(run / "kan_parameters.nc")
    lon, lat = _lonlat(ds.parent_cell.values)
    df = pd.DataFrame({"lon": lon, "lat": lat, **{k: ds[k].values for k in PARAMS}})
    cells = df.groupby(["lon", "lat"]).mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for ax, key in zip(axes, PARAMS, strict=False):
        v = cells[key]
        lo, hi = np.percentile(v, [2, 98])
        s = ax.scatter(cells.lon, cells.lat, c=v, s=5, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(LABELS[key])
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect(1.3)
        fig.colorbar(s, ax=ax, shrink=0.85)
    fig.suptitle("Learned KAN parameters over the DDM30 grid (cell means)", y=1.02)
    fig.tight_layout()
    p = run / "parameter_maps.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def parameter_relations(run: Path) -> Path:
    """Each parameter against drainage area and against slope, with distributions."""
    ds = xr.open_dataset(run / "kan_parameters.nc")
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    ua, slope = ds.uparea_km2.values, ds.slope.values
    for row, key in enumerate(PARAMS):
        v = ds[key].values
        axes[row, 0].hexbin(ua, v, xscale="log", gridsize=45, cmap="Blues", mincnt=1)
        axes[row, 0].set_ylabel(LABELS[key])
        axes[row, 1].hexbin(np.clip(slope, 1e-5, None), v, xscale="log", gridsize=45, cmap="Greens", mincnt=1)
        axes[row, 2].hist(v, bins=60, color="#0072B2", alpha=0.85)
        axes[row, 2].set_ylabel("reaches")
        for c in range(3):
            axes[row, c].grid(alpha=0.3)
    axes[2, 0].set_xlabel("upstream drainage area (km$^2$)")
    axes[2, 1].set_xlabel("channel slope (m/m)")
    axes[2, 2].set_xlabel("parameter value")
    axes[0, 0].set_title("vs drainage area")
    axes[0, 1].set_title("vs slope")
    axes[0, 2].set_title("distribution over all reaches")
    fig.suptitle(f"Parameter relations, {len(ua):,} sub-reaches", y=1.0)
    fig.tight_layout()
    p = run / "parameter_relations.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def convergence(run: Path) -> Path:
    """Median and spread of each parameter per epoch, plus training loss."""
    z = np.load(run / "parameter_epochs.npz")
    hist = pd.read_csv(run / "history.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, key in zip(axes[:3], PARAMS, strict=False):
        a = z[key]
        ep = np.arange(1, len(a) + 1)
        med = np.median(a, axis=1)
        ax.fill_between(
            ep,
            np.percentile(a, 10, axis=1),
            np.percentile(a, 90, axis=1),
            color="#0072B2",
            alpha=0.25,
            label="10-90th pct",
        )
        ax.plot(ep, med, color="k", lw=2, label="median")
        ax.set_xlabel("epoch")
        ax.set_ylabel(LABELS[key])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[3].plot(hist.epoch, hist.loss, color="#D55E00", marker="o", ms=4)
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("mean L1 loss (m$^3$/s)")
    axes[3].set_title("training loss (epoch mean over batches)")
    axes[3].grid(alpha=0.3)
    fig.suptitle("Parameter convergence, CONUS gridded run", y=1.03)
    fig.tight_layout()
    p = run / "convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def metrics_panel(run: Path) -> Path:
    """NSE distribution vs the summed-Q' baseline, by drainage area, and on a map."""
    g = pd.read_csv(run / "gauge_metrics.csv", dtype={"STAID": str})
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for col, lab, c in [("nse_routed", "routed", "#0072B2"), ("nse_summed", "summed Q'", "#D55E00")]:
        s = np.sort(g[col].clip(-1, 1))
        axes[0].plot(s, np.linspace(0, 1, len(s)), label=lab, color=c, lw=1.8)
    axes[0].axvline(0.5, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("NSE (clipped at -1)")
    axes[0].set_ylabel("cumulative fraction of gauges")
    axes[0].set_title(f"NSE CDF, {len(g)} gauges")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    d = g.nse_routed - g.nse_summed
    axes[1].hist(d.clip(-0.5, 0.5), bins=50, color="#009E73", alpha=0.85)
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_xlabel("NSE(routed) - NSE(summed)")
    axes[1].set_ylabel("gauges")
    axes[1].set_title(f"routing beats baseline at {100 * (d > 0).mean():.0f}%")
    axes[1].grid(alpha=0.3)

    bins = [2000, 5000, 10000, 25000, 1e7]
    g["da_bin"] = pd.cut(g.DRAIN_SQKM, bins)
    data = [g.loc[g.da_bin == b, "nse_routed"].clip(-1, 1) for b in g.da_bin.cat.categories]
    axes[2].boxplot(data, tick_labels=["2-5k", "5-10k", "10-25k", ">25k"], showfliers=False)
    axes[2].axhline(0.5, color="grey", ls=":", lw=1)
    axes[2].set_xlabel("drainage area (km$^2$)")
    axes[2].set_ylabel("NSE")
    axes[2].set_title("skill by basin size")
    axes[2].grid(alpha=0.3)

    lon = LON0 + (g.cell % NCOLS) * CELL
    lat = LAT0 + (g.cell // NCOLS) * CELL
    s = axes[3].scatter(lon, lat, c=g.nse_routed.clip(-0.5, 1), s=16, cmap="RdYlBu", vmin=-0.5, vmax=1)
    axes[3].set_title("NSE by gauge")
    axes[3].set_xlabel("longitude")
    axes[3].set_ylabel("latitude")
    axes[3].set_aspect(1.3)
    fig.colorbar(s, ax=axes[3], shrink=0.85)
    fig.tight_layout()
    p = run / "metrics.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def channel_geometry(run: Path) -> Path:
    """Width, depth and width:depth from the learned parameters at a reference flow."""
    import torch

    from ddr.geometry.trapezoidal import compute_trapezoidal_geometry

    ds = xr.open_dataset(run / "kan_parameters.nc")
    len(ds.reach)
    # reference discharge scaled with drainage area, a rough mean-flow surrogate
    q = np.clip(0.01 * ds.uparea_km2.values, 0.1, None)
    geom = compute_trapezoidal_geometry(
        n=torch.tensor(ds["n"].values, dtype=torch.float32),
        p_spatial=torch.tensor(ds["p_spatial"].values, dtype=torch.float32),
        q_spatial=torch.tensor(ds["q_spatial"].values, dtype=torch.float32),
        discharge=torch.tensor(q, dtype=torch.float32),
        slope=torch.tensor(np.clip(ds.slope.values, 1e-6, None), dtype=torch.float32),
        depth_lb=0.01,
        bottom_width_lb=0.1,
    )
    w, d = geom["top_width"].numpy(), geom["depth"].numpy()
    ok = np.isfinite(w) & np.isfinite(d) & (d > 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hexbin(
        ds.uparea_km2.values[ok], w[ok], xscale="log", yscale="log", gridsize=45, cmap="Blues", mincnt=1
    )
    axes[0].set_ylabel("top width (m)")
    axes[1].hexbin(
        ds.uparea_km2.values[ok], d[ok], xscale="log", yscale="log", gridsize=45, cmap="Greens", mincnt=1
    )
    axes[1].set_ylabel("depth (m)")
    axes[2].hexbin(ds.uparea_km2.values[ok], (w / d)[ok], xscale="log", gridsize=45, cmap="RdPu", mincnt=1)
    axes[2].axhspan(10, 60, color="grey", alpha=0.2)
    axes[2].set_ylabel("width : depth")
    axes[2].set_ylim(0, 80)
    for ax in axes:
        ax.set_xlabel("upstream drainage area (km$^2$)")
        ax.grid(alpha=0.3)
    # downstream hydraulic geometry exponents (Leopold & Maddock: b~0.5 for width, f~0.4 for depth)
    lq, lw, ld = np.log10(q[ok]), np.log10(w[ok]), np.log10(d[ok])
    b = np.polyfit(lq, lw, 1)[0]
    f = np.polyfit(lq, ld, 1)[0]
    fig.suptitle(
        f"Channel geometry from learned parameters | downstream exponents: width b = {b:.2f} "
        f"(L&M ~0.5), depth f = {f:.2f} (L&M ~0.4)",
        y=1.04,
    )
    fig.tight_layout()
    p = run / "channel_geometry.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("examples/juniata_gridded/runs/conus_12ep"))
    a = ap.parse_args()
    print(json.loads((a.run / "metrics.json").read_text()))
    for fn in (parameter_maps, parameter_relations, convergence, metrics_panel, channel_geometry):
        print("wrote", fn(a.run))


if __name__ == "__main__":
    main()
