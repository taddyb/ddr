"""Validate the gridded (DDM30) network, its attribute store, and MC routing stability.

Runs four families of checks and prints one table; exits non-zero if any fail.

1. flow direction — single downstream, strict lower-triangular (topological) order, no
   self loops, downstream cell is an 8-neighbour (longitude-wrap aware), no edges across
   basin boundaries, accumulated area conserved and increasing downstream
2. units — every attribute inside its documented physical range (``UNITS``)
3. attribute consistency — texture sums, WCpF2 <= WCsat, aridity = PET/P,
   log10_uparea >= own cell area
4. Muskingum-Cunge stability — ddr's own trapezoidal geometry, Cunge X and coefficients
   on every cell at realistic (area-scaled) flows, then a synthetic flood routed through
   the real network with ddr's implicit solve

On a 0.5 degree grid with dt = 3600 s the Cunge X saturates at its 0.5 cap, so
dt << 2KX and c1 is negative for most cells. That is the expected regime, not an
error: the implicit solve, not coefficient positivity, is what has to stay stable,
which is why the routing checks below are the ones that gate.

Usage:
    uv run python scripts/validate_gridded.py [--store <.nc>] [--adjacency <zarr>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr
from ddr_benchmarks.gridded import topo_accumulate
from ddr_engine.gridded.attributes import cell_area_km2
from ddr_engine.gridded.validate import (
    Check,
    check_attribute_consistency,
    check_mass_balance,
    check_network,
    check_units,
)
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

STORE = Path("/mnt/ssd1/data/icechunk/ddm30_conus_attributes.nc")
ADJACENCY = Path("data/ddm30/ddm30_adjacency.zarr")
FLOW_MULTIPLIERS = (0.2, 1.0, 5.0)  # low flow, long-term mean, high flow
RUNOFF_COEFF = 0.3  # fraction of precipitation reaching the channel
DT_S = 3600.0  # ddr's routing timestep
N_MANNING, Q_SPATIAL, P_SPATIAL = 0.05, 0.4, 21.0  # benchmark's fixed parameters


def mean_discharge(mean_p_mm_yr: np.ndarray, uparea_km2: np.ndarray) -> np.ndarray:
    """Rough long-term mean discharge (m3/s) per cell: runoff coefficient x precipitation x area."""
    return RUNOFF_COEFF * mean_p_mm_yr * uparea_km2 * 1000.0 / 31.5576e6


def coefficients(length: np.ndarray, slope: np.ndarray, discharge: np.ndarray) -> tuple[np.ndarray, ...]:
    """Ddr's Muskingum-Cunge coefficients (c1..c4) plus X and Courant number, per cell."""
    from ddr.geometry.trapezoidal import compute_trapezoidal_geometry

    n = len(length)
    length_t = torch.tensor(length, dtype=torch.float32)
    slope_t = torch.tensor(np.clip(slope, 1e-6, None), dtype=torch.float32)
    q_t = torch.tensor(np.clip(discharge, 1e-3, None), dtype=torch.float32)
    geom = compute_trapezoidal_geometry(
        n=torch.full((n,), N_MANNING),
        p_spatial=torch.full((n,), P_SPATIAL),
        q_spatial=torch.full((n,), Q_SPATIAL),
        discharge=q_t,
        slope=slope_t,
        depth_lb=0.01,
        bottom_width_lb=0.1,
    )
    v = torch.clamp(geom["velocity"], min=0.01, max=15.0)
    beta = 5.0 / 3.0 - (4.0 / 3.0) * geom["cross_sectional_area"] * torch.sqrt(
        1.0 + geom["side_slope"] ** 2
    ) / (geom["top_width"] * geom["wetted_perimeter"])
    celerity = v * beta
    x = torch.clamp(0.5 * (1.0 - q_t / (geom["top_width"] * slope_t * celerity * length_t)), min=0.0, max=0.5)
    k = length_t / celerity
    denom = 2.0 * k * (1.0 - x) + DT_S
    c_1 = (DT_S - 2.0 * k * x) / denom
    c_2 = (DT_S + 2.0 * k * x) / denom
    c_3 = (2.0 * k * (1.0 - x) - DT_S) / denom
    c_4 = (2.0 * DT_S) / denom
    courant = celerity * DT_S / length_t
    return tuple(t.numpy() for t in (c_1, c_2, c_3, c_4, x, courant, denom))


def mc_coefficients_check(length: np.ndarray, slope: np.ndarray, discharge: np.ndarray) -> list[Check]:
    """Coefficient sanity over a low/mean/high flow sweep, plus the regime diagnostic."""
    n_bad_denom = n_bad_coef = n_neg = total = 0
    xs, cs = [], []
    for mult in FLOW_MULTIPLIERS:
        c_1, c_2, c_3, _, x, courant, denom = coefficients(length, slope, discharge * mult)
        n_bad_denom += int((denom <= 0).sum())
        n_bad_coef += int((~np.isfinite(c_1 + c_2 + c_3)).sum())
        n_neg += int((c_1 < 0).sum())
        total += len(length)
        xs.append(x)
        cs.append(courant)
    return [
        Check("MC denominators positive", n_bad_denom == 0, f"{n_bad_denom} non-positive denominators"),
        Check("MC coefficients finite", n_bad_coef == 0, f"{n_bad_coef} non-finite coefficients"),
        Check(
            "MC regime (informational)",
            True,
            f"median X {np.median(np.concatenate(xs)):.3f}, median Courant "
            f"{np.median(np.concatenate(cs)):.3f}, c1<0 in {100 * n_neg / total:.0f}% of states "
            "— expected at 0.5 deg with dt=3600 s; the implicit solve is what must stay stable",
        ),
    ]


def routing_stability(
    rows: np.ndarray,
    cols: np.ndarray,
    length: np.ndarray,
    slope: np.ndarray,
    discharge: np.ndarray,
    n_steps: int = 72,
) -> list[Check]:
    """Route a synthetic flood through the real network with ddr's implicit solve.

    Mirrors ``mmc.route_timestep``: A = I - c1*Adj (lower triangular in topological
    order), b = c2*(Adj @ Q_t) + c3*Q_t + c4*q'. A 12-hour pulse then recession.
    """
    n = len(length)
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    c_1, c_2, c_3, c_4, *_ = coefficients(length, slope, discharge)
    a = (sparse.eye(n, format="csr") - sparse.diags(c_1) @ adj).tocsr()

    has_dn = np.zeros(n, dtype=bool)
    has_dn[cols] = True
    terminals = ~has_dn

    n_up = np.asarray(adj.sum(axis=1)).ravel()
    q_prime = np.where(n_up > 0, discharge / (n_up + 1.0), discharge)  # local inflow only
    q_t = np.zeros(n)
    n_neg = n_bad = 0
    peak = 0.0
    for step in range(n_steps):
        pulse = q_prime * (3.0 if step < 12 else 1.0)
        b = c_2 * (adj @ q_t) + c_3 * q_t + c_4 * pulse
        q_t = spsolve_triangular(a, b, lower=True)
        n_bad += int((~np.isfinite(q_t)).sum())
        n_neg += int((q_t < 0).sum())
        peak = max(peak, float(np.nanmax(q_t)))
    total = n * n_steps
    outlet, lateral = float(q_t[terminals].sum()), float(q_prime.sum())
    rel = abs(outlet - lateral) / lateral
    return [
        Check("routing solve finite", n_bad == 0, f"{n_bad} non-finite discharges over {n_steps} h"),
        Check(
            "routing discharge non-negative",
            100.0 * n_neg / total < 0.5,
            f"{n_neg}/{total} cell-hours ({100.0 * n_neg / total:.3f}%)",
        ),
        Check(
            "routing mass balance at outlets",
            rel < 0.05,
            f"outlets {outlet:.5g} vs lateral inflow {lateral:.5g} m3/s ({100 * rel:.2f}%)",
        ),
        Check("routing peak bounded", peak < 1e3 * lateral, f"peak {peak:.4g} m3/s"),
    ]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--adjacency", type=Path, default=ADJACENCY)
    args = parser.parse_args()

    g = zarr.open_group(args.adjacency, mode="r")
    order, lat, lon = g["order"][:], g["lat"][:], g["lon"][:]
    rows, cols = g["indices_0"][:], g["indices_1"][:]
    dn = np.full(len(order), -1, dtype=np.int64)
    dn[cols] = rows

    df = xr.open_dataset(args.store).to_dataframe()
    pos = {cell: i for i, cell in enumerate(order)}
    idx = np.array([pos[c] for c in df.index])
    area = cell_area_km2(order)

    # subnetwork restricted to the store's cells, renumbered but still topological
    local = {int(p): i for i, p in enumerate(idx)}
    keep = np.array([r in local and c in local for r, c in zip(rows, cols, strict=True)])
    sub_rows = np.array([local[int(r)] for r in rows[keep]], dtype=np.int64)
    sub_cols = np.array([local[int(c)] for c in cols[keep]], dtype=np.int64)
    q_mean = mean_discharge(df["meanP"].to_numpy(), 10 ** df["log10_uparea"].to_numpy())
    length, slope = g["length_m"][:][idx], g["slope"][:][idx]

    sections = {
        "flow direction": check_network(rows, cols, lat, lon, g["basin"][:], ncols=720)
        + check_mass_balance(area, topo_accumulate(area, dn), dn),
        "units": check_units(df),
        "attribute consistency": check_attribute_consistency(df),
        "MC coefficients": mc_coefficients_check(length, slope, q_mean),
        "routing stability": routing_stability(sub_rows, sub_cols, length, slope, q_mean),
    }

    failed = 0
    for section, checks in sections.items():
        print(f"\n{section}")
        for c in checks:
            failed += not c.passed
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:34s} {c.detail}")
    print(f"\n{sum(len(v) for v in sections.values())} checks, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
