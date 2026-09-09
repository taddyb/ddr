---
name: gridded-routing-example
description: How the DDM30 (ISIMIP 0.5°) gridded routing example is wired through ddr — data products, the Ddm30 dataset class, the negative-coefficient MC regime, and the two-tier acceptance gates. Local mirror of wiki/plan-gridded-juniata.md. Both wiki/ and .claude/ are gitignored in this repo; the tracked home for plans is docs/superpowers/plans/.
---

# Gridded (DDM30) routing example — key concepts

Full plan: `wiki/plan-gridded-juniata.md`. Findings: `wiki/isimip-gridded-routing.md`.

## The sample being mirrored
`examples/juniata/` lives on branch `juniata-sample` (unmerged; merge-base with feat/gridded-engine = 3125dac):
`extract_bundle.py` + `train_and_test.py` (`make_config / train / test / summed_qprime_baseline / main`) +
`tests/examples/test_juniata_bundle.py`. Reference: routed NSE 0.784 / KGE 0.877 vs summed 0.695 / 0.820
(30 CPU epochs, single gauge 01567000, test 1995/10/01–2010/09/30).

## Juniata on DDM30 (verified 2026-09-05)
- 4 cells {138443, 138444, 139163, 138445}; outlet 138445 (global position 29238); edges 29237←29236, 29237←29235, 29238←29237.
- length_m 42,434 / 42,434 / 69,845 / 42,434; slope 0.00386 / 0.002003 / 0.005477 / 0.001202.
- own area ≈ 2,359 km² per cell; DDM30 upstream at outlet 9,419 km² vs gauge 8,657 km² (da_ratio 1.088).
- FLOW_SCALE (ddr's partial-area mechanism, outlet cell Q' only) = (2359.2 − 762.0)/2359.2 = 0.677.
- 230 MERIT COMIDs feed the 4 cells (flowline-midpoint assignment, `data/ddm30/comid_cell.parquet`).
- Attribute store has 23 variables, no `Porosity` → use `WCsat` in the KAN input list.

## Minimal code surface (additive)
- `GeoDataset.DDM30 = "ddm30"` + `Ddm30(Merit)` thin subclass (`src/ddr/geodatazoo/ddm30.py`) overriding only `_load_attributes`.
- `AttributesReader` and `scripts/summed_q_prime.py`: one `elif` each.
- Helpers in `ddr_benchmarks.gridded`: `aggregate_qprime_to_cells`, `cell_centre`, `gauge_table`.
- Scripts: `build_gridded_qprime.py` (→ `ddm30_conus_qprime.ic`, Qr(divide_id=cell, time) daily m³/s from 1980-01-01),
  `build_gridded_gauges.py` (DA-matched snap → CSV with COMID=cell, FLOW_SCALE, DA_VALID; gauge zarr via cell-centre coords).
- Engine, mmc.py, torch_mc.py, merit.py, configs.py untouched.

## MC at 0.5° with dt = 3600 s (expected regime, not a bug)
K = L/c ≈ 5–13 h ≫ dt; Cunge X saturates at 0.5 →
c1 = (dt−K)/(K+dt) < 0, c2 = 1, c3 = (K−dt)/(K+dt), c4 = c1+c2. c1+c2+c3 = 1 so steady state is exact
topological accumulation (Q = I + q'); hotstart `(I−N)Q = q'(t0)` is that steady state. Implicit solve stable
(0.049 % neg_solve on CONUS). Gate every acceptance test on neg_solve < 0.5 %; report c1<0 share (~88 %).

## Acceptance
- Tier A (bundle, 4 cells, 30 epochs): NSE ≥ 0.70, KGE ≥ 0.75, ≥ summed+0.03, neg_solve < 0.5 %.
- Tier B (CONUS gauges > 5,000 km², Juniata held out): Juniata NSE ≥ 0.75; CONUS median ≥ 0.60, routing beats summed at ≥ 75 %.

## Diagram
```
 DDM30 rasters ─► ddm30_adjacency.zarr (order, COO, length_m, slope, lat, lon)
 dhbv2_gages.csv ─► snap_gauges (3×3 DA) ─► ddm30_gages_conus.csv + ddm30_conus_gages_adjacency.zarr
 merit_dhbv2_UH_retrospective.ic ─► comid_cell.parquet ─► Σ per cell ─► ddm30_conus_qprime.ic
 native rasters ─► extractrs ─► ddm30_conus_attributes.nc (COMID = cell id)
 usgs_daily_observations ──────────────────────────────────────────────┐
 ══════════════════════════════════════════ ddr ═══════════════════════╪═════
 Config(geodataset="ddm30") ─► Ddm30(Merit) ─► RoutingDataclass (CSR, length, slope, outflow_idx = gauge cell, flow_scale)
 StreamflowReader (repeat 24) ─► q' (T_h × N)      kan(z-scored attrs) ─► n, q_spatial, p_spatial
 dmc ─► MuskingumCunge (dt 3600, X→0.5, c1<0, lower-tri solve, neg_solve counted) ─► runoff (gauges × T_h)
 tau_trim_and_downsample(9) ─► daily ─► L1 vs USGS[warmup:] ─► backprop ─► KAN
 Tier A: examples/juniata_gridded/train_and_test.py     Tier B: ddr train/test with config/ddm30_*_config.yaml
```
