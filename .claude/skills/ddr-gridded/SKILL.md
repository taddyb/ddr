---
name: ddr-gridded
description: Run DDR on the gridded ISIMIP DDM30 0.5-degree network - build the network and attribute stores, regrid forcing onto cells, split cells into sub-reaches, train on the Juniata or CONUS gauges, validate, and plot. Use when the user mentions DDM30, ISIMIP, gridded routing, grid cells, sub-reaches, the Courant number, cell attributes, or wants to train the gridded model. Trigger on "gridded model", "DDM30", "ISIMIP grid", "0.5 degree", "sub-reach", "regrid the forcing", "gridded attributes". For MERIT or Lynker training use ddr-training instead.
---

# Running the gridded (DDM30) model

The gridded path routes on the ISIMIP3b DDM30 0.5-degree drainage-direction grid
instead of MERIT flowlines. It reuses the routing engine unchanged; only the network,
attributes and forcing differ.

## Fastest path: the self-contained example

Needs nothing external. Inputs ship in `examples/juniata_gridded/data` (1.6 MB).

    uv run python examples/juniata_gridded/train_gridded.py --epochs 30
    uv run python examples/juniata_gridded/eval_plots.py --run examples/juniata_gridded/runs/latest

Reference: routed NSE 0.795 / KGE 0.737 against summed-Q' 0.594 / 0.671, zero negative
solves, about 15 minutes on CPU including the 15-year evaluation. Plots land in the run
directory.

## CONUS training

Needs the full stores. Build them with `docs/gridded_data.md`, then:

    uv run python examples/juniata_gridded/train_conus.py --epochs 45 --batch-size 16
    uv run python examples/juniata_gridded/eval_plots_conus.py --run examples/juniata_gridded/runs/conus_45ep

Reference (620 gauges, 45 epochs, 4.6 h CPU): median NSE 0.542 against summed 0.390,
beating the baseline at 81%, NSE > 0.5 at 336 gauges.

## Building the inputs

`docs/gridded_data.md` is the authoritative walkthrough. In order:

    curl -O https://files.isimip.org/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_{flowdir,basins,slopes}_cru_neva.nc
    uv run python -m ddr_engine.gridded data/ddm30 --path data/ddm30
    uv run scripts/acquire_gee_attributes.py --project <gee-project>   # add --global for the whole grid
    uv run python scripts/build_gridded_attributes.py                  # add --global
    uv run python scripts/build_gridded_qprime.py
    uv run python scripts/build_subdivided_adjacency.py --sizing-flow mean
    uv run python scripts/validate_gridded.py                          # 67 checks, exits non-zero on failure

Long builds get killed if run as a plain background job. Use
`setsid nohup ... > log 2>&1 < /dev/null &` and poll the log.

Never run two Earth Engine jobs at once; parallel jobs trip the rate limit and layers
fail with HTTP 429. The acquisition script is resumable, so just rerun it.

## Facts that matter

**Cell ids** are `row * 720 + col`, row 0 at 55.75°S. Sub-reach node ids are
`cell_id * 1000 + sub_index`, upstream-most first.

**A cell is about 2,350 km².** Basins smaller than that cannot be represented. Of the
3,211 gauges in `gages_3000.csv`, 2,416 are sub-cell; training uses the 620 above
2,000 km² that snap within drainage-area tolerance. Snap gauges with the 3×3
drainage-area match in `ddr_benchmarks.gridded.snap_gauges`; nearest-centre snapping
misplaces about half of large gauges.

**Attributes come from native rasters only.** Never resample basin-averaged values onto
the grid. Aggregate with extractrs coverage-weighted zonal means
(`ddr_engine.gridded.attributes`).

**Forcing is aggregated, not interpolated.** Q' is volumetric per catchment, so it is
split across cells by overlapping area (`ddr_engine.gridded.forcing`). About 40% of
catchments straddle a cell edge, so midpoint assignment misplaces water.

**The negative-coefficient regime is expected.** At 0.5° with dt = 3600 s the Cunge X
saturates at 0.5, so c1 is negative for about 88% of cell-flow states. The implicit
solve absorbs this and mass conservation holds because c1+c2+c3 = 1 and c4 = c1+c2.
Sub-reach splitting removes it but does not improve daily skill: paired across 321
gauges the median ΔNSE is −0.0002. Do not treat negative coefficients as a bug.

**Sub-reach sizing** targets Courant 1. Both Muskingum sign conditions reduce to
2X ≤ C ≤ 2(1−X) with C = c·dt/L, and since X ≤ 0.5, C = 1 always sits inside. The ideal
reach is c·dt, with tolerance ±Q/(T·S·c). Celerity comes from the summed-Q' baseline.
Size for mean flow: sizing for floods is worse overall because most days are near mean
flow. Shorter is not better, since below about 4 km reaches are too short for a 1-hour
step and c3 goes negative instead.

## Known data quirks

- geedim GeoTIFFs carry a trailing `FILL_MASK` band; band 1 is the data.
- The Earth Engine `dir` band is uint8: sinks (−1) arrive as 255, mouths (0) share a
  value with ocean nodata. Both are trace terminals.
- HiHydroSoil is stored as integers ×10,000 and `WCsat` has an unflagged int32-max pixel.
- SoilGrids fills 0 over permanent water where HiHydroSoil returns nodata.
- Hyper-arid cells round to 0 mm/yr precipitation; aridity floors P at 1 mm/yr.
- MERIT Hydro stops at 60°N, excluding 17,238 global cells, so elevation and slope come
  from GMTED2010. Slope must use each raster's own pixel size, not an assumed one.

## Open questions

Roughness initializes at 0.111, the midpoint of its [0.02, 0.2] bounds, which is a
floodplain value rather than a channel one. After 45 epochs medians converge but the
10-90 spread saturates against the bounds. Channel-geometry exponents come out backwards
(width −0.09 against Leopold & Maddock ≈ 0.5, depth 0.75 against ≈ 0.4) because nothing
in an L1-on-daily-flow objective rewards physical channel shape. Both are unresolved;
say so rather than presenting learned parameters as physically calibrated.
