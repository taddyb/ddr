# Gridded (ISIMIP DDM30) Juniata sample

The gridded counterpart of [`examples/juniata`](../juniata/README.md). Same basin and
same gauge, but the river network is the ISIMIP DDM30 0.5° drainage-direction grid
instead of MERIT flowlines: four grid cells, split into 27 sub-reaches of about 7 km.

Everything needed is in `data/` (1.6 MB). No HPC, S3, or external stores.

## Quickstart

    git clone <repo> && cd ddr
    uv sync --all-packages
    uv run python examples/juniata_gridded/train_gridded.py --epochs 30
    uv run python examples/juniata_gridded/eval_plots.py --run examples/juniata_gridded/runs/latest

Training is 30 optimizer steps, one random 90-day window per epoch on a single gauge.
It demonstrates that the gridded network trains and routes, not a converged model.

Reference run (CPU, 30 epochs, ~15 min including the 15-year evaluation):

| | routed | summed-Q′ baseline |
|---|---|---|
| NSE | 0.795 | 0.594 |
| KGE | 0.737 | 0.671 |

Negative solves: 0.0000%. The sub-reaches are sized so the Courant number is near 1,
where every Muskingum coefficient stays non-negative; see "Why sub-reaches" below.

## What's in the bundle

| File | Contents |
|---|---|
| `juniata_subreach_adjacency.zarr` | binsparse COO subgraph of 27 sub-reaches with `length_m`, `slope`, `lat`, `lon`, `parent_cell`, reindexed to 0..26 |
| `ddm30_conus_attributes.nc` | 25 cell attributes for all 5,198 CONUS cells, so normalisation statistics match a CONUS run rather than being computed from four cells |
| `juniata_qprime.ic` | icechunk, `Qr(divide_id, time)` daily m³/s for the four cells, 1980–2010 (dHBV2 UH retrospective, area-weighted onto cells) |
| `juniata_obs.ic` | icechunk, `streamflow(gage_id, time)` daily m³/s, USGS 01567000 |
| `juniata_gage.csv` | one-row gauge metadata including the snapped cell and drainage-area ratio |

Node ids encode their grid cell: `node_id = cell_id * 1000 + sub_index`, upstream-most
sub-reach first. Cell ids are `row * 720 + col` with row 0 at 55.75°S.

## What the model learns

A KAN reads ten attributes per sub-reach and predicts Manning's `n` and the
Leopold-Maddock `p` and `q` that set channel width and depth. Sub-reaches inherit
their parent cell's attributes, except `log10_uparea`, which is recomputed per reach
from accumulated area so the network sees within-cell variation.

With only four cells the KAN has four distinct attribute rows, so the learned field is
nearly uniform. That is a property of the sample, not a bug. For a real parameter field
use the CONUS trainer:

    uv run python examples/juniata_gridded/train_conus.py --epochs 45
    uv run python examples/juniata_gridded/eval_plots_conus.py --run examples/juniata_gridded/runs/conus_45ep

That one needs the full gridded stores; see [docs/gridded_data.md](../../docs/gridded_data.md).

## Why sub-reaches

A 0.5° cell is 33–79 km long. With the router's 1-hour timestep the wave travel time
K = L/c is 5–13 h, so the Cunge weighting X saturates at its 0.5 cap and the Muskingum
coefficient `c1` goes negative for about 88% of cell-flow states. Both sign conditions
reduce to one band on the Courant number C = c·dt/L:

    2X ≤ C ≤ 2(1−X)

Since X can never exceed 0.5, C = 1 always sits inside it. Sizing each reach at c·dt
puts it there. Sub-reach lengths come from the summed-Q′ baseline, which supplies the
discharge that celerity needs.

Splitting eliminates negative solves but does **not** improve daily skill: paired
across 321 CONUS gauges the median ΔNSE is −0.0002. The implicit triangular solve
already absorbs negative coefficients, and mass conservation is guaranteed by
`c1+c2+c3 = 1` and `c4 = c1+c2`. The cell network is available for cheap iteration.

## Regenerating the bundle

Maintainer only, needs the full stores:

    uv run python examples/juniata_gridded/extract_bundle.py --out examples/juniata_gridded/data
