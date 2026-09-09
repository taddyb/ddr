<p align="center">
  <img src="docs/images/ddr_logo_5.png" width="50%"/>
</p>

# Distributed Differentiable Routing (DDR)

An implementation of differentiable river routing methods for the NextGen Framework and Hydrofabric

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

### Installation

DDR uses [uv](https://docs.astral.sh/uv/) for dependency management. From the repo root:

```sh
# Full workspace — CPU (includes ddr, ddr-engine, and ddr-benchmarks)
uv sync --all-packages

# Or core DDR only (skip engine and benchmarks)
uv sync --package ddr

# GPU support (adds CuPy for CUDA 12.4)
uv sync --all-packages --group cuda12

# GPU support (adds CuPy for CUDA 13.0)
uv sync --all-packages --group cuda13
```

This installs the `ddr` CLI entry point. Verify with `ddr --help`.

### Quick Start

1. **Copy a config template** from `config/templates/` and customize paths:
   ```sh
   cp config/templates/merit_training.yaml config/my_training.yaml
   # Edit config/my_training.yaml — set data_sources paths to your data
   ```

   Templates use `${oc.env:DDR_DATA_DIR,./data}` so you can set the
   `DDR_DATA_DIR` environment variable or edit the paths directly.

2. **Train a model** using the `ddr` CLI:
   ```sh
   ddr train --config-name=my_training
   ```

3. **Evaluate** the trained model:
   ```sh
   ddr test --config-name=my_testing
   ```

Available subcommands: `train`, `test`, `route`, `train-and-test`, `summed-q-prime`.

You can also call scripts directly if preferred:
```sh
uv run python scripts/train.py --config-name=my_training
```

4. **Generate geometry predictions** from a trained checkpoint:
   ```sh
   uv run python scripts/geometry_predictor.py --config-name=merit_geometry_config
   ```
   This predicts KAN parameters (n, p, q) for all MERIT reaches and computes
   trapezoidal geometry statistics (depth, top width, bottom width, side slope,
   hydraulic radius) using real accumulated discharge over a water year.
   CONUS reaches get full geometry stats; non-CONUS reaches get KAN parameters only.
   Output is a NetCDF file. Override the water year with:
   ```sh
   uv run python scripts/geometry_predictor.py --config-name=merit_geometry_config \
       experiment.start_time=2001/10/01 experiment.end_time=2002/09/30
   ```

### Data Engine

Before training, you need to create adjacency matrices for your domain:
- Requires the `ddr-engine` local package (installed automatically by `uv sync --all-packages`)
- The gauges.csv can be found [here](https://github.com/DeepGroundwater/references/tree/master/mhpi/dHBV2.0UH)

```sh
uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py <PATH/TO/conus_nextgen.gpkg> data/ --gages references/mhpi/dHBV2.0UH/training_gauges.csv
```

This creates two files used for routing:
- `hydrofabric_v2.2_conus_adjacency.zarr` — sparse COO matrix of the full CONUS river network
- `hydrofabric_v2.2_gages_conus_adjacency.zarr` — zarr.Group of sparse COO matrices for networks upstream of USGS gauges

### Gridded (ISIMIP DDM30) routing

DDR also routes on the ISIMIP3b DDM30 0.5-degree drainage-direction grid instead of vector
flowlines. A self-contained sample runs with no external data:

```sh
uv run python examples/juniata_gridded/train_gridded.py --epochs 30
```

See [`examples/juniata_gridded/README.md`](examples/juniata_gridded/README.md) to run it,
[`docs/gridded_data.md`](docs/gridded_data.md) to obtain every input, and
[`docs/gridded_methods.md`](docs/gridded_methods.md) for how it works.

#### How the gauges are selected

Grid resolution decides which gauges are usable, so selection is a filter chain rather
than a choice. A 0.5-degree cell is about 2,350 km², so a basin smaller than one routing
element cannot be represented at all. Starting from `references/gage_info/gages_3000.csv`:

| Step | Gauges | Why |
|---|---|---|
| In `gages_3000.csv` | 3,211 | the standard training set |
| Drainage area > 2,000 km² | 890 | 2,416 gauges are sub-cell and cannot be resolved |
| Snap within area tolerance | 667 | 3×3 drainage-area match, median ratio 1.01 |
| Observation coverage > 90% | 620 | usable over the evaluation period |

Snapping searches the 3×3 cell neighbourhood around the gauge coordinate and keeps the
cell minimising `|log(cell_upstream_area / gauge_drainage_area)|`, taking the better gauge
when two land in the same cell. Assigning each gauge to the nearest cell centre instead
was tested and rejected: it misplaced roughly half of the large gauges, in one case by a
factor of 201 in drainage area, because a gauge near a cell edge often sits closest to a
cell its river never enters.

The area tolerance is `DA_TOLERANCE = (0.7, 1.4)` in `ddr_benchmarks.gridded`, bounding
the ratio of the cell's accumulated area to the gauge's reported drainage area. A gauge
survives only if its best-matching cell drains between 70% and 140% of the reported
basin. The bounds look lopsided but are nearly symmetric in the space the matching works
in, since log(0.7) = −0.357 and log(1.4) = +0.336 sit the same distance either side of a
perfect match. That matters because drainage area is multiplicative: a cell draining twice
too much and one draining half as much are equally wrong, which a linear window would not
capture. The tolerance rejects rather than constrains — the search always returns its
closest match, and the tolerance then discards it if even that is too far off. It is not
binding for large basins, where the median accepted ratio is 1.01, but bites near the
resolution floor, where a single cell is a large fraction of the basin. Override it per
call with `snap_gauges(..., tolerance=...)`.

Raise the threshold with `--min-da-km2` if you want a stricter set; above 5,000 km² gives
337 gauges. The remaining area mismatch is corrected by scaling predictions by the inverse
of the drainage-area ratio, applied identically in training and evaluation.

The lateral inflow and the attribute set both come from the high-resolution CONUS dataset
of Song, Bindas et al. (2025) — the dHBV2.0 differentiable model that provides 40 years of
daily runoff for ~180,000 MERIT unit catchments, and whose attribute tables define the
predictors used by the routing parameterization. The dataset is archived at
[10.5281/zenodo.13774373](https://doi.org/10.5281/zenodo.13774373) and the paper at
[10.1029/2024WR038928](https://doi.org/10.1029/2024WR038928); see the citation below.

### Pre-trained Examples

The `examples/` directory contains pre-trained weights and notebooks for both
MERIT and Lynker Hydrofabric datasets. See [`examples/README.md`](examples/README.md).

### Documentation

Build and serve the docs locally:

```sh
uv run zensical serve
```

#### Previous Work and Citations

- [dHBV2.0](https://github.com/mhpi/dHBV2.0)
```bibtex

@article{song_high-resolution_2025,
	title = {High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning},
	volume = {61},
	copyright = {© 2025 The Author(s).},
	issn = {1944-7973},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1029/2024WR038928},
	doi = {10.1029/2024WR038928},
	abstract = {The National Water Model (NWM) is a key tool for flood forecasting, planning, and water management. Key challenges facing the NWM include calibration and parameter regionalization when confronted with big data. We present two novel versions of high-resolution (∼37 km2) differentiable models (a type of hybrid model): one with implicit, unit-hydrograph-style routing and another with explicit Muskingum-Cunge routing in the river network. The former predicts streamflow at basin outlets whereas the latter presents a discretized product that seamlessly covers rivers in the conterminous United States (CONUS). Both versions use neural networks to provide a multiscale parameterization and process-based equations to provide a structural backbone, which were trained simultaneously (“end-to-end”) on 2,807 basins across the CONUS and evaluated on 4,997 basins. Both versions show great potential to elevate future NWM performance for extensively calibrated as well as ungauged sites: the median daily Nash-Sutcliffe efficiency of all 4,997 basins is improved to around 0.68 from 0.48 of NWM3.0. As they resolve spatial heterogeneity, both versions greatly improved simulations in the western CONUS and also in the Prairie Pothole Region, a long-standing modeling challenge. The Muskingum-Cunge version further improved performance for basins {\textgreater}10,000 km2. Overall, our results show how neural-network-based parameterizations can improve NWM performance for providing operational flood predictions while maintaining interpretability and multivariate outputs. The modeling system supports the Basic Model Interface (BMI), which allows seamless integration with the next-generation NWM. We also provide a CONUS-scale hydrologic data set for further evaluation and use.},
	language = {en},
	number = {4},
	urldate = {2025-04-11},
	journal = {Water Resources Research},
	author = {Song, Yalan and Bindas, Tadd and Shen, Chaopeng and Ji, Haoyu and Knoben, Wouter J. M. and Lonzarich, Leo and Clark, Martyn P. and Liu, Jiangtao and van Werkhoven, Katie and Lamont, Sam and Denno, Matthew and Pan, Ming and Yang, Yuan and Rapp, Jeremy and Kumar, Mukesh and Rahmani, Farshid and Thébault, Cyril and Adkins, Richard and Halgren, James and Patel, Trupesh and Patel, Arpita and Sawadekar, Kamlesh Arun and Lawson, Kathryn},
	year = {2025},
	note = {\_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1029/2024WR038928
tex.ids= song2024highresolution},
	keywords = {*CIROH Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *CIROH subaward A23-0271-S001 from Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *DoE DE-SC0016605 (HyperFacets), *CA DWR 4600014294 (California Department of Water Resources Atmospheric River Program Phase III), *CIROH subaward A23-0252-S002 from Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *DoE NERSC award ERCAP0024296},
	pages = {e2024WR038928},
	file = {Full Text PDF:/Users/taddbindas/Zotero/storage/TWJMPMM9/Song et al. - High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-infor.pdf:application/pdf;Full Text PDF:/Users/taddbindas/Zotero/storage/46ZGUIZ5/Song et al. - 2025 - High-Resolution National-Scale Water Modeling Is E.pdf:application/pdf;Full Text PDF:/Users/taddbindas/Zotero/storage/JX2MD99P/Song et al. - 2025 - High-Resolution National-Scale Water Modeling Is Enhanced by Multiscale Differentiable Physics-Informed Machine Learning.pdf:application/pdf},
}

```

- [dMC-Juniata-HydroDL2](https://github.com/mhpi/dMC-Juniata-hydroDL2)
```bibtex
@article{https://doi.org/10.1029/2023WR035337,
author = {Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and Rahmani, Farshid and Feng, Dapeng and Bian, Yuchen and Lawson, Kathryn and Shen, Chaopeng},
title = {Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning},
journal = {Water Resources Research},
volume = {60},
number = {1},
pages = {e2023WR035337},
keywords = {flood, routing, deep learning, physics-informed machine learning, Manning's roughness},
doi = {https://doi.org/10.1029/2023WR035337},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023WR035337},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023WR035337},
note = {e2023WR035337 2023WR035337},
abstract = {Abstract Recently, rainfall-runoff simulations in small headwater basins have been improved by methodological advances such as deep neural networks (NNs) and hybrid physics-NN models—particularly, a genre called differentiable modeling that intermingles NNs with physics to learn relationships between variables. However, hydrologic routing simulations, necessary for simulating floods in stem rivers downstream of large heterogeneous basins, had not yet benefited from these advances and it was unclear if the routing process could be improved via coupled NNs. We present a novel differentiable routing method (δMC-Juniata-hydroDL2) that mimics the classical Muskingum-Cunge routing model over a river network but embeds an NN to infer parameterizations for Manning's roughness (n) and channel geometries from raw reach-scale attributes like catchment areas and sinuosity. The NN was trained solely on downstream hydrographs. Synthetic experiments show that while the channel geometry parameter was unidentifiable, n can be identified with moderate precision. With real-world data, the trained differentiable routing model produced more accurate long-term routing results for both the training gage and untrained inner gages for larger subbasins (>2,000 km2) than either a machine learning model assuming homogeneity, or simply using the sum of runoff from subbasins. The n parameterization trained on short periods gave high performance in other periods, despite significant errors in runoff inputs. The learned n pattern was consistent with literature expectations, demonstrating the framework's potential for knowledge discovery, but the absolute values can vary depending on training periods. The trained n parameterization can be coupled with traditional models to improve national-scale hydrologic flood simulations.},
year = {2024}
}
```

#### Related works
The internal solver within DDR was inspired by the work from [RAPID River routing](http://rapid-hub.org/docs/RAPID_Parallel_Computing.pdf#page=4.00)
```bibtex
@article{david_river_2011,
	title = {River {Network} {Routing} on the {NHDPlus} {Dataset}},
	volume = {12},
	issn = {1525-7541, 1525-755X},
	url = {https://journals.ametsoc.org/view/journals/hydr/12/5/2011jhm1345_1.xml},
	doi = {10.1175/2011JHM1345.1},
	abstract = {Abstract The mapped rivers and streams of the contiguous United States are available in a geographic information system (GIS) dataset called National Hydrography Dataset Plus (NHDPlus). This hydrographic dataset has about 3 million river and water body reaches along with information on how they are connected into networks. The U.S. Geological Survey (USGS) National Water Information System (NWIS) provides streamflow observations at about 20 thousand gauges located on the NHDPlus river network. A river network model called Routing Application for Parallel Computation of Discharge (RAPID) is developed for the NHDPlus river network whose lateral inflow to the river network is calculated by a land surface model. A matrix-based version of the Muskingum method is developed herein, which RAPID uses to calculate flow and volume of water in all reaches of a river network with many thousands of reaches, including at ungauged locations. Gauges situated across river basins (not only at basin outlets) are used to automatically optimize the Muskingum parameters and to assess river flow computations, hence allowing the diagnosis of runoff computations provided by land surface models. RAPID is applied to the Guadalupe and San Antonio River basins in Texas, where flow wave celerities are estimated at multiple locations using 15-min data and can be reproduced reasonably with RAPID. This river model can be adapted for parallel computing and although the matrix method initially adds a large overhead, river flow results can be obtained faster than with the traditional Muskingum method when using a few processing cores, as demonstrated in a synthetic study using the upper Mississippi River basin.},
	language = {EN},
	number = {5},
	urldate = {2023-08-10},
	journal = {Journal of Hydrometeorology},
	author = {David, Cédric H. and Maidment, David R. and Niu, Guo-Yue and Yang, Zong-Liang and Habets, Florence and Eijkhout, Victor},
	month = oct,
	year = {2011},
	note = {Publisher: American Meteorological Society
Section: Journal of Hydrometeorology},
	pages = {913--934},
}
```
