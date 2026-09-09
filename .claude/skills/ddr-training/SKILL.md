---
name: ddr-training
description: Run DDR's standard training, testing, routing, and summed-Q' baseline through the `ddr` CLI and Hydra configs on MERIT or Lynker Hydrofabric networks. Use when the user wants to train a model, evaluate a checkpoint, route flow with trained weights, produce the unrouted baseline, resume from a checkpoint, or debug a config. Trigger on "train the model", "run training", "test my checkpoint", "route flow", "summed q prime", "which config", "training is slow", "loss is NaN". For the gridded DDM30 path use ddr-gridded instead; for the self-contained single-basin sample see examples/juniata.
---

# Running DDR training

The `ddr` CLI dispatches to Hydra scripts. Every subcommand takes `--config-name`
naming a YAML in `config/` without the extension.

    uv run ddr train --config-name=merit_training_config
    uv run ddr test --config-name=merit_testing_config
    uv run ddr route --config-name=merit_routing_config
    uv run ddr train-and-test --config-name=lynker_train_and_test_config
    uv run ddr summed-q-prime --config-name=summed_merit_q_prime

Always `uv run`. Never `pip` or a bare `python`.

## Pick the config

| Goal | Config |
|---|---|
| MERIT training | `merit_training_config` |
| MERIT evaluation | `merit_testing_config` |
| MERIT forward routing | `merit_routing_config` |
| Lynker Hydrofabric, train then auto-test | `lynker_train_and_test_config` |
| Unrouted baseline | `summed_merit_q_prime` |
| Channel-geometry dataset | `merit_geometry_config` |

`config/templates/` holds starting points, `config/experiments/` active runs.
Generate the field reference with `uv run python scripts/gen_config_docs.py`.

## Override without editing files

Hydra takes dotted overrides, which is the right way to sweep or smoke-test:

    uv run ddr train --config-name=merit_training_config experiment.epochs=1 experiment.batch_size=4
    uv run ddr train --config-name=merit_training_config device=cpu hydra.run.dir=/tmp/smoke

Always smoke-test one epoch with a small batch before a long run.

## What the fields mean

    mode: training              # training | testing | routing
    geodataset: merit           # merit | lynker_hydrofabric
    device: 0                   # GPU index, or cpu

    experiment:
      epochs: 5
      batch_size: 64            # gauges per batch, not timesteps
      rho: 90                   # days per training window
      warmup: 5                 # leading days dropped from loss and metrics
      learning_rate: {1: 0.001, 3: 0.0005}   # keyed by epoch
      checkpoint: null          # path to resume

    kan:
      input_var_names: [...]    # attribute columns, must exist in the attributes file
      learnable_parameters: [n, q_spatial, p_spatial]

`batch_size` counts gauges. The routed network per batch is the union of those
gauges' upstream subgraphs, so memory scales with basin size, not gauge count.

## Outputs

Hydra writes a timestamped run directory containing the resolved config, logs, and
`saved_models/`. Testing writes `model_test.zarr` plus JSON and CSV metrics.
`ddr.validation.Metrics` supplies NSE, KGE, RMSE and the flow-fraction metrics;
`ddr.validation.plots` has hydrograph, CDF and parameter-map helpers, and
`examples/eval/` holds reference notebooks.

## Reading a run

- **`neg_solve` rate** is logged per batch. Around 0.1% is the MERIT baseline. A
  spike means the Muskingum coefficients are going negative, usually from reach
  length or timestep mismatch.
- **Parameter spread** is logged as min/median/max for each learnable parameter. If a
  parameter pins at a bound for most reaches, the bound is wrong or the parameter is
  absorbing error it cannot fit. Widening blindly rarely helps.
- **Loss across epochs is not comparable** when each epoch samples a different random
  window. Judge convergence from the parameter trajectories, not the loss curve.

## Failure modes

| Symptom | Cause |
|---|---|
| `KeyError` on `gage_id` | Gauge IDs lost leading zeros. They are strings; `str.zfill(8)`. |
| Pydantic validation error at startup | A required `data_sources` field is missing. The error names it. |
| NaN loss immediately | Observations are all NaN in the sampled window, or Q' has NaNs. Check coverage first. |
| Very slow on CPU | Set `device` to a GPU index. The sparse triangular solve dominates. |
| Attribute `KeyError` | A name in `kan.input_var_names` is absent from the attributes file. |

## Before claiming a run worked

Compare against `summed-q-prime` on the same gauges and period. Routing that does not
beat the unrouted baseline has not demonstrated anything, whatever its NSE.
