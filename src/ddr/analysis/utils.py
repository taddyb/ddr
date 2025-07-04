from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def save_state(
    epoch: int, mini_batch: int, mlp: nn.Module, optimizer: nn.Module, name: str, saved_model_path: Path
) -> None:
    """Save model state

    Parameters
    ----------
    epoch : int
        The epoch number
    mini_batch : int
        The mini batch number
    mlp : nn.Module
        The MLP model
    optimizer : nn.Module
        The optimizer
    loss_idx_value : int
        The loss index value
    name: str
        The name of the file we're saving
    """
    mlp_state_dict = {key: value.cpu() for key, value in mlp.state_dict().items()}
    cpu_optimizer_state_dict = {}
    for key, value in optimizer.state_dict().items():
        if key == "state":
            cpu_optimizer_state_dict[key] = {}
            for param_key, param_value in value.items():
                cpu_optimizer_state_dict[key][param_key] = {}
                for sub_key, sub_value in param_value.items():
                    if torch.is_tensor(sub_value):
                        cpu_optimizer_state_dict[key][param_key][sub_key] = sub_value.cpu()
                    else:
                        cpu_optimizer_state_dict[key][param_key][sub_key] = sub_value
        elif key == "param_groups":
            cpu_optimizer_state_dict[key] = []
            for param_group in value:
                cpu_param_group = {}
                for param_key, param_value in param_group.items():
                    cpu_param_group[param_key] = param_value
                cpu_optimizer_state_dict[key].append(cpu_param_group)
        else:
            cpu_optimizer_state_dict[key] = value

    state = {
        "model_state_dict": mlp_state_dict,
        "optimizer_state_dict": cpu_optimizer_state_dict,
        "rng_state": torch.get_rng_state(),
        # "loss_idx_value": loss_idx_value,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    if mini_batch == -1:
        state["epoch"] = epoch + 1
        state["mini_batch"] = 0
    else:
        state["epoch"] = epoch
        state["mini_batch"] = mini_batch

    torch.save(
        state,
        saved_model_path / f"_{name}_epoch_{state['epoch']}_mb_{state['mini_batch']}.pt",
    )


def log_eval_metrics(log, nse, rmse, kge):
    """
    Logs evaluation metrics in a formatted and readable way.

    Parameters
    ----------
    log : Logger
        The logger object.
    nse : np.ndarray
        NumPy array of Nash-Sutcliffe Efficiency values.
    rmse : np.ndarray
        NumPy array of Root Mean Squared Error values.
    kge : np.ndarray
        NumPy array of Kling-Gupta Efficiency values.
    """
    log.info("Evaluation Results:")
    log.info("----------------------------------------")  # Separator line
    log.info(f"{'Metric':<10} | {'Mean':>12} | {'Median':>12}")  # Header row
    log.info("----------------------------------------")
    log.info(f"{'NSE':<10} | {np.nanmean(nse):12.4f} | {np.nanmedian(nse[~np.isinf(nse)]):12.4f}")
    log.info(f"{'RMSE':<10} | {np.nanmean(rmse):12.4f} | {np.nanmedian(rmse):12.4f}")
    log.info(f"{'KGE':<10} | {np.nanmean(kge):12.4f} | {np.nanmedian(kge):12.4f}")
    log.info("----------------------------------------")
