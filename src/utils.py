"""
Utility functions for experiments.

Some code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/utils.py
"""

import numbers
import os
import psutil
import yaml
import socket
import logging
import numpy as np

import torch


logging.basicConfig(level=logging.INFO)

def is_child_process():
    # is_child = torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() > 0
    is_child = int(os.environ.get('NODE_RANK', 0)) + int(os.environ.get('LOCAL_RANK', 0)) > 0
    # print(f"GLOBAL_RANK: {os.environ.get('RANK')}")
    # print(f"NODE_RANK: {os.environ.get('NODE_RANK')}")
    # print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    # print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    # print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    # print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    # print(f"MASTER_ADDRESS: {os.environ.get('MASTER_ADDRESS')}")    
    return is_child
    

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("GLOBAL_RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK") # NODE_RANK RANK
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def get_system_ram_usage():
    """ Returns the system RAM usage as a percentage. """
    mem = psutil.virtual_memory()
    
    # Convert bytes to Gigabytes (or Megabytes if preferred)
    total_gb = round(mem.total / (1024**3), 2)
    used_gb = round(mem.used / (1024**3), 2)
    available_gb = round(mem.available / (1024**3), 2)
    percent_used = mem.percent

    print(f"Total RAM: {total_gb} GB")
    print(f"Used RAM: {used_gb} GB")
    print(f"Available RAM: {available_gb} GB")
    print(f"Percentage Used: {percent_used}%")
    
    return mem    
    if os.name == 'nt':
        return psutil.virtual_memory().percent
    else:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
        mem_total = float(meminfo[0].split()[1])
        mem_free = float(meminfo[1].split()[1])
        mem_available = float(meminfo[2].split()[1])
        mem_used = mem_total - mem_available
        mem_usage_percent = (mem_used / mem_total) * 100.0
        return mem_usage_percent


def has_wandb_connectivity(host="api.wandb.ai", port=443, timeout=2.0):
    try:
        s = socket.create_connection((host, port), timeout)
        s.close()
        return True
    except OSError:
        return False


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    logger = logging.getLogger(name)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def is_scalar(v):
    # if v is None or v is False:
        # return False
    if isinstance(v, numbers.Number):
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, str):
        return False
    if isinstance(v, (list, tuple, dict, set)):
        return False
    if np.isscalar(v):
        return True
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return True
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return True
    if isinstance(v, (int, float, complex)):
        return True
    return False


def to_yaml(in_data, yaml_path=None):

    def _to_yaml(in_data):
        if isinstance(in_data, (list, tuple)):
            return in_data
        elif isinstance(in_data, np.ndarray):
            return in_data.tolist()
        elif isinstance(in_data, np.generic):
            return in_data.item()
        elif isinstance(in_data, torch.Tensor):
            return in_data.cpu().numpy().tolist()
        elif isinstance(in_data, dict):
            return {k: _to_yaml(v) for k, v in in_data.items()}
        else:
            return in_data

    yaml_data = _to_yaml(in_data)
    if yaml_path is not None:
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)
    return yaml_data


def concat_dicts_outputs(epoch_outputs):
    """ Concatenate list of batch outputs to single batched outputs."""
    # initialize a new dictionary to hold all outputs
    dataset_feats, dataset_preds = {}, {}
    for batch_feats, batch_preds in epoch_outputs:
        for key, value in batch_feats.items():
            if key not in dataset_feats:
                dataset_feats[key] = []
            dataset_feats[key].append(value)
        for key, value in batch_preds.items():
            if key not in dataset_preds:
                dataset_preds[key] = []
            dataset_preds[key].append(value)

    # concatenate lists into single tensors if they are tensors or numpy arrays
    def _concat_lists_of_dict(dict_of_lists):
        for key in dict_of_lists:
            if isinstance(dict_of_lists[key][0], torch.Tensor):
                dict_of_lists[key] = torch.cat(dict_of_lists[key], dim=0)
            elif isinstance(dict_of_lists[key][0], np.ndarray):
                dict_of_lists[key] = np.concatenate(dict_of_lists[key], axis=0)
            elif isinstance(dict_of_lists[key][0], (list, tuple)):
                dict_of_lists[key] = sum(dict_of_lists[key], start=[])
            else:
                continue
        return dict_of_lists

    dataset_feats = _concat_lists_of_dict(dataset_feats)
    dataset_preds = _concat_lists_of_dict(dataset_preds)

    return dataset_feats, dataset_preds


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def peekaboo_tensors(vars_dict, prefix='', recursive=False):
    """ Display variables (a dict of tensors) in a formatted table. """
    if vars_dict is None or not hasattr(vars_dict, 'items'):
        return
    
    if prefix:
        print(f"\n{prefix} Variables:")
    else:
        print("\nVariables:")

    print(f"{'Key':<23} {'Type':<20} {'Shape':<25} {'Dtype':<15} {'Device':<15} {'Requires Grad':<15}")
    print("-" * 110)
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            vtype = "Tensor"
            shape = str(tuple(v.shape))
            dtype = str(v.dtype)
            device = str(v.device)
            requires_grad = str(v.requires_grad)
        else:
            vtype = type(v).__name__
            shape = str(tuple(v.shape)) if hasattr(v, 'shape') else f'({len(v)})' if isinstance(v, (list, tuple)) else "-"
            dtype = str(v.dtype) if hasattr(v, 'dtype') else "-"
            device = "-"
            requires_grad = "-"
        print(f"{k:<23} {vtype:<20} {shape:<25} {dtype:<15} {device:<15} {requires_grad:<15}")
    print("-" * 110)

    if recursive:
        for k, v in vars_dict.items():
            if isinstance(v, dict):
                peekaboo_tensors(v, prefix=f"{prefix}['{k}']", recursive=True)