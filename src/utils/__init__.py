from statistics import mode
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import logging

info = logging.getLogger(__name__).info


def get_fname_prefix(center: str, patient_id: int, loc: str, grade: str):
    """
    template: UVA-0088_RBM_Benign
    """
    return center + "-" + str(patient_id).zfill(4) + "_" + loc + "_" + grade


import torch


def compute_batched(
    large_batch: torch.Tensor,
    fn,
    batch_size=None,
):
    """Infers on a large tensor which does not necessarily fit into GPU
    memory by splitting into chunks along the batch dimension 0"""

    if batch_size is None:
        batch_size = find_largest_batch_size(
            fn,
            large_batch.shape[1:],
            binary_search=False,
        )
        batch_size = min(len(large_batch), batch_size)

    out = []
    batch_idx = 0
    while True:
        beginning = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(large_batch))
        small_batch = large_batch[beginning:end]
        out.append(fn(small_batch))
        if end == len(large_batch):
            break
        batch_idx += 1
    out = torch.concat(out, dim=0)

    return out


from logging import info


def find_largest_batch_size(fn, input_shape, binary_search=False):
    """
    Finds the largest batch size which can be used for the specified function
    without throwing a CUDA out of memory error or other error.

    Args:
        fn: the function to test
        input_shape: the (unbatched) input shape for the function
        binary search: Whether to perform an additional binary search to finetune the batch size.
            If set to true, will find the largest batch size such that batch size + 1 raises an error.
            Otherwise, the function will find the largest batch size such that batch size * 2 raises an error.
    """

    min = 0
    max = None
    test = 1
    verbose = True

    def _try(test):
        if verbose:
            info(f"Testing batch size {test}...")
        try:
            fn(torch.randn(test, *input_shape))
            if verbose:
                info(f"Batch size {test} successful.")
            return True
        except RuntimeError as e:
            if verbose:
                info(f"Batch size {test} failed.")
            return False

    while max is None:
        if _try(test):
            min = test
            test *= 2
        else:
            max = test

    if binary_search:
        while min != max - 1:
            test = int((min + max) / 2)
            if _try(test):
                min = test
            else:
                max = test

    if verbose:
        info(f"Search complete. Largest batch size found {min}.")

    return min


def add_prefix(dict, prefix, separator="/"):
    return {f"{prefix}{separator}{k}": v for k, v in dict.items()}


def map_dict(dict, fn):
    return {k: fn(v) for k, v in fn.items()}


def print_config(config):
    from rich import syntax
    from rich import tree
    import rich

    t = tree.Tree(label="CONFIG")
    for k, v in config.items():
        branch = t.add(k)
        if isinstance(v, dict):
            from omegaconf import OmegaConf

            branch.add(
                syntax.Syntax(OmegaConf.to_yaml(v), "yaml", theme="solarized-light")
            )

        if isinstance(v, DictConfig):
            from omegaconf import OmegaConf

            branch.add(
                syntax.Syntax(OmegaConf.to_yaml(v), "yaml", theme="solarized-light")
            )
        else:
            branch.add(str(v))
    rich.print(t)


def printl():
    print(
        "=========================================================================================="
    )


def verbose(function):
    from functools import wraps

    @wraps(function)
    def wrapped(*args, **kwargs):
        printl()
        info(f"Calling function {function.__name__}")
        out = function(*args, **kwargs)
        if out is not None:
            info(f"{function.__name__} produced object {out}.")
        printl()
        return out

    return wrapped


_triggered = False


def load_dotenv():
    global _triggered
    if _triggered:
        return

    from dotenv import load_dotenv, dotenv_values, find_dotenv
    import os

    info(f"Attempting to load dotenv.")
    if f := find_dotenv(usecwd=True):
        info(f"Found file {f}")
        if load_dotenv(f):
            info(f"Load dotenv successful. Set variables\n{dotenv_values(f)}")
        else:
            info("Load dotenv failed.")
            return
    else:
        info(f"Could not find .env file")
        return

    _triggered = True


from contextlib import contextmanager


@contextmanager
def capture_setattr(self, callback):
    cls = type(self)

    def new_setattr(self, name, value):
        callback(self, name, value)
        object.__setattr__(self, name, value)

    old_setattr = cls.__setattr__
    cls.__setattr__ = new_setattr

    yield
    cls.__setattr__ = old_setattr
