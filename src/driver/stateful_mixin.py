raise DeprecationWarning()

from abc import ABC
import random

import numpy as np
import torch
from .experiment_base import DriverBase
from typing import Dict, List, Protocol


class Stateful(Protocol):
    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, dict):
        ...


class StatefulAttribute(Stateful):
    def __init__(self, value=None):
        self.value = value

    def load_state_dict(self, dict):
        self.value = dict["value"]

    def state_dict(self) -> dict:
        return {"value": self.value}


class RNGStatefulProxy(Stateful):
    """
    Class which allows rng states to be saved and loaded via the stateful protocol
    """

    def state_dict(self) -> dict:
        return {
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "torch_rng": torch.random.get_rng_state(),
        }

    def load_state_dict(self, dict):
        np.random.set_state(dict["np_rng"])
        torch.random.set_rng_state(dict["torch_rng"])
        random.setstate(dict["py_rng"])


class StatefulComponentsMixin:
    """
    This class allows you to register stateful components and automatically makes the
    overall class stateful by composing the state of its components.
    """

    def register_stateful(self, name, stateful: Stateful):

        if not hasattr(self, "_stateful_components"):
            self._stateful_components = {}

        self._stateful_components[name] = stateful

    def state_dict(self):
        return {k: v.state_dict() for k, v in self._stateful_components.items()}

    def load_state_dict(self, sd):
        if not hasattr(self, "_stateful_components"):
            raise ValueError(
                "Attempting to load state dict, but no stateful components were found"
            )
        for k, d in sd.items():
            assert (
                k in self._stateful_components
            ), f"Attempting to load state dict for component {k}, but {k} was not registered."
            self._stateful_components[k].load_state_dict(d)
