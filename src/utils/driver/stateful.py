from abc import ABC
import random

import numpy as np
import torch
from typing import Dict, List, Protocol, Any, runtime_checkable
from contextlib import contextmanager


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state_dict, strict=True) -> Any:
        ...


class StatefulAttribute(Stateful):
    def __init__(self, value=None):
        self.value = value

    def load_state_dict(self, state_dict, strict=True):
        self.value = state_dict["value"]

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

    def load_state_dict(self, state_dict, strict=True):
        np.random.set_state(state_dict["np_rng"])
        torch.random.set_rng_state(state_dict["torch_rng"])
        random.setstate(state_dict["py_rng"])


class WrappedStateful(Stateful):
    def __init__(self, obj):
        self.obj = obj

    def update(self, obj):
        self.obj = obj

    def state_dict(self) -> dict:
        return self.obj.state_dict()

    def load_state_dict(self, state_dict, strict=True) -> Any:
        return self.obj.load_state_dict(state_dict, strict)


class StatefulCollection(Stateful):
    """
    This class allows you to register stateful components and automatically makes the
    overall class stateful by composing the state of its components.
    """

    def __init__(self):
        self._stateful_components = {}

    def register_stateful(self, name, stateful: Stateful):
        self._stateful_components[name] = stateful

    def register_stateful_dict(self, d):
        for k, v in d.items():
            self.register_stateful(k, v)

    def state_dict(self):
        return {k: v.state_dict() for k, v in self._stateful_components.items()}

    def load_state_dict(self, state_dict, strict=True):
        if not hasattr(self, "_stateful_components"):
            raise ValueError(
                "Attempting to load state dict, but no stateful components were found"
            )
        for k, d in state_dict.items():
            assert (
                k in self._stateful_components
            ), f"Attempting to load state dict for component {k}, but {k} was not registered."
            self._stateful_components[k].load_state_dict(d)

    @contextmanager
    def auto_register(self, obj):
        """
        When entered, automatically registers any stateful
        components assigned to the object to the state dictionary.

        Usage:
            > coll = StatefulCollection()
            > a = AnyClass()
            > with coll.auto_register(a)
            >   a.some_stateful = SomeStateful()
            >
            > coll.state_dict()
            > {
            >   'some_stateful' : {...}
            > }

        """
        old_setattr = obj.__class__.__setattr__

        def new_setattr(instance, __name, __value):
            if isinstance(__value, Stateful):
                self.register_stateful(__name, __value)
            old_setattr(instance, __name, __value)

        obj.__class__.__setattr__ = new_setattr

        yield None

        obj.__class__.__setattr__ = old_setattr


def all_state(obj, ignore_keys=[]):
    """
    Returns a state dictionary containing the state dictionaries
    of all stateful components of the object
    """
    d = {}
    for k, v in obj.__dict__.items():
        if isinstance(v, Stateful) and not k in ignore_keys:
            d[k] = v.state_dict()

    return d


def load_all_state(obj, sd, ignore_keys=[]):
    for k, v in obj.__dict__.items():
        if isinstance(v, Stateful) and not k in ignore_keys:
            state = sd[k]
            v.load_state_dict(state)


def model_state(obj):
    """
    Returns a state dictionary containing the state dictionaries
    of all nn.module attributes of the object
    """
    d = {}
    for k, v in obj.__dict__.items():
        if isinstance(v, torch.nn.Module):
            d[k] = v.state_dict()
    return d


from functools import partial


def auto_stateful(ignore_keys=[]):
    """
    Decorator that adds state_dict and load_state_dict methods to a class.
    All of the stateful components of the class are identified and by default
    are included in the state dictionary.

    args:
        ignore_keys (List[str]) - if there are any stateful attributes of the class that
            don't need to be included in the state dictionary, list their attribute names here.
    """

    def state_dict(self):
        return all_state(self, ignore_keys=ignore_keys)

    def load_state_dict(self, state_dict):
        return load_all_state(self, state_dict, ignore_keys=ignore_keys)

    def state_wrapper(cls):
        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        return cls

    return state_wrapper
