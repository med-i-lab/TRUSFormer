from .stateful import Stateful
from copy import copy


class Counters(Stateful):
    def __init__(self) -> None:
        self._counters = {}

    def add_counter(self, name, initial_value=0):
        self._counters[name] = initial_value

    def add_counters(self, names):
        for name in names:
            self.add_counter(name)

    def increment(self, name, by=1):
        self._counters[name] += by

    def set(self, name, value: int):
        self._counters[name] = value

    @property
    def counters(self):
        return copy(self._counters)

    def state_dict(self) -> dict:
        return self._counters

    def load_state_dict(self, state_dict, strict=True):
        self._counters = state_dict
