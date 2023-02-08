from .stateful import Stateful
from dataclasses import dataclass
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class ScoreImprovementMonitor(Stateful):
    def __init__(self, mode="max", verbose=True):
        self._improvement_callbacks = []
        self._no_improvement_callbacks = []
        self.best_score = None
        assert mode in ["max", "min"]
        self.mode = mode
        self.verbose = verbose

    def _condition(self, new_score, old_best):
        if old_best is None:
            return True
        if self.mode == "max":
            return new_score > old_best
        else:
            return new_score < old_best

    def on_improvement(self, *callbacks):
        for callback in callbacks:
            self._improvement_callbacks.append(callback)

    def on_no_improvement(self, *callbacks):
        for callback in callbacks:
            self._improvement_callbacks.append(callback)

    def update(self, score):
        if self._condition(score, self.best_score):
            if self.verbose:
                logger.info(f"New score {score} exceeds old score {self.best_score}!")
            self.best_score = score
            [callback(score) for callback in self._improvement_callbacks]
        else:
            [callback(score) for callback in self._no_improvement_callbacks]

    def state_dict(self) -> dict:
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict, strict=True):
        self.best_score = state_dict["best_score"]
