import logging
from src.utils.driver.stateful import auto_stateful, StatefulAttribute


@auto_stateful()
class EarlyStoppingMonitor:
    def __init__(self, *callbacks, patience=5):
        self.patience = patience
        self.callbacks = callbacks
        self.reset()
        self.logger = logging.getLogger("Early Stopping")

    def update(self, score):
        if score > self.best_score.value:
            self.strikes.value = 0
            self.logger.info(
                f"Registered score of {score} which is higher than previous best {self.best_score.value}"
            )
            self.best_score.value = score
        else:
            self.strikes.value += 1
            if self.strikes.value >= self.patience:
                for callback in self.callbacks:
                    callback()
                self.logger.info("Early stopping triggered. ")

    def reset(self):
        self.strikes = StatefulAttribute(0)
        self.best_score = StatefulAttribute(-1e9)


@auto_stateful()
class ScoreMonitor:
    def __init__(self, *callbacks, mode="max", verbose=True):
        assert mode in ["min", "max"], f"Only modes `min` and `max` are supported."
        self.mode = mode
        self.callbacks = callbacks

        best = -1e9 if mode == "max" else 1e9
        self.best = StatefulAttribute(best)

        self.verbose = verbose

    def condition(self, old_score, new_score):
        if self.mode == "max":
            return new_score > old_score
        else:
            return new_score < old_score

    def update(self, new_value):
        """Updates the value and returns True if this is the best score"""
        if self.condition(self.best.value, new_value):
            if self.verbose:
                logging.getLogger(str(self.__class__)).info(
                    f"Current score {new_value:.2f} better than previous {self.best.value:.2f}"
                )
            self.best.value = new_value
            for callback in self.callbacks:
                callback(new_value)
