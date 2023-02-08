from .stateful import Stateful


class EarlyStoppingMonitor(Stateful):
    def __init__(self, patience):
        self.patience = patience
        self.strikes = 0

    def update(self, improvement: bool):
        if improvement:
            self.strikes = 0
        else:
            self.strikes += 1

    def should_early_stop(self):
        return self.strikes > self.patience

    def state_dict(self) -> dict:
        return {"strikes": self.strikes}

    def load_state_dict(self, state_dict, strict=True):
        self.strikes = state_dict["strikes"]
