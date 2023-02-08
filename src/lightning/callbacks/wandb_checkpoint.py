from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from wandb.util import generate_id
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from warnings import warn

raise NotImplementedError()

warn(f"You are using the WandbCheckpoint callback but it is untested.")


class WandbLoggerCheckpoint(Callback):
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:

        if isinstance(trainer.logger, WandbLogger):
            self.logger = trainer.logger
        else:
            self.logger = None

    # def state_dict(self) -> Dict[str, Any]:
    #    out = {}
    #    if self.logger is not None:
    #        out["wandb_id"] = self.logger._experiment.id


#
#    return out

# def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#
#    if self.logger is not None:
#        if id := state_dict.get("wandb_id") is not None:
#
#            _wandb_init = self.logger._wandb_init
#            _wandb_init["id"] = id
#            self.logger._experiment = wandb.init(**_wandb_init, resume=True)
