from datetime import datetime
from distutils.log import error
import os
import pytorch_lightning as pl
from typing import Literal
from warnings import warn


class CheckpointMetadataSaver(pl.Callback):
    """A callback to save the checkpoint information and path to the checkpoint
    in a nicely structured yaml file"""

    def __init__(
        self, name, fpath, error_strategy: Literal["raise", "warn", "ignore"] = "warn"
    ) -> None:
        self.name = name
        self.fpath = fpath
        self.error_strategy = error_strategy
        if not os.path.isfile(fpath):
            try:
                open(self.fpath, "a").close()
            except:
                self._error(f"file at path {fpath} not found and could not be created.")

    def _error(self, msg):
        if self.error_strategy == "raise":
            raise RuntimeError(msg)
        elif self.error_strategy == "ignore":
            pass
        else:
            warn(msg)

    def _update_file(self, new_info: dict):
        """
        Parses the target checkpoint file to the correct format, which should be a list of dictionaries
        """
        from omegaconf import OmegaConf

        try:
            obj = OmegaConf.load(self.fpath)
        except IOError:
            self._error(
                f"Cannot parse file {self.fpath}. Is it a correctly formatted yaml file?"
            )
            return

        if not obj:
            obj = []
        else:
            obj = OmegaConf.to_object(obj)
        obj.append(new_info)  # type:ignore
        OmegaConf.save(obj, self.fpath)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if trainer.checkpoint_callback is None:
            self._error(
                f"CheckpointSaver requires Checkpoint Callback be available, but trainer.checkpoint_callback is None."
            )
            return

        info = {}
        info["name"] = self.name
        info["time"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        info["best_score"] = {
            "name": trainer.checkpoint_callback.monitor,
            "value": trainer.checkpoint_callback.best_model_score.item(),  # type:ignore
        }
        info["best_checkpoint"] = trainer.checkpoint_callback.best_model_path

        info["last_checkpoint"] = trainer.checkpoint_callback.last_model_path
        info["_target_"] = str(pl_module.__class__).split("'")[1]

        self._update_file(info)
