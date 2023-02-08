from abc import abstractmethod
from typing import Any, List, Sequence
from src.typing import SupportsDatasetNameConvention

from ....typing import FeatureExtractionProtocol
from ..configure_optimizer_mixin import ConfigureOptimizersMixin, OptimizerConfig
from pytorch_lightning import LightningModule
import torch
from src.layers.losses.losses import vicreg_loss_func
from torch import nn


class SSLBase(ConfigureOptimizersMixin, LightningModule):
    def __init__(
        self,
        batch_size: int,
        num_epochs: int, 
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):

        # set up optimizing
        self.set_optimizer_configs(opt_cfg)

        # assert isinstance(self.trainer.datamodule, SupportsDatasetNameConvention)\
        #     , "Datamodule should have train_ds"

        self._num_epochs = num_epochs
        self._num_training_steps = None

        self.batch_size = batch_size
        self.inferred_no_centers = 1

        super().__init__()

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            len_ds = len(self.trainer.datamodule.train_ds)  # type:ignore
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    def log_losses(self, losses, state="train"):

        for name, loss in losses.items():

            self.log(
                f"{state}/ssl/{name}",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

   