from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from src.typing import SupportsDatasetNameConvention
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
import torch
import torch_optimizer
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from .configure_optimizer_mixin import ConfigureOptimizersMixin, OptimizerConfig


@dataclass
class SharedStepOutput:
    logits: torch.Tensor
    y: torch.Tensor
    loss: torch.Tensor
    metadata: list


class EvaluationBase(ConfigureOptimizersMixin, pl.LightningModule, ABC):
    """
    Makes the model compatible with logging the validation metrics,
    and allows supervised training by only defining shared_step() method and
    trainable_parameters() method
    """

    def __init__(
        self,
        batch_size: int,
        epochs: int = 100,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):

        super().__init__()

        # assert isinstance(self.trainer.datamodule, SupportsDatasetNameConvention)\
        #     , "Datamodule should have train_ds"

        self.batch_size = batch_size
        self.train_acc = Accuracy()

        self.inferred_no_centers = 1

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

        self._num_epochs = epochs

        self.set_optimizer_configs(opt_cfg)

    @abstractmethod
    def shared_step(self, batch) -> SharedStepOutput:
        """This step should be defined by the child class"""

    def training_step(self, batch, batch_idx):

        output = self.shared_step(batch)
        self.train_acc(output.logits.softmax(-1), output.y)

        self.log(
            "train/finetune_loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/finetune_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return output.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        out = self.shared_step(batch)

        if dataloader_idx is not None:
            self.logging_combined_centers_loss(dataloader_idx, out.loss)

        return out.loss, out.logits, out.y, *out.metadata

    def validation_epoch_end(self, outs):
        kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }
        self.log(
            "val/finetune_loss",
            torch.mean(torch.tensor(self.val_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )
        self.log(
            "test/finetune_loss",
            torch.mean(torch.tensor(self.test_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        return out.loss, out.logits, out.y, *out.metadata

    def on_epoch_end(self):
        self.train_acc.reset()

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if not hasattr(self, "_num_training_steps"):
            len_ds = len(self.trainer.datamodule.train_ds)  # type:ignore
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    def logging_combined_centers_loss(self, dataloader_idx, loss):
        """macro loss for centers"""
        self.inferred_no_centers = (
            dataloader_idx + 1
            if dataloader_idx + 1 > self.inferred_no_centers
            else self.inferred_no_centers
        )

        if dataloader_idx < self.inferred_no_centers / 2.0:
            self.val_macroLoss_all_centers.append(loss)
        else:
            self.test_macroLoss_all_centers.append(loss)

    @property
    def num_epochs(self) -> int:
        return self._num_epochs
