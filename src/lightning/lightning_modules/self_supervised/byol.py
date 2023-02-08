

from dataclasses import dataclass
from src.models.self_supervised.finetuner_module import ExactFineTuner
from .exact_ssl_module import ExactSSLModule
from torch import nn
from timm.utils.model_ema import ModelEmaV2
from typing import List, Dict, Sequence, Any
import torch


class BYOL(ExactSSLModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        ema_decay_initial: float = 0.9,
        ema_decay_final: float = 1,
        proj_hidden_dim: int = 128,
        proj_output_dim: int = 64,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):
        """Implements BYOL

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature: parameter for simclr loss
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
            nn.ReLU(),
            nn.Linear(proj_output_dim, proj_output_dim),
        )

        self.teacher = ModelEmaV2(
            nn.Sequential(self.backbone, self.projector), ema_decay_initial
        )

        self.inferred_no_centers = 1

        self.tracked_loss_names = ["regression_mse_loss"]

        self.losses_all_centers = {}
        self.losses_all_centers["val"] = {k: [] for k in self.tracked_loss_names}
        self.losses_all_centers["test"] = {k: [] for k in self.tracked_loss_names}

        self.ema_decay_initial = ema_decay_initial
        self.ema_decay_final = ema_decay_final

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)

        # # for applications that need feature vector only
        # if not self.training:
        #     return out

        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        # update the teacher model
        self.teacher.update(nn.Sequential(self.backbone, self.projector))

        if self.scheduler is not None:
            lr = self.scheduler_obj.get_last_lr()
            self.log("lr", lr[0], on_step=False, on_epoch=True, prog_bar=False)

        (X1, X2), label = batch

        rep1 = self.backbone(X1)
        proj1 = self.projector(rep1)
        pred = self.predictor(proj1)
        pred = pred / torch.norm(pred, dim=-1, p=2, keepdim=True)

        with torch.no_grad():
            rep2 = self.backbone(X2)
            proj2 = self.projector(rep2)
            target = proj2 / torch.norm(pred, dim=-1, p=2, keepdim=True)

        loss = nn.functional.mse_loss(pred, target)

        loss_dict = {"regression_mse_loss": loss}

        [
            self.log(
                f"train/ssl/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            for loss_name, loss_value in loss_dict.items()
        ]

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.log("ema_decay", self.teacher.decay)
        self.teacher.decay = self.get_momentum_rate()

    def validation_step(
        self, batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ) -> Any:
        """Validation step for VICReg reusing BaseMethod validation step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        (X1, X2), label = batch

        with torch.no_grad():

            rep1 = self.backbone(X1)
            proj1 = self.projector(rep1)
            pred = self.predictor(proj1)
            pred = pred / torch.norm(pred, dim=-1, p=2, keepdim=True)

            rep2 = self.backbone(X2)
            proj2 = self.projector(rep1)
            target = proj2 / torch.norm(pred, dim=-1, p=2, keepdim=True)

            loss = nn.functional.mse_loss(pred, target)

        loss_dict = {"regression_mse_loss": loss}

        self.logging_combined_centers_losses(dataloader_idx, loss_dict)

        return (
            loss_dict["regression_mse_loss"],
            [],
            [],
        )  # these two lists are a workaround to use online_evaluator + metric logger

    def get_momentum_rate(self):
        """
        Computes the decay rate for the current epoch which linearly anneals from
        ema_decay_initial to ema_decay final
        """
        x = self.current_epoch
        x_max = self.max_epochs
        y_0 = self.ema_decay_initial
        y_max = self.ema_decay_final

        a = (y_max - y_0) / x_max
        b = y_0

        y = a * x + b

        return y

    def validation_epoch_end(self, outs: List[Any]):
        kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }

        [
            self.log(
                f"val/ssl/{loss_name}", torch.mean(torch.tensor(loss_list)), **kwargs
            )
            for loss_name, loss_list in self.losses_all_centers["val"].items()
        ]

        [loss_list.clear() for loss_list in self.losses_all_centers["val"].values()]

        [
            self.log(
                f"test/ssl/{loss_name}", torch.mean(torch.tensor(loss_list)), **kwargs
            )
            for loss_name, loss_list in self.losses_all_centers["test"].items()
        ]

        [loss_list.clear() for loss_list in self.losses_all_centers["test"].values()]

    def logging_combined_centers_losses(self, dataloader_idx, loss_dict):

        assert all([key in self.tracked_loss_names for key in loss_dict])

        self.inferred_no_centers = (
            dataloader_idx + 1
            if dataloader_idx + 1 > self.inferred_no_centers
            else self.inferred_no_centers
        )

        val_or_test = (
            "val" if dataloader_idx < self.inferred_no_centers / 2.0 else "test"
        )

        [
            self.losses_all_centers[val_or_test][loss_name].append(loss_value)
            for loss_name, loss_value in loss_dict.items()
        ]
