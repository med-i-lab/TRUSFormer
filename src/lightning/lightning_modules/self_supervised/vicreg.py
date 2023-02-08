from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
from omegaconf import MISSING, DictConfig

from .ssl_base import SSLBase

from ....typing import FeatureExtractionProtocol
from ....modeling.registry import create_model
from ....modeling.optimizer_factory import (
    OptimizerConfig,
    SchedulerOptions,
)
from ....modeling.mlp import MLPClassifier
from ..configure_optimizer_mixin import ConfigureMultipleOptimizersMixin
from pytorch_lightning import LightningModule
import torch
from src.layers.losses import vicreg_loss_func
from torch import nn
import logging
import pytorch_lightning as pl
from hydra.utils import instantiate
from src.utils import add_prefix
from torchmetrics.classification.accuracy import Accuracy


logger = logging.getLogger(__name__)


@dataclass
class VICRegConfig:
    _target_: str = "src.lightning.lightning_modules.VICReg"
    _recursive_: bool = False

    backbone_name: Optional[str] = None
    backbone: Optional[Any] = None
    batch_size: int = 32
    proj_output_dim: int = 512
    proj_hidden_dim: int = 512
    sim_loss_weight: float = 25.0
    var_loss_weight: float = 25.0
    cov_loss_weight: float = 1.0
    num_epochs: int = 200
    opt_cfg: OptimizerConfig = OptimizerConfig(
        weight_decay=1e-4, scheduler_options=SchedulerOptions(warmup_epochs=10)
    )


class VICReg(SSLBase):
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        backbone: Optional[DictConfig] = None,
        batch_size: int = 32,
        proj_output_dim: int = 512,
        proj_hidden_dim: int = 512,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        num_epochs: int = 200,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):

        super().__init__(batch_size, num_epochs, opt_cfg)

        if backbone is None:
            if backbone_name is None:
                raise ValueError(f"Must specify either backbone or backbone_name")
            logger.info(f"Instantiating backbone {backbone_name}")
            self.backbone = create_model(backbone_name)
            if not isinstance(self.backbone, FeatureExtractionProtocol):
                raise ValueError(
                    f"VICReg model backbone must support the feature extraction protocol"
                )

        else:
            from hydra.utils import instantiate

            logger.info(f"Instantiating backbone <{backbone._target_}>")
            self.backbone = instantiate(backbone)

        # implement feature extraction protocol
        self.get_features = self.backbone.get_features
        self.features_dim = self.backbone.features_dim

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.batch_size = batch_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # add linear layer to make loading checkpoint with online eval callback work
        self.linear_layer = nn.Linear(self.features_dim, 2)

        self.save_hyperparameters()

    @property
    def learnable_parameters(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
        ]

    def training_step(self, batch: Sequence[Any], batch_idx: int):
        (X1, X2), pos, label, metadata = batch

        assert isinstance(self.backbone, FeatureExtractionProtocol)
        feats1 = self.backbone.get_features(X1)
        feats2 = self.backbone.get_features(X2)
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- vicreg loss -------
        vicreg_loss, all_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        # divide loss by sum of loss weights to compensate -
        # otherwise higher loss weights mean higher effective learning rate.
        vicreg_loss = vicreg_loss / (
            self.var_loss_weight + self.sim_loss_weight + self.cov_loss_weight
        )

        loss_dict = {
            "vicreg_loss": vicreg_loss,
            "sim_loss": all_loss[0],
            "var_loss": all_loss[1],
            "cov_loss": all_loss[2],
        }

        self.log_losses(loss_dict)

        return {"loss": vicreg_loss, "features": feats1}

    def validation_step(
        self, batch: Sequence[Any], batch_idx: int, dataloader_idx=None
    ):
        """Validation step for VICReg reusing BaseMethod validation step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """
        (X1, X2), pos, label, metadata = batch

        assert isinstance(self.backbone, FeatureExtractionProtocol)
        feats1 = self.backbone.get_features(X1)
        feats2 = self.backbone.get_features(X2)
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- vicreg loss -------
        vicreg_loss, all_losses = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss = vicreg_loss / (
            self.var_loss_weight + self.sim_loss_weight + self.cov_loss_weight
        )

        loss_dict = {
            "vicreg_loss": vicreg_loss,
            "sim_loss": all_losses[0],
            "var_loss": all_losses[1],
            "cov_loss": all_losses[2],
        }

        self.log_losses(loss_dict, state="val")

        return {"features": feats1}


@dataclass
class VICRegOptions:
    proj_output_dim: int = 512
    proj_hidden_dim: int = 512
    sim_loss_weight: float = 25.0
    var_loss_weight: float = 25.0
    cov_loss_weight: float = 1.0


@dataclass
class DiscriminatorOptions:

    disc_criterion: Any = None

    disc_opt_cfg: OptimizerConfig = OptimizerConfig()
    gen_opt_cfg: OptimizerConfig = OptimizerConfig()
    disc_to_main_loss_ratio_initial: float = 0.5
    disc_to_main_loss_ratio_final: float = 0.5
    optimization_cycle: tuple[int, int] = 100, 100
    disc_layers: int = 2
    disc_mlp_dropout: float = 0.1
    num_classes: int = 5


@dataclass
class VICRegWithCenterDiscConfig:

    disc_options: DiscriminatorOptions = DiscriminatorOptions()
    backbone: Any = None

    _target_: str = __name__ + ".VICRegWithCenterDisc"
    _recursive_: bool = False
    ssl_opt_config: OptimizerConfig = OptimizerConfig()
    vicreg_options: VICRegOptions = VICRegOptions()


class VICRegWithCenterDisc(pl.LightningModule):
    def __init__(
        self,
        backbone: Any,
        ssl_opt_config: OptimizerConfig,
        disc_options: DiscriminatorOptions,
        vicreg_options: VICRegOptions,
    ):
        super().__init__()
        self.save_hyperparameters()

        logger.info(f"Instantiating backbone {backbone._target_}")
        self.backbone = instantiate(backbone)
        assert isinstance(
            self.backbone, FeatureExtractionProtocol
        ), f"Backbone must support feature extraction protocol."
        self.features_dim = self.backbone.features_dim
        self.get_features = self.backbone.get_features

        logger.info(f"Instantiating projector")
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.features_dim, vicreg_options.proj_hidden_dim),
            nn.BatchNorm1d(vicreg_options.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(vicreg_options.proj_hidden_dim, vicreg_options.proj_hidden_dim),
            nn.BatchNorm1d(vicreg_options.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(vicreg_options.proj_hidden_dim, vicreg_options.proj_output_dim),
        )

        self.disc_options = disc_options
        logger.info(f"Instantiating Discriminator")
        self.discriminator = self._build_mlp(
            self.backbone.features_dim,
            self.disc_options.num_classes,
            self.disc_options.disc_mlp_dropout,
            self.disc_options.disc_layers,
        )

        self.ssl_opt_cfg = ssl_opt_config

        self.discrimination_criterion = instantiate(self.disc_options.disc_criterion)
        self.vicreg_options = vicreg_options

        self.disc_acc_train = Accuracy(num_classes=5, average="macro")
        self.disc_acc_val = Accuracy(num_classes=5, average="macro")

    def configure_optimizers(self):
        num_training_steps_per_epoch = len(
            self.trainer.datamodule.train_dataloader()
        )  # type:ignore
        num_epochs = self.trainer.max_epochs  # type:ignore

        optimizers = []

        from src.modeling.optimizer_factory import configure_optimizers

        optimizers.append(
            configure_optimizers(
                [
                    {"params": self.backbone.parameters(), "name": "backbone"},
                    {"params": self.projector.parameters(), "name": "projector"},
                ],
                self.ssl_opt_cfg,
                num_epochs,
                num_scheduling_steps_per_epoch=num_training_steps_per_epoch,
                return_dict=True,
            )
        )

        optimizers.append(
            configure_optimizers(
                [{"params": self.backbone.parameters(), "name": "backbone_gen"}],
                self.disc_options.gen_opt_cfg,
                num_epochs,
                num_training_steps_per_epoch,
                return_dict=True,
            )
        )

        optimizers.append(
            configure_optimizers(
                [{"params": self.discriminator.parameters(), "name": "discriminator"}],
                self.disc_options.disc_opt_cfg,
                num_epochs,
                num_training_steps_per_epoch,
                return_dict=True,
            )
        )

        return optimizers

    def _build_mlp(self, start_dim, num_classes, dropout, num_hidden_layers):
        inner_dims = [start_dim]
        for i in range(num_hidden_layers):
            dim = inner_dims[i]
            next_dim = int(dim / 2)
            inner_dims.append(next_dim)
        return MLPClassifier(*inner_dims, num_classes=num_classes, dropout=dropout)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (x1, x2), pos, label, metadata = batch

        if optimizer_idx == 0:
            z1 = self.projector(self.backbone(x1))
            z2 = self.projector(self.backbone(x2))
            loss_dict = self.get_vicreg_loss(z1, z2)
            self.log_dict(add_prefix(loss_dict, "vicreg_training"))
            return loss_dict["vicreg_loss"]

        elif optimizer_idx == 1:
            feats: torch.Tensor = self.backbone(x1)
            self._cached_feats = feats.clone().detach()

            center_logits = self.discriminator(feats)
            center_loss = -self.discrimination_criterion(
                center_logits, metadata["center_idx"]
            )
            self.log("vicreg_training/gen_loss", center_loss)
            self.disc_acc_train(center_logits, metadata["center_idx"])
            self.log("vicreg_training/disc_macro_acc", self.disc_acc_train)
            return center_loss

        else:
            center_logits = self.discriminator(self._cached_feats)
            center_loss = self.discrimination_criterion(
                center_logits, metadata["center_idx"]
            )
            self.log("vicreg_training/disc_loss", center_loss)
            return center_loss

    def validation_step(self, batch, batch_idx):
        (x1, x2), pos, label, metadata = batch

        feats1 = self.backbone(x1)
        z1 = self.projector(feats1)
        z2 = self.projector(self.backbone(x2))
        loss_dict = self.get_vicreg_loss(z1, z2)
        self.log_dict(add_prefix(loss_dict, "vicreg_validation"))

        center_logits = self.discriminator(feats1)
        center_loss = self.discrimination_criterion(
            center_logits, metadata["center_idx"]
        )
        self.log("vicreg_validation/gen_loss", -center_loss)
        self.disc_acc_val(center_logits, metadata["center_idx"])
        self.log("vicreg_validation/disc_macro_acc", self.disc_acc_val)

    def get_vicreg_loss(self, z1, z2):
        vicreg_loss, all_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.vicreg_options.sim_loss_weight,
            var_loss_weight=self.vicreg_options.var_loss_weight,
            cov_loss_weight=self.vicreg_options.cov_loss_weight,
        )

        # divide loss by sum of loss weights to compensate -
        # otherwise higher loss weights mean higher effective learning rate.
        vicreg_loss = vicreg_loss / (
            self.vicreg_options.var_loss_weight
            + self.vicreg_options.sim_loss_weight
            + self.vicreg_options.cov_loss_weight
        )
        loss_dict = {
            "vicreg_loss": vicreg_loss,
            "sim_loss": all_loss[0],
            "var_loss": all_loss[1],
            "cov_loss": all_loss[2],
        }

        return loss_dict
