import pytorch_lightning as pl
from torch import nn
from src.layers.losses import vicreg_loss_func
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class VICReg(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        proj_output_dim: int = 512,
        proj_hidden_dim: int = 512,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        lr: float = 0.0001,
    ):
        super().__init__()

        self.backbone = backbone
        self.proj_output_dim = proj_output_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.lr = lr

        assert hasattr(
            self.backbone, "features_dim"
        ), "Backbone should have features_dim attribute"
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return x

    def training_step(self, batch, batch_idx):
        (X1, X2) = batch[0]

        z1 = self(X1)
        z2 = self(X2)

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

        self.log_losses(loss_dict, batch_size=X1.shape[0])

        return loss_dict["vicreg_loss"]

    def validation_step(self, batch, batch_idx):
        (X1, X2) = batch[0]

        z1 = self(X1)
        z2 = self(X2)

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

        self.log_losses(loss_dict, batch_size=X1.shape[0], state="val")

    def log_losses(self, losses, batch_size=None, state="train"):
        for name, loss in losses.items():
            self.log(
                f"{state}/ssl/{name}",
                loss.item(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
