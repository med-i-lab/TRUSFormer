from dataclasses import dataclass
from typing import Optional, Tuple
from ..evaluation_base import EvaluationBase, SharedStepOutput
from ..configure_optimizer_mixin import OptimizerConfig
import torch
from torch.nn import functional as F
from ....modeling import create_model
from ..configure_optimizer_mixin import OptimizerConfig
import pytorch_lightning as pl
from ....modeling.mlp import MLPClassifier
from ....typing import FeatureExtractionProtocol
from hydra.utils import instantiate
from ....modeling.optimizer_factory import configure_optimizers
from ....utils.metrics import (
    OutputCollector,
    compute_center_and_macro_metrics,
    apply_metrics_to_patch_and_core_output,
    add_prefix,
)
from contextlib import nullcontext
from typing import Any
from torchmetrics.functional import accuracy, auroc


@dataclass
class SupervisedModelConfig:

    _target_: str = __name__ + ".SupervisedModel"

    backbone_name: str = "resnet10"
    batch_size: int = 32
    epochs: int = 100
    loss_weights: Optional[Tuple[float, float]] = None
    opt_cfg: OptimizerConfig = OptimizerConfig()


class SupervisedModel(EvaluationBase):
    def __init__(
        self,
        backbone_name: str,
        batch_size: int,
        epochs: int = 100,
        loss_weights=None,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(batch_size=batch_size, epochs=epochs, opt_cfg=opt_cfg)
        self.loss_weights = loss_weights
        self.save_hyperparameters()
        self.backbone = create_model(backbone_name)

    def shared_step(self, batch) -> SharedStepOutput:
        X, pos, y, metadata = batch

        logits = self.backbone(X)

        loss = F.cross_entropy(logits, y, weight=self.loss_weights)

        return SharedStepOutput(logits, y, loss, [metadata])

    @property
    def learnable_parameters(self):
        return [{"params": self.backbone.parameters()}]


CENTER_LOSS_WEIGHTS = {
    "UVA": 2.8295503211991435,
    "PCC": 4.131957473420888,
    "CRCEO": 4.497617426820967,
    "JH": 10.72564935064935,
    "PMCC": 11.236394557823129,
}
from src.lightning.datamodules.exact_datamodule import CENTERS

CENTER_LOSS_WEIGHTS = tuple([CENTER_LOSS_WEIGHTS[center] for center in CENTERS])


@dataclass
class DiscriminatorOptions:
    disc_opt_cfg: OptimizerConfig = OptimizerConfig()
    center_loss_weights: Optional[Tuple] = CENTER_LOSS_WEIGHTS
    disc_to_main_loss_ratio_initial: float = 0.5
    disc_to_main_loss_ratio_final: float = 0.5
    optimization_cycle: Tuple[int, int] = 100, 100
    num_centers: int = 5


@dataclass
class SupervisedPatchModelWithCenterDiscConfig:

    backbone: Any
    opt_cfg: OptimizerConfig = OptimizerConfig()
    mlp_hidden_layers: int = 2
    mlp_dropout: float = 0.1
    cancer_loss_weights: Optional[Any] = None
    disc_options: DiscriminatorOptions = DiscriminatorOptions()

    _target_: str = __name__ + ".SupervisedPatchModelWithCenterDiscriminator"
    # _target_: str =  "src.lightning.lightning_modules.supervised.supervised_patch_model" + \
    # ".SupervisedPatchModelWithCenterDiscriminator"
    _recursive_: bool = False


class SupervisedPatchModelWithCenterDiscriminator(pl.LightningModule):
    def __init__(
        self,
        backbone,
        opt_cfg,
        mlp_hidden_layers=2,
        mlp_dropout=0.1,
        cancer_loss_weights=None,
        disc_options: DiscriminatorOptions = DiscriminatorOptions(),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = instantiate(backbone)
        assert isinstance(self.backbone, FeatureExtractionProtocol)
        self.features_dim = self.backbone.features_dim
        self.get_features = self.backbone.get_features
        self.cancer_clf = self._build_mlp(
            self.features_dim, 2, mlp_dropout, mlp_hidden_layers
        )
        self.center_clf = self._build_mlp(
            self.features_dim, disc_options.num_centers, mlp_dropout, mlp_hidden_layers
        )
        self.disc_options = disc_options
        self.opt_cfg = opt_cfg
        self.disc_opt_cfg = disc_options.disc_opt_cfg
        self.optimization_cycle = disc_options.optimization_cycle
        if self.disc_options.center_loss_weights is not None:
            self.center_loss_weights = torch.tensor(disc_options.center_loss_weights)
        else:
            self.center_loss_weights = None
        if cancer_loss_weights is not None:
            self.cancer_loss_weights = cancer_loss_weights
        else:
            self.cancer_loss_weights = None
        self.num_centers = disc_options.num_centers

        self.train_output_collector = OutputCollector()
        self.val_output_collector = OutputCollector()
        self.test_output_collector = OutputCollector()

    def on_train_epoch_start(self) -> None:
        self.train_output_collector.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_output_collector.reset()
        self.test_output_collector.reset()

    def _build_mlp(self, start_dim, num_classes, dropout, num_hidden_layers):
        inner_dims = [start_dim]
        for i in range(num_hidden_layers):
            dim = inner_dims[i]
            next_dim = int(dim / 2)
            inner_dims.append(next_dim)
        return MLPClassifier(*inner_dims, num_classes=num_classes, dropout=dropout)

    def get_disc_loss_ratio(self):
        """
        Computes the current discriminator loss weight based on linear annealing from initial loss
        weight to final loss weight
        """
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs

        x = epoch / max_epochs

        y_0 = self.disc_options.disc_to_main_loss_ratio_initial
        y_f = self.disc_options.disc_to_main_loss_ratio_final

        y = (y_f - y_0) * (x) + y_0

        return y

    def configure_optimizers(self):
        assert self.trainer is not None
        total_num_training_steps_per_epoch = len(
            self.trainer.datamodule.train_dataloader()
        )
        total_num_cycles_per_epoch = int(
            total_num_training_steps_per_epoch / sum(self.optimization_cycle)
        )
        total_clf_steps_per_epoch = (
            int(total_num_cycles_per_epoch) * self.optimization_cycle[0]
        )
        total_disc_steps_per_epoch = (
            int(total_num_cycles_per_epoch) * self.optimization_cycle[1]
        )
        total_epochs = self.trainer.max_epochs
        opt1 = configure_optimizers(
            [
                {"params": self.backbone.parameters()},
                {"params": self.cancer_clf.parameters()},
            ],
            self.opt_cfg,
            total_epochs,
            total_clf_steps_per_epoch,
            return_dict=True,
        )
        opt1["frequency"] = self.optimization_cycle[0]
        if opt1.get("lr_scheduler"):
            opt1["lr_scheduler"]["name"] = "Main_opt"
        opt2 = configure_optimizers(
            [{"params": self.center_clf.parameters()}],
            self.disc_opt_cfg,
            total_epochs,
            total_disc_steps_per_epoch,
            return_dict=True,
        )
        opt2["frequency"] = self.optimization_cycle[1]
        if opt2.get("lr_scheduler"):
            opt2["lr_scheduler"]["name"] = "Disc_opt"
        return opt1, opt2

    def shared_step(self, batch, opt_idx):
        patch, pos, cancer_label, metadata = batch
        center_label = metadata["center_idx"]
        with nullcontext() if opt_idx == 0 else torch.no_grad():
            features = self.get_features(patch)
            cancer_logits = self.cancer_clf(features)
        with nullcontext() if opt_idx == 1 else torch.no_grad():
            center_logits = self.center_clf(features)

        return cancer_logits, cancer_label, center_logits, center_label, metadata

    def training_step(self, batch, batch_idx, optimizer_idx):
        (
            cancer_logits,
            cancer_label,
            center_logits,
            center_label,
            metadata,
        ) = self.shared_step(batch, optimizer_idx)

        self.train_output_collector.collect_batch(
            {
                "cancer_logits": cancer_logits,
                "cancer_labels": cancer_label,
                "center_logits": center_logits,
                "center_labels": center_label,
                **metadata,
            }
        )

        weight = self.cancer_loss_weights
        if weight is not None:
            weight = weight.to(self.device)
        classification_loss = torch.nn.functional.cross_entropy(
            cancer_logits, cancer_label, weight=weight
        )

        weight = self.center_loss_weights
        if weight is not None:
            weight = weight.to(self.device)
        center_loss = torch.nn.functional.cross_entropy(
            center_logits, center_label, weight=weight
        )

        self.log("train/center_loss", center_loss)
        self.log("train/cancer_loss", classification_loss)

        ratio = self.get_disc_loss_ratio()

        if optimizer_idx == 0:
            # divide by ratio so effective learning rate of adv agent stays fixed
            loss = classification_loss - center_loss * ratio
            self.log("train/main_loss", loss)
            return loss
        else:
            adv_loss = center_loss * ratio
            self.log("train/adversarial_loss", adv_loss)
            return adv_loss

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=None,
    ):
        (
            cancer_logits,
            cancer_label,
            center_logits,
            center_label,
            metadata,
        ) = self.shared_step(batch, None)

        collector = (
            self.val_output_collector
            if dataloader_idx is None or dataloader_idx == 0
            else self.test_output_collector
        )
        collector.collect_batch(
            {
                "cancer_logits": cancer_logits,
                "cancer_labels": cancer_label,
                "center_logits": center_logits,
                "center_labels": center_label,
                **metadata,
            }
        )

    def training_epoch_end(self, outputs) -> None:
        out = self.train_output_collector.compute()
        metrics = self.metrics(out)
        self.log_dict(add_prefix(metrics, "train", "/"))
        self.log("disc_loss_ratio", self.get_disc_loss_ratio())

    def validation_epoch_end(self, outputs) -> None:
        out = self.val_output_collector.compute()
        metrics = self.metrics(out)
        self.log_dict(add_prefix(metrics, "val", "/"))
        out = self.test_output_collector.compute()
        if out:
            metrics = self.metrics(out)
            self.log_dict(add_prefix(metrics, "test", "/"))

    def metrics(self, out):
        cancer_clf_out = {
            "preds": out["cancer_logits"].softmax(-1),
            "labels": out["cancer_labels"],
            **out,
        }
        cancer_metrics = apply_metrics_to_patch_and_core_output(
            cancer_clf_out,
            lambda out: compute_center_and_macro_metrics(
                out, self._cancer_base_metrics
            ),
        )
        center_acc_macro = accuracy(
            out["center_logits"],
            out["center_labels"],
            average="macro",
            num_classes=self.num_centers,
        )
        return {**cancer_metrics, "center_acc_macro": center_acc_macro}

    def _cancer_base_metrics(self, out):
        metrics = {}
        metrics["auroc"] = auroc(out["preds"], out["labels"], num_classes=2)
        metrics["macro_acc"] = accuracy(
            out["preds"], out["labels"], average="macro", num_classes=2
        )
        return metrics
