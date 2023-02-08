import pytorch_lightning as pl
from src.modeling.optimizer_factory import OptimizerConfig, configure_optimizers
from hydra.utils import instantiate
from src.typing import FeatureExtractionProtocol
from torchmetrics import Accuracy, AUROC
from src.modeling.mlp import MLPClassifier, MLPRegressor
import torch
from dataclasses import dataclass
from einops import rearrange
from torchmetrics.regression.r2 import R2Score
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CenterClassifierConfig:

    backbone: Any = None
    optimizer_config: OptimizerConfig = OptimizerConfig()
    finetune: bool = False
    mlp_dropout: float = 0.1
    mlp_hidden_layers: int = 2
    mlp_hidden_dim: int = 128

    _target_: str = (
        "src.lightning.lightning_modules.supervised.center_classifier.CenterClassifier"
    )
    _recursive_: bool = False


class CenterClassifier(pl.LightningModule):
    """
    Attempts to classify the center
    """

    def __init__(
        self,
        backbone,
        optimizer_config=OptimizerConfig,
        finetune=False,
        mlp_dropout=0.1,
        mlp_hidden_layers=2,
        mlp_hidden_dim=128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = instantiate(backbone)
        assert isinstance(
            self.backbone, FeatureExtractionProtocol
        ), f"{self.backbone} does not implement FeatureExtractionProtocol"

        self.optimizer_config = optimizer_config
        self.finetune = finetune
        self.mlp_dropout = mlp_dropout
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule = self.trainer.datamodule
        metadata = self.datamodule.train_ds.metadata

        centers_for_cores = list(metadata["center"])
        number_of_cores_for_each_center = {}
        for center in centers_for_cores:
            if not center in number_of_cores_for_each_center:
                number_of_cores_for_each_center[center] = 0
            number_of_cores_for_each_center[center] += 1
        self.centers = list(number_of_cores_for_each_center.keys())
        logger.info(f"Using centers: {self.centers}")
        self.center_to_center_idx = {center: i for i, center in enumerate(self.centers)}
        self.center_weights = torch.tensor(
            [1 / number_of_cores_for_each_center[center] for center in self.centers]
        )
        logger.info(f"Center weights: {self.center_weights}")
        num_classes = len(self.centers)

        self.train_acc = Accuracy(num_classes=num_classes, average="macro")
        self.train_auroc = AUROC(num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(num_classes=num_classes, average="macro")
        self.val_auroc = AUROC(num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(num_classes=num_classes, average="macro")
        self.test_auroc = AUROC(num_classes=num_classes, average="macro")

        self.classifier = MLPClassifier(
            *[
                self.backbone.features_dim,
                *([self.mlp_hidden_dim] * self.mlp_hidden_layers),
            ],
            dropout=self.mlp_dropout,
            num_classes=num_classes,
        )

        self.optimizer_config = self.optimizer_config
        self.finetune = self.finetune

    def configure_optimizers(self):
        params = [{"params": self.classifier.parameters()}]
        if self.finetune:
            params.append({"params": self.backbone.parameters()})
        return configure_optimizers(
            params,
            self.optimizer_config,
            self.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )

    def on_train_epoch_start(self) -> None:
        if not self.finetune:
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        center = metadata["center"]
        center_idx = torch.tensor([self.center_to_center_idx[c] for c in center]).to(
            self.device
        )

        with torch.no_grad() if not self.finetune else torch.enable_grad():
            features = self.backbone.get_features(patch)
        center_predictions = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(
            center_predictions,
            center_idx,
            weight=self.center_weights.to(center_idx.device),
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_acc(center_predictions, center_idx)
        self.train_auroc(center_predictions, center_idx)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_auroc",
            self.train_auroc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        patch, pos, label, metadata = batch
        center = metadata["center"]
        center_idx = torch.tensor([self.center_to_center_idx[c] for c in center]).to(
            self.device
        )

        with torch.no_grad():
            features = self.backbone.get_features(patch)
            center_predictions = self.classifier(features)
            loss = torch.nn.functional.cross_entropy(
                center_predictions,
                center_idx,
                weight=self.center_weights.to(center_idx.device),
            )
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.val_acc(center_predictions, center_idx)
            self.val_auroc(center_predictions, center_idx)
            self.log(
                "val_acc",
                self.val_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_auroc",
                self.val_auroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        center = metadata["center"]
        center_idx = torch.tensor([self.center_to_center_idx[c] for c in center]).to(
            self.device
        )

        with torch.no_grad():
            features = self.backbone.get_features(patch)
            center_predictions = self.classifier(features)
            loss = torch.nn.functional.cross_entropy(
                center_predictions,
                center_idx,
                weight=self.center_weights.to(center_idx.device),
            )
            self.log(
                "test_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.test_acc(center_predictions, center_idx)
            self.test_auroc(center_predictions, center_idx)
            self.log(
                "test_acc",
                self.test_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_auroc",
                self.test_auroc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )


@dataclass
class RandomPatientDivisionClassifierConfig:
    backbone: Any = None
    optimizer_config: OptimizerConfig = OptimizerConfig()
    seed: int = 0
    finetune: bool = False
    mlp_dropout: float = 0.1
    mlp_hidden_layers: int = 2
    mlp_hidden_dim: int = 128

    _target_: str = __name__ + ".RandomPatientDivisionClassifier"
    _recursive_: bool = False


class RandomPatientDivisionClassifier(pl.LightningModule):
    """
    Attempts to classify the center
    """

    def __init__(
        self,
        backbone,
        optimizer_config=OptimizerConfig,
        seed=0,
        finetune=False,
        mlp_dropout=0.1,
        mlp_hidden_layers=2,
        mlp_hidden_dim=128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = instantiate(backbone)
        assert isinstance(
            self.backbone, FeatureExtractionProtocol
        ), f"{self.backbone} does not implement FeatureExtractionProtocol"

        self.optimizer_config = optimizer_config
        self.finetune = finetune
        self.mlp_dropout = mlp_dropout
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.seed = seed  # for reproducibility

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule = self.trainer.datamodule
        metadata = self.datamodule.train_ds.metadata

        train_patients = list(metadata["patient_specifier"].unique())
        val_patients = list(
            self.datamodule.val_ds.metadata["patient_specifier"].unique()
        )
        test_patients = list(
            self.datamodule.test_ds.metadata["patient_specifier"].unique()
        )
        all_patients = train_patients + val_patients + test_patients

        from sklearn.model_selection import train_test_split

        group_1, group_2 = train_test_split(
            all_patients, test_size=0.5, random_state=self.seed
        )
        self.group_1 = group_1
        self.group_2 = group_2
        number_of_cores_for_each_group = {}
        for patient in list(metadata["patient_specifier"]):
            group_for_patient = "group_1" if patient in self.group_1 else "group_2"
            if not group_for_patient in number_of_cores_for_each_group:
                number_of_cores_for_each_group[group_for_patient] = 0
            number_of_cores_for_each_group[group_for_patient] += 1

        self.group_weights = torch.tensor(
            [
                1 / number_of_cores_for_each_group[group]
                for group in ["group_1", "group_2"]
            ]
        )
        logger.info(
            f"Created arbitrary patient groups. Group1 has {len(group_1)} cores and group2 has {len(self.group_2)} cores."
        )
        num_classes = 2

        self.train_acc = Accuracy(num_classes=num_classes, average="macro")
        self.train_auroc = AUROC(num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(num_classes=num_classes, average="macro")
        self.val_auroc = AUROC(num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(num_classes=num_classes, average="macro")
        self.test_auroc = AUROC(num_classes=num_classes, average="macro")

        self.classifier = MLPClassifier(
            *[
                self.backbone.features_dim,
                *([self.mlp_hidden_dim] * self.mlp_hidden_layers),
            ],
            dropout=self.mlp_dropout,
            num_classes=num_classes,
        )

        self.optimizer_config = self.optimizer_config
        self.finetune = self.finetune

    def configure_optimizers(self):
        params = [{"params": self.classifier.parameters()}]
        if self.finetune:
            params.append({"params": self.backbone.parameters()})
        return configure_optimizers(
            params,
            self.optimizer_config,
            self.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )

    def on_train_epoch_start(self) -> None:
        if not self.finetune:
            self.backbone.eval()

    def patient_specifier_to_label_tensor(self, patient_specifier):
        return (
            torch.tensor(
                [1 if patient in self.group_1 else 0 for patient in patient_specifier]
            )
            .long()
            .to(self.device)
        )

    def training_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        patient_label = self.patient_specifier_to_label_tensor(
            metadata["patient_specifier"]
        )

        with torch.no_grad() if not self.finetune else torch.enable_grad():
            features = self.backbone.get_features(patch)
        patient_group_predictions = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(
            patient_group_predictions,
            patient_label,
            weight=self.group_weights.to(self.device),
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_acc(patient_group_predictions, patient_label)
        self.train_auroc(patient_group_predictions, patient_label)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_auroc",
            self.train_auroc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0:
            return self.test_step(batch, batch_idx)

        patch, pos, label, metadata = batch
        patient_label = self.patient_specifier_to_label_tensor(
            metadata["patient_specifier"]
        )

        with torch.no_grad():
            features = self.backbone.get_features(patch)
            patient_group_predictions = self.classifier(features)
            loss = torch.nn.functional.cross_entropy(
                patient_group_predictions,
                patient_label,
                weight=self.group_weights.to(self.device),
            )
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.val_acc(patient_group_predictions, patient_label)
            self.val_auroc(patient_group_predictions, patient_label)
            self.log(
                "val_acc",
                self.val_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_auroc",
                self.val_auroc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        patient_label = self.patient_specifier_to_label_tensor(
            metadata["patient_specifier"]
        )

        with torch.no_grad():
            features = self.backbone.get_features(patch)
            patient_group_predictions = self.classifier(features)
            loss = torch.nn.functional.cross_entropy(
                patient_group_predictions,
                patient_label,
                weight=self.group_weights.to(self.device),
            )
            self.log(
                "test_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.test_acc(patient_group_predictions, patient_label)
            self.test_auroc(patient_group_predictions, patient_label)
            self.log(
                "test_acc",
                self.test_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_auroc",
                self.test_auroc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )


@dataclass
class CorePositionRegressorConfig:
    backbone: Any = None
    opt_cfg: Any = OptimizerConfig()
    finetune: bool = True

    _target_: str = __name__ + ".CorePositionRegressor"
    _recursive_: bool = False


class CorePositionRegressor(pl.LightningModule):
    def __init__(self, backbone, opt_cfg, finetune=True) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = instantiate(backbone)
        self.lateral_regressor = MLPRegressor(
            *[self.backbone.features_dim, 128, 128], dropout=0.0
        )
        self.axial_regressor = MLPRegressor(
            *[self.backbone.features_dim, 128, 128], dropout=0.0
        )

        self.optimizer_config = opt_cfg
        self.finetune = finetune

        self.train_axial_r2 = R2Score()
        self.train_lateral_r2 = R2Score()
        self.val_axial_r2 = R2Score()
        self.val_lateral_r2 = R2Score()
        self.test_axial_r2 = R2Score()
        self.test_lateral_r2 = R2Score()

    def configure_optimizers(self):
        params = [
            {"params": self.lateral_regressor.parameters()},
            {"params": self.axial_regressor.parameters()},
        ]
        if self.finetune:
            params.append({"params": self.backbone.parameters()})

        return configure_optimizers(
            params,
            self.optimizer_config,
            self.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )

    def on_train_epoch_start(self) -> None:
        if not self.finetune:
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        pos: torch.Tensor
        pos = pos.float()
        with torch.no_grad() if not self.finetune else torch.enable_grad():
            features = self.backbone.get_features(patch)
        lateral_predictions = self.lateral_regressor(features)
        lateral_predictions = lateral_predictions.squeeze()
        axial_predictions = self.axial_regressor(features)
        axial_predictions = axial_predictions.squeeze()

        lateral_loss = torch.nn.functional.mse_loss(lateral_predictions, pos[:, 2])
        axial_loss = torch.nn.functional.mse_loss(axial_predictions, pos[:, 0])
        loss = lateral_loss + axial_loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_lateral_r2(lateral_predictions, pos[:, 2])
        self.train_axial_r2(axial_predictions, pos[:, 0])
        self.log(
            "train_lateral_r2",
            self.train_lateral_r2,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_axial_r2",
            self.train_axial_r2,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        patch, pos, label, metadata = batch
        pos = pos.float()
        with torch.no_grad():
            features = self.backbone.get_features(patch)
            lateral_predictions = self.lateral_regressor(features).squeeze()
            axial_predictions = self.axial_regressor(features).squeeze()
            lateral_loss = torch.nn.functional.mse_loss(lateral_predictions, pos[:, 2])
            axial_loss = torch.nn.functional.mse_loss(axial_predictions, pos[:, 0])
            loss = lateral_loss + axial_loss
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.val_lateral_r2(lateral_predictions, pos[:, 2])
            self.val_axial_r2(axial_predictions, pos[:, 0])
            self.log(
                "val_lateral_r2",
                self.val_lateral_r2,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_axial_r2",
                self.val_axial_r2,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def on_test_epoch_start(self) -> None:
        self._all_axial_predictions = []
        self._all_lateral_predictions = []
        self._all_axial_targets = []
        self._all_lateral_targets = []

    def test_step(self, batch, batch_idx):
        patch, pos, label, metadata = batch
        pos = pos.float()
        with torch.no_grad():
            features = self.backbone.get_features(patch)
            lateral_predictions = self.lateral_regressor(features).squeeze()
            axial_predictions = self.axial_regressor(features).squeeze()
            lateral_loss = torch.nn.functional.mse_loss(lateral_predictions, pos[:, 2])
            axial_loss = torch.nn.functional.mse_loss(axial_predictions, pos[:, 0])
            loss = lateral_loss + axial_loss
            self.log(
                "test_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.test_lateral_r2(lateral_predictions, pos[:, 2])
            self.test_axial_r2(axial_predictions, pos[:, 0])
            self.log(
                "test_lateral_r2",
                self.test_lateral_r2,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_axial_r2",
                self.test_axial_r2,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self._all_axial_predictions.append(axial_predictions.detach().cpu())
            self._all_lateral_predictions.append(lateral_predictions.detach().cpu())
            self._all_axial_targets.append(pos[:, 0].detach().cpu())
            self._all_lateral_targets.append(pos[:, 2].detach().cpu())

    def on_test_epoch_end(self) -> None:
        self._all_axial_predictions = torch.cat(self._all_axial_predictions)
        self._all_lateral_predictions = torch.cat(self._all_lateral_predictions)
        self._all_axial_targets = torch.cat(self._all_axial_targets)
        self._all_lateral_targets = torch.cat(self._all_lateral_targets)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(self._all_axial_targets, self._all_axial_predictions)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title("Axial Depth Regression")
        fig.savefig("axial_regression.png")

        import wandb

        wandb.log({"axial_regression": wandb.Image(fig)})

        fig, ax = plt.subplots()
        ax.scatter(self._all_lateral_targets, self._all_lateral_predictions)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title("Lateral Depth Regression")
        fig.savefig("lateral_regression.png")

        wandb.log({"lateral_regression": wandb.Image(fig)})
