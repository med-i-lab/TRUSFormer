import pytorch_lightning as pl
from ....modeling.unet.unet_model import UNetWithDiscreteFeatureEmbedding, UNet
from ....modeling.segmentation.losses import SegmentationCriterion
from ....modeling.segmentation.metrics import Metrics
from src.data.exact.dataset import BModeSegmentationDataset
from src.data.exact.segmentation_transforms import Preprocessor
from torch.utils.data import DataLoader
import torch
from src.modeling.optimizer_factory import OptimizerConfig, configure_optimizers

ANATOMICAL_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]

ANATOMICAL_LOCATIONS_INV = {name: idx for idx, name in enumerate(ANATOMICAL_LOCATIONS)}


class ExactSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        use_anatomical_location_embeddings=True,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
        dice_loss_weight: float = 0.5,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.opt_cfg = opt_cfg

        self.use_anatomical_location_embeddings = use_anatomical_location_embeddings

        if use_anatomical_location_embeddings:
            self.model = UNetWithDiscreteFeatureEmbedding(
                1, len(ANATOMICAL_LOCATIONS), 2
            )
        else:
            self.model = UNet(1, 2)

        self.criterion = SegmentationCriterion(dice_loss_weight=dice_loss_weight)
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()
        self.test_metrics = Metrics()

        self.train_ds = None
        self.val_ds = None

    def configure_optimizers(self):
        return configure_optimizers(
            [{"params": self.model.parameters()}],
            self.opt_cfg,
        )

    def configure_callbacks(self):
        from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
        from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar

        return LearningRateMonitor(), RichModelSummary()

    def forward(self, bmode, info=None):
        if self.use_anatomical_location_embeddings:
            return self.model(bmode, features=info)
        else:
            return self.model(bmode)

    def training_step(self, batch, batch_idx=None):

        bmode, seg, info = batch
        logits = self(bmode, info=info)

        loss = self.criterion(logits, seg)
        metrics = self.train_metrics(logits, seg)

        self._log_dict(metrics, "train")

        return {
            "loss": loss,
            "bmode": bmode.cpu(),
            "seg": seg.cpu(),
            "logits": logits.detach().cpu(),
            "info": info,
        }

    def training_epoch_end(self, outputs) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        bmode, seg, info = batch
        logits = self(bmode, info=info)

        self._log_dict(self.val_metrics(logits, seg), "val")

        return {
            "bmode": bmode.cpu(),
            "seg": seg.cpu(),
            "logits": logits.detach().cpu(),
            "info": info,
        }

    def validation_epoch_end(self, outputs) -> None:
        self.val_metrics.reset()
        self.test_metrics.reset()

    def _log_dict(self, d, prefix=""):
        self.log_dict({f"{prefix}_{k}": v for k, v in d.items()})
