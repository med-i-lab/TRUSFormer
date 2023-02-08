from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch

segmentation_classes = ["background", "prostate"]


def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(
        bg_img,
        masks={
            "prediction": {"mask_data": pred_mask, "class_labels": labels()},
            "ground truth": {"mask_data": true_mask, "class_labels": labels()},
        },
    )


class LogMasksCallback(Callback):
    def __init__(self, num_batches_to_log=10):
        self.num_batches_to_log = num_batches_to_log

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch,
        batch_idx: int,
        unused: int = 0,
    ) -> None:

        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        if self.num_batches_to_log and batch_idx >= self.num_batches_to_log:
            return

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).numpy()
        seg = outputs["seg"].numpy()
        bmode = outputs["bmode"].numpy()

        images = [
            wb_mask(bg_img, pred_mask, true_mask)
            for (bg_img, pred_mask, true_mask) in zip(bmode, preds, seg)
        ]

        wandb.log({f"train/batch_{batch_idx}_images": images})

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch,
        batch_idx: int,
        dataloader_idx=None,
    ) -> None:

        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        if self.num_batches_to_log and batch_idx >= self.num_batches_to_log:
            return

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).numpy()
        seg = outputs["seg"].numpy()
        bmode = outputs["bmode"].numpy()

        images = [
            wb_mask(bg_img, pred_mask, true_mask)
            for (bg_img, pred_mask, true_mask) in zip(bmode, preds, seg)
        ]

        wandb.log({f"val/batch_{batch_idx}_images": images})