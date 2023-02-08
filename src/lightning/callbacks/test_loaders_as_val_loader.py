from typing import Optional, Sequence
from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl 


def wrap_to_list(obj): 
    if not isinstance(obj, Sequence): 
        return [obj]
    else: return obj


class TestAsValLoader(Callback):

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:

        dm: LightningDataModule
        dm = trainer.datamodule
        dm._old_val_dataloader = dm.val_dataloader

        def val_dataloader():
            val_loaders = wrap_to_list(dm._old_val_dataloader())
            test_loaders = wrap_to_list(dm.test_dataloader())
            return val_loaders + test_loaders   #type:ignore

        dm.val_dataloader = val_dataloader

        