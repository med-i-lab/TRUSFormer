from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
import torch


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, train_transform, val_transform, batch_size=64, num_workers=8):
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=self.train_transform,
            target_transform=self._target_transform,
        )
        self.val_dataset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=self.val_transform,
            target_transform=self._target_transform,
        )

    def _target_transform(self, target):
        return torch.tensor(target).long()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
