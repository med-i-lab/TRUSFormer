import torch
from torch.nn import Sequential, Linear, ReLU
from .unet import UNet


def ShaoEtAlFeatureExtractor():
    unet = UNet(1, 32)
    dropout = torch.nn.Dropout(0.5)
    fc = torch.nn.Linear((256 * 256 * 32), 1000)

    return torch.nn.Sequential(unet, torch.nn.Flatten(), dropout, fc)


def ShaoEtAlMLP(num_classes):
    return Sequential(
        Linear(1000, 1000),
        ReLU(),
        Linear(1000, 1000),
        ReLU(),
        Linear(1000, num_classes),
    )
