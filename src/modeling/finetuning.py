from torch import nn
import torch
from .mlp import MLP


class Finetuner(nn.Module):
    def __init__(
        self, backbone: nn.Module, mlp_dims, features_dim, freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.mlp_dims = mlp_dims
        self.classifier = MLP(*[features_dim, *mlp_dims])
        self.features_dim = features_dim
        self.frozen_backbone = freeze_backbone

    def train(self):
        """keep backbone in eval mode if frozen"""
        super().train()
        if self.frozen_backbone:
            self.backbone.eval()

    def parameters(self, recurse: bool = True):
        if self.frozen_backbone:
            return self.classifier.parameters(recurse=recurse)
        else:
            return super().parameters(recurse=recurse)

    def forward(self, X):
        if self.frozen_backbone:
            with torch.no_grad():
                X = self.backbone(X)
        else:
            X = self.backbone(X)
        X = self.classifier(X)
        return X

    def reinit_classifier(self):
        for param in self.classifier.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
