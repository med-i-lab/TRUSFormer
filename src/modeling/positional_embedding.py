from torch import nn
import torch


class GridPositionEmbedder2d(nn.Module):
    def __init__(self, embedding_dim, grid_shape):
        super().__init__()
        self.embeddings = nn.parameter.Parameter(
            torch.randn(*grid_shape, embedding_dim)
        )

    def forward(self, x, y):
        return self.embeddings[x, y]
