import torch
from torch.nn import functional as F
from torch import nn


class TorchPCA(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        self.register_buffer("Vh", None)
        self.S = None
        self.U = None
        self.register_buffer("mean", None)

    def fit(self, X: torch.Tensor) -> None:
        """
        Fits this data using singular value decomposition in order to be able to perform projection onto the
        principal components.

        Args:
            X (torch.Tensor): The data of shape m, n where m is the number of examples and n is the number of features.
        """

        self.mean = X.mean(dim=0)
        X = X - self.mean

        U, S, Vh = torch.linalg.svd(X)
        self.Vh = Vh
        self.S = S
        self.U = U

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Reduces the dimensionality of the data

        Args:
            X (torch.Tensor): The data of shape
        """

        with torch.no_grad():
            X = X - self.mean
            return F.linear(X, self.Vh[: self.out_features])

    def fit_transform(self, X) -> torch.Tensor:
        self.fit(X)
        return self(X)
