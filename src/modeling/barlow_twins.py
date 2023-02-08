import torch
from torch import nn
from .mlp import MLP
from torch.distributed import all_reduce, is_initialized, get_world_size


class BarlowTwins(nn.Module):
    def __init__(self, backbone, proj_dims, features_dim, lambda_=5e-3):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(*[features_dim, *proj_dims], use_batchnorm=True)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(proj_dims[-1], affine=False)

        self.lambda_ = lambda_

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        bsz = z1.shape[0]

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        if is_initialized():
            # sum the cross-correlation matrix between all gpus
            num_gpus = get_world_size()
            total_batch_size = bsz * num_gpus
            c.div_(total_batch_size)
            all_reduce(c)

        else:
            c.div_(bsz)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_ * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
