import numpy as np
import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


class SimpleAggregNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, num_classes),
        )

    def _forward_impl(self, x):
        batch_size, *args = x.size()

        x = x.view(batch_size, -1)

        return self.model(x)

    def forward(self, x, corelen, *args):
        corelen_start = corelen
        corelen_end = np.append(corelen[1:].detach().cpu(), None)  # type:ignore

        core_rep = []
        for i, j in zip(corelen_start, corelen_end):
            x_core = x[i:j, ...]
            x_core = torch.sum(x_core, dim=0)
            core_rep.append(x_core)

        agg_core_reps = torch.stack(core_rep, dim=0)  # batch*repres_size

        return self._forward_impl(agg_core_reps[:, None, ...])
