from torch import nn
from src.layers.losses.losses import vicreg_loss_func
from src.modeling.mlp import MLP


class VICReg(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        proj_dims: list,
        features_dim: int,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        inv_loss_weight: float = 25.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(*[features_dim, *proj_dims])
        self.features_dim = features_dim
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.inv_loss_weight = inv_loss_weight

    def forward(self, X1, X2):
        X1 = self.backbone(X1)
        X2 = self.backbone(X2)

        X1 = self.projector(X1)
        X2 = self.projector(X2)

        loss = vicreg_loss_func(
            X1,
            X2,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
            sim_loss_weight=self.inv_loss_weight,
            return_dict=True,
        )
        return loss
