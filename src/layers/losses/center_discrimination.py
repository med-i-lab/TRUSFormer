
CENTER_LOSS_WEIGHTS = {
    "UVA": 2.8295503211991435,
    "PCC": 4.131957473420888,
    "CRCEO": 4.497617426820967,
    "JH": 10.72564935064935,
    "PMCC": 11.236394557823129,
}
from src.lightning.datamodules.exact_datamodule import CENTERS

CENTER_LOSS_WEIGHTS = tuple([CENTER_LOSS_WEIGHTS[center] for center in CENTERS])


class CenterDiscriminationLoss:
    def __init__(self, center_loss_weights=CENTER_LOSS_WEIGHTS):
        self.center_loss_weights = torch.tensor(center_loss_weights)

    def __call__(self, preds, targets):
        from torch.nn.functional import cross_entropy

        self.center_loss_weights = self.center_loss_weights.to(preds.device)
        return cross_entropy(preds, targets, weight=self.center_loss_weights)

