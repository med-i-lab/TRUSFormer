import einops
import torch
from torch.nn.functional import cross_entropy


def dice_score(mask, target_mask, reduce_across_batch=True):
    """
    Computes the dice score between mask and target mask for a single class
    semantic segmentation
    """
    # check if the mask is raw logits 
    if mask.shape[1] > 1:
        mask = torch.argmax(mask, dim=1)

    intersection = torch.logical_and(mask, target_mask)

    size_of_intersection = einops.reduce(intersection, "b h w -> b", reduction="sum")

    size_of_mask = einops.reduce(mask, "b h w -> b", reduction="sum")

    size_of_target_mask = einops.reduce(target_mask, "b h w -> b", reduction="sum")

    dice_scores = 2 * size_of_intersection / (size_of_mask + size_of_target_mask)

    if reduce_across_batch:
        dice_scores = torch.mean(dice_scores)

    return dice_scores


def jaccard(
    mask: torch.Tensor, target_mask: torch.Tensor, reduce_scores_across_batch=True
):
    """
    Computes Jaccard index (IoU) between mask and target mask for a single class semantic
    segmentation task.
    """

    intersection = torch.logical_and(mask, target_mask)
    union = torch.logical_or(mask, target_mask)

    size_of_intersection = einops.reduce(intersection, "b h w -> b", "sum")
    size_of_union = einops.reduce(union, "b h w -> b", "sum")

    iou = size_of_intersection / size_of_union

    if reduce_scores_across_batch:
        iou = torch.mean(iou)

    return iou


class Metrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dice = []
        self.loss = []
        self.jaccard = []

    def forward(self, logits, mask):

        pred_mask = torch.argmax(logits, dim=1)

        loss = cross_entropy(logits, mask).item()
        self.loss.append(loss)

        dice = dice_score(pred_mask, mask).item()
        self.dice.append(dice)

        jaccard_ = jaccard(pred_mask, mask).item()
        self.jaccard.append(jaccard_)

        return {"jaccard": jaccard_, "loss": loss, "dice": dice}

    def compute(self):

        if len(self.dice) == 0:
            raise ValueError("must call __call__ before compute")

        dice = sum(self.dice) / len(self.dice)
        jaccard_ = sum(self.jaccard) / len(self.jaccard)
        loss = sum(self.loss) / len(self.loss)

        return {"jaccard": jaccard_, "loss": loss, "dice": dice}

    def reset(self):
        self.dice = []
        self.loss = []
        self.jaccard = []
