import contextlib

from ....modeling.registry import create_model

from ..configure_optimizer_mixin import OptimizerConfig
from ..evaluation_base import EvaluationBase, SharedStepOutput
import torch
from torch.nn import functional as F


def get_remember_rate(
    current_epoch,
    max_epochs,
    final_remember_rate,
    final_remember_rate_epoch_frac: float,
):

    x_end = int(max_epochs * final_remember_rate_epoch_frac)

    if current_epoch > x_end:
        return final_remember_rate

    y_0 = 1
    y_end = final_remember_rate

    x = current_epoch

    b = y_0
    a = (y_end - b) / x_end

    y = a * x + b
    return y


class SupervisedCoteachingModel(EvaluationBase):
    def __init__(
        self,
        model1_name: str,
        model2_name: str,
        final_remember_rate,
        final_remember_rate_epoch_frac,
        batch_size: int,
        epochs: int = 100,
        learning_rate: float = 0.1,
    ):

        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            opt_cfg=OptimizerConfig(learning_rate=learning_rate),
        )

        self.final_remember_rate = final_remember_rate
        self.final_remember_rate_epoch_frac = final_remember_rate_epoch_frac

        self.model1 = create_model(model1_name)
        self.model2 = create_model(model2_name)

    def get_learnable_parameters(self):
        from itertools import chain

        return chain(self.model1.parameters(), self.model2.parameters())

    def shared_step(self, batch) -> SharedStepOutput:

        X, y, metadata = batch

        with contextlib.nullcontext() if self.training else torch.no_grad():
            logits1 = self.model1(X)
            logits2 = self.model2(X)

            loss1 = F.cross_entropy(logits1, y, reduce=False)
            loss2 = F.cross_entropy(logits2, y, reduce=False)

            r_t = self.get_remember_rate()
            total_samples = len(y)
            samples_to_remember = int(r_t * total_samples)

            _, ind_for_loss1 = torch.topk(loss2, samples_to_remember, largest=False)
            _, ind_for_loss2 = torch.topk(loss1, samples_to_remember, largest=False)

            loss_filter_1 = torch.zeros((loss1.size(0))).to(self.device)
            loss_filter_1[ind_for_loss1] = 1.0
            loss1 = (loss_filter_1 * loss1).sum()

            loss_filter_2 = torch.zeros((loss2.size(0))).to(self.device)
            loss_filter_2[ind_for_loss2] = 1.0
            loss2 = (loss_filter_2 * loss2).sum()

            loss = loss1 + loss2

        return SharedStepOutput(logits=logits1, y=y, loss=loss, metadata=[metadata])

    def get_remember_rate(self):

        return get_remember_rate(
            self.current_epoch,
            self.epochs,
            self.final_remember_rate,
            self.final_remember_rate_epoch_frac,
        )

    def on_epoch_end(self):
        self.log("remember_rate", self.get_remember_rate())

    def forward(self, X):
        return self.model1(X)
