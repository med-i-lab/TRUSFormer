from dataclasses import dataclass, field
from abc import ABC, abstractproperty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from omegaconf import ValidationError
import torch
import torch_optimizer
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ChainedScheduler, LambdaLR
from ..utils.modeling import LARSWrapper
from torch.optim.lr_scheduler import MultiStepLR
from functools import partial
from typing import Literal


_SUPPORTED_ALGORITHMS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "Novograd": torch_optimizer.NovoGrad,
}


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


@dataclass
class LarsOptions:
    eta_lars: float
    grad_clip_lars: bool
    exclude_bias_n_norm: bool


@dataclass
class SchedulerOptions:
    warmup_epochs: int = 5
    warmup_start_lr: float = 0
    min_lr: float = 0
    scheduler_interval: str = "step"
    final_lr: float = 0.0
    decay_epochs: Tuple[int, int] = (60, 80)
    scheduler_type: str = "warmup_cosine"
    max_epochs: Optional[int] = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0001
    weight_decay: float = 1e-6
    nesterov: bool = False
    gamma: float = 0.1
    optim_algo: str = "Adam"

    extra_opt_args: Dict = field(default_factory=dict)

    lars_options: Optional[LarsOptions] = None
    scheduler_options: Optional[SchedulerOptions] = SchedulerOptions()

    def __post_init__(self):
        if self.optim_algo not in _SUPPORTED_ALGORITHMS:
            raise ValidationError(
                f"Given algorithm {self.optim_algo} is not supported."
            )


def configure_optimizers(
    parameters,
    opt_config: OptimizerConfig,
    num_epochs=None,
    num_scheduling_steps_per_epoch=None,
    lightning_format=True,
    return_dict=False,
):

    """Collects learnable parameters and configures the optimizer and learning rate scheduler.

    parameters: the parameters to optimize
    opt_config: the optimizer configuration
    num_training_steps_per_epoch: necessary to program the scheduler
    num_epochs: total number of training epochs for scheduling
    lightning_format: if true, the output is in the form preferred by pytorch lightning's
        `model.configure_optimizers()` function. Otherwise, simply returns opt, sched as a tuple.
    """

    # collect learnable parameters
    idxs_no_scheduler = [
        i for i, m in enumerate(parameters) if m.pop("static_lr", False)
    ]

    # create optimizer
    optim_algo = _SUPPORTED_ALGORITHMS[opt_config.optim_algo]
    optimizer = optim_algo(
        parameters,
        lr=opt_config.learning_rate,
        weight_decay=opt_config.weight_decay,
        **opt_config.extra_opt_args,
    )

    # optionally wrap with lars
    if opt_config.lars_options:
        assert opt_config.optim_algo == "SGD", "LARS is only compatible with SGD."
        optimizer = LARSWrapper(
            optimizer,
            eta=opt_config.lars_options.eta_lars,
            clip=opt_config.lars_options.grad_clip_lars,
            exclude_bias_n_norm=opt_config.lars_options.exclude_bias_n_norm,
        )

    if opt_config.scheduler_options is None:
        if return_dict:
            return {"optimizer": optimizer}
        return [optimizer], [None]

    if opt_config.scheduler_options.scheduler_type == "warmup_cosine":
        if num_epochs is None:
            raise ValueError("Must specify number of epochs for this scheduler")
        if opt_config.scheduler_options.scheduler_interval == "step":
            if num_scheduling_steps_per_epoch is None:
                raise ValueError(
                    "Must specify number of steps per epoch when using `step` as the scheduling interval"
                )

        else:
            num_scheduling_steps_per_epoch = 1

        max_epochs = (
            opt_config.scheduler_options.max_epochs
            if opt_config.scheduler_options.max_epochs is not None
            else num_epochs
        )

        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,  # type:ignore
                warmup_epochs=opt_config.scheduler_options.warmup_epochs
                * num_scheduling_steps_per_epoch,
                max_epochs=max_epochs * num_scheduling_steps_per_epoch,
                warmup_start_lr=opt_config.scheduler_options.warmup_start_lr
                if opt_config.scheduler_options.warmup_epochs > 0
                else opt_config.learning_rate,
                eta_min=opt_config.scheduler_options.min_lr,
            ),
            "interval": opt_config.scheduler_options.scheduler_interval,
            "frequency": 1,
        }

    elif opt_config.scheduler_options.scheduler_type == "step":
        scheduler = MultiStepLR(
            optimizer, opt_config.scheduler_options.decay_epochs  # type:ignore
        )

    else:
        raise ValueError(
            f"{opt_config.scheduler_options.scheduler_type} not in (warmup_cosine, cosine, step)"
        )

    if idxs_no_scheduler:
        partial_fn = partial(
            static_lr,
            get_lr=scheduler["scheduler"].get_lr
            if isinstance(scheduler, dict)
            else scheduler.get_lr,
            param_group_indexes=idxs_no_scheduler,
            lrs_to_replace=[opt_config.learning_rate] * len(idxs_no_scheduler),
        )
        if isinstance(scheduler, dict):
            scheduler["scheduler"].get_lr = partial_fn
        else:
            scheduler.get_lr = partial_fn

    if lightning_format:
        if return_dict:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        else:
            return [optimizer], [scheduler]

    else:
        if isinstance(scheduler, Dict):
            scheduler = scheduler["scheduler"]

        if return_dict:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        else:
            return optimizer, scheduler
