from dataclasses import dataclass, field
from abc import ABC, abstractproperty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from omegaconf import ValidationError
import torch
import torch_optimizer
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from ...utils.modeling import LARSWrapper
from torch.optim.lr_scheduler import MultiStepLR
from functools import partial
from ...modeling.optimizer_factory import OptimizerConfig, configure_optimizers


#
#
# _SUPPORTED_ALGORITHMS = {
#    "SGD": torch.optim.SGD,
#    "Adam": torch.optim.Adam,
#    "AdamW": torch.optim.AdamW,
#    "Novograd": torch_optimizer.NovoGrad,
# }
#
#
# def static_lr(
#    get_lr: Callable,
#    param_group_indexes: Sequence[int],
#    lrs_to_replace: Sequence[float],
# ):
#    lrs = get_lr()
#    for idx, lr in zip(param_group_indexes, lrs_to_replace):
#        lrs[idx] = lr
#    return lrs
#
#
# @dataclass
# class LarsOptions:
#    eta_lars: float
#    grad_clip_lars: bool
#    exclude_bias_n_norm: bool
#
#
# @dataclass
# class SchedulerOptions:
#    warmup_epochs: int = 5
#    warmup_start_lr: float = 0
#    min_lr: float = 0
#    scheduler_interval: str = "step"
#    final_lr: float = 0.0
#    decay_epochs: Tuple[int, int] = (60, 80)
#    scheduler_type: str = "warmup_cosine"
#    max_epochs: Optional[int] = None
#
#
# @dataclass
# class OptimizerConfig:
#    learning_rate: float = 0.0001
#    weight_decay: float = 1e-6
#    nesterov: bool = False
#    gamma: float = 0.1
#    optim_algo: str = "Adam"
#
#    extra_opt_args: Dict = field(default_factory=dict)
#
#    lars_options: Optional[LarsOptions] = None
#    scheduler_options: Optional[SchedulerOptions] = SchedulerOptions()
#
#    def __post_init__(self):
#        if self.optim_algo not in _SUPPORTED_ALGORITHMS:
#            raise ValidationError(
#                f"Given algorithm {self.optim_algo} is not supported."
#            )
#
#
class ConfigureOptimizersMixin(ABC):
    """
    Defines the `configure_optimizers` method for you as long as you provide the config
    via the `set_optimizer_configs` method and override the `learnable_params`, `num_training_steps`,
    and `num_epochs` properties
    """

    def set_optimizer_configs(self, cfg: OptimizerConfig):
        self._opt_cfg = cfg

    @abstractproperty
    def learnable_parameters(self) -> List[dict]:
        ...

    @abstractproperty
    def num_training_steps(self) -> int:
        ...

    @abstractproperty
    def num_epochs(self) -> int:
        ...

    def configure_optimizers(self):
        return configure_optimizers(
            self.learnable_parameters,
            self._opt_cfg,
            num_epochs=self.num_epochs,
            num_scheduling_steps_per_epoch=self.num_training_steps,
        )


class ConfigureMultipleOptimizersMixin(ConfigureOptimizersMixin):
    @abstractproperty
    def learnable_parameters(self) -> Tuple[List[dict]]:
        ...

    def set_optimizer_configs(self, *cfg: OptimizerConfig):
        self._opt_cfgs = cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        opts, sched = [], []

        for cfg, param in zip(self._opt_cfgs, self.learnable_parameters):
            out = configure_optimizers(
                param, cfg, self.num_training_steps, self.num_epochs
            )
            opts.extend(out[0])
            sched.extend(out[1])

        return opts, sched
