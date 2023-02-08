import os
from pytorch_lightning.callbacks import Callback
from typing import Optional, Sequence, List
from pytorch_lightning import LightningDataModule
from torch import nn
import pytorch_lightning as pl
import logging
import torch
from torchmetrics.functional import auroc, accuracy, average_precision
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from pytorch_lightning.loggers import LightningLoggerBase
import torch
from .components.early_stopping_monitor import EarlyStoppingMonitor, ScoreMonitor
from ...typing import FeatureExtractionProtocol
from contextlib import nullcontext
from typing import Callable
from src.utils.metrics import (
    OutputCollector,
    patch_out_to_core_out,
    add_prefix,
    apply_metrics_to_patch_and_core_output,
    compute_center_and_macro_metrics,
)
from src.utils.driver.stateful import (
    auto_stateful,
    StatefulAttribute,
    StatefulCollection,
)
from src.utils.driver.checkpoint_helper import CheckpointMemory
import copy
from contextlib import contextmanager


logger = logging.getLogger("Callbacks.OnlineEvaluation")


def add_prefix_to_dict(d, prefix: str, seperator="/"):
    return {f"{prefix}{seperator}{k}": v for k, v in d.items()}


@contextmanager
def temporary_torch_seed(seed):
    current_seed = torch.random.get_rng_state()
    torch.manual_seed(seed)
    yield None
    torch.random.set_rng_state(current_seed)


class OnlineEvaluationBase(Callback):
    def __init__(
        self,
        datamodule: LightningDataModule,
        metrics_fn: Callable,
        num_classes: int = 2,
        num_epochs_per_run: int = 10,
        evaluate_every_n_epochs: Optional[int] = None,
        start_on_epoch: Optional[int] = None,
        evaluate_once_on_epoch: Optional[int] = None,
        log_best_only=True,
        lr=1e-4,
        weight_decay=1e-6,
        scheduler_epochs=100,
        warmup_epochs=10,
        patience=100,
        monitored_metric="patch_auroc",
        finetune=False,
        prefix="lin_eval",
        linear_layer_seed: int = 0,
    ):
        ...


class OnlineEvaluation(Callback):
    def __init__(
        self,
        datamodule: LightningDataModule,
        metrics_fn: Callable,
        num_classes: int = 2,
        num_epochs_per_run: int = 10,
        evaluate_every_n_epochs: Optional[int] = None,
        start_on_epoch: Optional[int] = None,
        evaluate_once_on_epoch: Optional[int] = None,
        log_best_only=True,
        lr=1e-4,
        weight_decay=1e-6,
        scheduler_epochs=100,
        warmup_epochs=10,
        patience=100,
        monitored_metric="patch_auroc",
        finetune=False,
        prefix="lin_eval",
        linear_layer_seed: int = 0,
    ):
        self.num_classes = num_classes
        self.datamodule = datamodule
        self.num_epochs_per_run = num_epochs_per_run
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.start_on_epoch = start_on_epoch
        self.evaluate_once_on_epoch = evaluate_once_on_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_epochs = scheduler_epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.log_best_only = log_best_only
        self.monitored_metric = monitored_metric
        self.metrics_fn = metrics_fn
        self.prefix = prefix
        self.lin_layer_seed = linear_layer_seed
        self.finetune = finetune

        self.state_collector = StatefulCollection()

        with self.state_collector.auto_register(self):
            self._total_epochs = StatefulAttribute(0)
            self._best_train_metrics_global = StatefulAttribute({})
            self._best_val_metrics_global = StatefulAttribute({})
            self._best_test_metrics_global = StatefulAttribute({})
            self._best_score_global = StatefulAttribute(-1e9)
            self._current_run_best_score = StatefulAttribute(-1e9)
            self._best_train_metrics = StatefulAttribute({})
            self._best_val_metrics = StatefulAttribute({})
            self._best_test_metrics = StatefulAttribute({})
            self._online_eval_epoch = StatefulAttribute(0)
            self._should_early_stop = StatefulAttribute(False)
            self._run_completed_for_epoch = StatefulAttribute([])

            self.early_stopping_monitor = EarlyStoppingMonitor(
                self._trigger_early_stop, patience=self.patience
            )
            self._model_checkpoint_memory = CheckpointMemory(memory=2)

        self._cached_sched_state: Optional[dict] = None
        self._cached_opt_state: Optional[dict] = None
        self._cached_model_state: Optional[dict] = None
        self._resuming = False

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:

        logger.info("Setting up datamodule for online evaluation")
        self.datamodule.setup()
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()
        if isinstance(self.val_loader, Sequence):
            self.val_loader = self.val_loader[0]
        self.test_loader = self.datamodule.test_dataloader()
        if isinstance(self.test_loader, Sequence):
            self.test_loader = self.test_loader[0]
        self.num_training_steps_per_epoch = len(self.datamodule.train_dataloader())

        self.pl_module = pl_module
        assert hasattr(
            self.pl_module, "backbone"
        ), f"Module must have `backbone` attribute which is the feature extractor"
        backbone = self.pl_module.backbone
        assert isinstance(
            backbone, FeatureExtractionProtocol
        ), f"Backbone module should implement the <src.models.typing.FeatureExtractionProtocol> interface."
        self.feature_dim = backbone.features_dim
        self.linear_layer = torch.nn.Linear(self.feature_dim, self.num_classes)

        self.backbone = copy.deepcopy(backbone)

        # attach linear layer reference to pl_module so it gets moved to correct device
        # self.pl_module.linear_layer = self.linear_layer

        # make sure checkpoint directory can be accessed from trainer
        ERR_MSG = "Trainer must be configured with checkpoint callback to use checkpoint for online trainer"
        assert trainer.checkpoint_callback is not None, ERR_MSG
        assert trainer.checkpoint_callback.dirpath is not None, ERR_MSG
        self.checkpoint_dir = trainer.checkpoint_callback.dirpath

        self.trainer = trainer
        # attach reference to the trainer so that you can access features of the callbacks
        trainer.online_eval_callback = self  # type:ignore
        self._build_optimizer()

    def _build_optimizer(self):

        params = [{"params": self.linear_layer.parameters()}]
        if self.finetune:
            params.append({"params": self.backbone.parameters()})
        optimizer = torch.optim.Adam(
            params,
            self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_epochs * self.num_training_steps_per_epoch,
            max_epochs=self.scheduler_epochs * self.num_training_steps_per_epoch,
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

    def _trigger_early_stop(self):
        self._should_early_stop.value = True

    def _online_evaluation_run_start(self):
        device = self.pl_module.device
        self.linear_layer.to(device)

        with temporary_torch_seed(self.lin_layer_seed + self.current_epoch_for_run):
            self.linear_layer.reset_parameters()

        self.backbone.to(device)

        # load state into backbone from pl module
        self.backbone.load_state_dict(self.pl_module.backbone.state_dict())

        if self._resuming:
            self._load_model_state()

        self._build_optimizer()

        if self._resuming:
            self._load_optimizer_state()

        self._resuming = False

    def _load_model_state(self):
        assert self._cached_model_state is not None
        sd = self._cached_model_state
        self.linear_layer.load_state_dict(sd["linear_layer"])
        self.backbone.load_state_dict(sd["backbone"])

    def _online_evaluation_run_end(self):
        self._best_test_metrics.value = {}
        self._best_val_metrics.value = {}
        self._best_train_metrics.value = {}
        self._online_eval_epoch.value = 0
        self._should_early_stop.value = False
        self._current_run_best_score.value = -1e9
        self.early_stopping_monitor.reset()
        self._run_completed_for_epoch.value.append(self.pl_module.current_epoch)
        # reset linear layer parameters

    def _load_optimizer_state(self):
        assert self._cached_opt_state is not None
        self.optimizer.load_state_dict(self._cached_opt_state)
        assert self._cached_sched_state is not None
        self.scheduler.load_state_dict(self._cached_sched_state)

    @property
    def _should_continue_training(self):
        if self._should_early_stop.value:
            return False
        if self.current_epoch_for_run >= self.num_epochs_per_run:
            return False
        return True

    @property
    def current_epoch_for_run(self) -> int:
        return self._online_eval_epoch.value  # type:ignore

    @property
    def _should_run_this_epoch(self):
        current_pretrain_epoch = self.pl_module.current_epoch
        if current_pretrain_epoch in self._run_completed_for_epoch.value:
            return False
        if self.evaluate_once_on_epoch is not None:
            return current_pretrain_epoch == self.evaluate_once_on_epoch
        else:
            assert (
                self.evaluate_every_n_epochs is not None
            ), f"Must specify epoch interval if evaluating more than once"
            if self.start_on_epoch is None:
                self.start_on_epoch = self.evaluate_every_n_epochs
            rel_epoch = current_pretrain_epoch - self.start_on_epoch
            return rel_epoch >= 0 and rel_epoch % self.evaluate_every_n_epochs == 0

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        assert self.pl_module is not None

        if not self._should_run_this_epoch:
            return

        logging.info("Online evaluation Beginning:")

        self._online_evaluation_run_start()

        while self._should_continue_training:

            self._checkpoint_state()

            # run train, val, test
            train_metrics = self._epoch(
                self.train_loader,
                True,
                state_description=f"{self.prefix.capitalize()}; Epoch {self.current_epoch_for_run}; Train",
            )
            val_metrics = self._epoch(
                self.val_loader,
                False,
                state_description=f"{self.prefix.capitalize()}; Epoch {self.current_epoch_for_run}; Val",
            )
            test_metrics = self._epoch(
                self.test_loader,
                False,
                state_description=f"{self.prefix.capitalize()}; Epoch {self.current_epoch_for_run}; Test",
            )

            score_for_epoch = val_metrics[self.monitored_metric]

            # check and update best metrics for this online evaluation routine
            if score_for_epoch > self._current_run_best_score.value:
                self._current_run_best_score.value = score_for_epoch
                self._best_test_metrics.value = test_metrics
                self._best_val_metrics.value = val_metrics
                self._best_train_metrics.value = train_metrics

            # check and update best metrics across all online evaluation routines
            if score_for_epoch > self._best_score_global.value:
                self._best_score_global.value = score_for_epoch
                self._best_test_metrics_global.value = test_metrics
                self._best_val_metrics_global.value = val_metrics
                self._best_train_metrics_global.value = train_metrics
                self._checkpoint_model(score_for_epoch)

            # maybe_trigger early stopping
            self.early_stopping_monitor.update(score_for_epoch)

            # log epoch metrics
            if not self.log_best_only:
                self.log_metrics(
                    add_prefix_to_dict(train_metrics, "train", "_"),
                )
                self.log_metrics(
                    add_prefix_to_dict(val_metrics, "val", "_"),
                )
                self.log_metrics(
                    add_prefix_to_dict(test_metrics, "test", "_"),
                )

            self._online_eval_epoch.value += 1  # type:ignore
            self._total_epochs.value += 1  # type:ignore

        # log best metrics for this run
        self.log_metrics(
            add_prefix_to_dict(self._best_train_metrics.value, "best_train", "_"),
        )
        self.log_metrics(
            add_prefix_to_dict(self._best_val_metrics.value, "best_val", "_"),
        )
        self.log_metrics(
            add_prefix_to_dict(self._best_test_metrics.value, "best_test", "_"),
        )

        self._online_evaluation_run_end()
        logging.info("Online evaluation run complete.")

    def _checkpoint_model(self, score):
        fpath = os.path.join(
            self.checkpoint_dir,
            f"{self.prefix}_online_best_{self.monitored_metric}_{score:.3f}.ckpt",
        )
        logger.info(
            f"""
        Saving checkpoint to: 
            {fpath}
        """
        )
        self.trainer.save_checkpoint(fpath)
        self._model_checkpoint_memory.update_and_cleanup(fpath)

    def _checkpoint_state(self):
        """Saves checkpoint for the trainer in the same spot where the experiment state checkpoint (`last`)
        is being saved."""
        ckpt_path = os.path.join(self.checkpoint_dir, "last.ckpt")
        self.trainer.save_checkpoint(ckpt_path)

    def _epoch(self, loader, train=True, state_description="Online Evaluation"):

        assert self.pl_module is not None
        if train and self.finetune:
            logger.debug(f"Setting backbone to train mode.")
            self.backbone.train()
        else:
            logger.debug(f"Setting backbone to evaluation mode.")
            self.backbone.eval()

        assert self.linear_layer is not None, "Call self._build_linear_layer()"

        optimizer, scheduler = self.optimizer, self.scheduler

        collector = OutputCollector()

        with tqdm(loader, desc=state_description, leave=False) as pbar:
            for batch in pbar:

                x, pos, y, metadata = batch

                x = x.to(self.pl_module.device)
                y = y.to(self.pl_module.device)

                if not self.finetune:
                    with torch.no_grad():
                        features = self.backbone(x)
                else:
                    features = self.backbone(x)

                logits = self.linear_layer(features)
                loss = nn.functional.cross_entropy(logits, y)

                collector.collect_batch({"logits": logits, "labels": y, **metadata})

                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            if self.scheduler:
                self.log_metrics(
                    {"lr": self.scheduler.get_last_lr()[-1]},
                )

            out = collector.compute()
            metrics = self._compute_metrics(out)
            # pbar.set_postfix({"auroc": metrics["auroc"]})

        return metrics

    def _compute_metrics(self, out):
        return self.metrics_fn(out)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.log_metrics(
            add_prefix_to_dict(
                self._best_train_metrics_global.value,
                f"global_best_train",
                "_",
            )
        )
        self.log_metrics(
            add_prefix_to_dict(
                self._best_val_metrics_global.value, "global_best_val", "_"
            )
        )
        self.log_metrics(
            add_prefix_to_dict(
                self._best_test_metrics_global.value,
                "global_best_test",
                "_",
            )
        )

    def log_metrics(self, metrics: dict):
        import wandb

        metrics.update({"total_epochs": self._total_epochs.value})
        metrics = add_prefix_to_dict(metrics, self.prefix, "/")
        wandb.log(metrics)

    def load_state_dict(self, state_dict) -> None:
        self.state_collector.load_state_dict(state_dict["state"])
        self._resuming = True
        self._cached_opt_state = state_dict["opt"]
        self._cached_sched_state = state_dict["sched"]
        self._cached_model_state = state_dict["model"]

    def state_dict(self):
        return {
            "model": {
                "backbone": self.backbone.state_dict(),
                "linear_layer": self.linear_layer.state_dict(),
            },
            "state": self.state_collector.state_dict(),
            "opt": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
        }

    @property
    def state_key(self) -> str:
        return self._generate_state_key(prefix=self.prefix)
