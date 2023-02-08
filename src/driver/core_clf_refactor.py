from dataclasses import dataclass
import os
import random
from omegaconf import OmegaConf
import wandb
from typing import Any, Deque, Optional, Tuple
from typing import Protocol, runtime_checkable
from collections import deque
from hydra.utils import instantiate
import numpy as np
from ..utils.metrics import compute_center_and_macro_metrics, add_prefix
from contextlib import nullcontext
from torchmetrics.functional import accuracy, auroc
from tqdm import tqdm
from ..typing import FeatureExtractionProtocol
from ..modeling.optimizer_factory import OptimizerConfig, configure_optimizers
import logging
from pytorch_lightning.callbacks import RichModelSummary
from src.utils.metrics import OutputCollector, group_output_by_centers
import torch
from dataclasses import field
from ..utils.driver.stateful import (
    Stateful,
    StatefulAttribute,
    StatefulCollection,
    RNGStatefulProxy,
)
from ..utils.driver.counters import Counters
from ..utils.driver.checkpoint_helper import CheckpointHelper, setup_ckpt_dir
from ..utils.driver.monitor import ScoreImprovementMonitor
from ..utils.driver import config_to_dict
from ..utils.driver.early_stopping import EarlyStoppingMonitor
from functools import partial


logger = logging.getLogger(__name__)
log = logger.info


def base_metrics(out):
    preds = out["preds"]
    labels = out["labels"]
    return {
        "auroc": auroc(preds, labels, num_classes=2).item(),
        # "macro_acc": accuracy(preds, labels, average="macro", num_classes=2).item(),
    }


@dataclass
class LoggingOptions:
    run_id: Optional[str] = None
    project: str = "Exact_IPCAI"
    run_name: Optional[str] = None
    entity: Optional[str] = None


def detect_checkpoint_dir():
    import os

    id = os.getenv("SLURM_JOB_ID")
    if id is None:
        return None
    else:
        return f"/checkpoint/pwilson/{id}"


@dataclass
class CheckpointOptions:
    checkpoint_dir: str = "checkpoints"
    save_k_best: int = 5
    monitored_metric: str = "val/micro_avg_auroc"
    symlink_target: Optional[str] = field(default_factory=detect_checkpoint_dir)


@dataclass
class TrainingOptions:
    num_epochs: int = 100
    accumulate_grad_batches: int = 64


@dataclass
class ExperimentConfig:

    seed: int
    core_dm: Any

    feat_extractor: Any
    seq_model: Any
    device: str = "cuda:0"
    patch_dm: Optional[Any] = None
    use_psa_as_feature: bool = False
    training_options: TrainingOptions = TrainingOptions()
    logging_options: LoggingOptions = LoggingOptions()
    checkpoint_options: CheckpointOptions = CheckpointOptions()
    opt_config: OptimizerConfig = OptimizerConfig(
        scheduler_options=None,
        learning_rate=3e-5,
    )
    feat_extractor_opt_config: Optional[OptimizerConfig] = None
    start_training_feat_extractor_at_epoch: int = 20
    early_stopping_patience: int = 10


@dataclass
class ExperimentConfigWrapper:

    config: ExperimentConfig
    _target_: str = __name__ + ".CoreClassificationWithTransformers"
    _recursive_: bool = False


@runtime_checkable
class MustFitLatentSpace(Protocol):
    def fit_latent_space(self, all_train_latents):
        ...


class CoreClassificationWithTransformers:
    def __init__(self, config: ExperimentConfig):

        from torch.multiprocessing import set_sharing_strategy

        set_sharing_strategy("file_system")

        self.config = config

        log(f"Setting global seed to {config.seed}")
        from pytorch_lightning import seed_everything

        seed_everything(config.seed)

        self.model_state = StatefulCollection()
        self.experiment_state = StatefulCollection()

        self._setup_data()
        self._setup_models()
        self._setup_optimization()

        self.counters = Counters()
        self.counters.add_counters(["epoch", "global_step", "train_step", "opt_step"])
        self.experiment_state.register_stateful("counters", self.counters)

        self.rng = RNGStatefulProxy()
        self.experiment_state.register_stateful("rng", self.rng)

        self.early_stopping_monitor = EarlyStoppingMonitor(
            self.config.early_stopping_patience
        )
        self.experiment_state.register_stateful(
            "early_stopping_monitor", self.early_stopping_monitor
        )
        self.score_improvement_monitor = ScoreImprovementMonitor()
        self.score_improvement_monitor.on_improvement(
            self.update_best_metrics, self.checkpoint_model
        )
        self.score_improvement_monitor.on_improvement(
            lambda score: self.early_stopping_monitor.update(improvement=True)
        )
        self.score_improvement_monitor.on_no_improvement(
            lambda score: self.early_stopping_monitor.update(improvement=False)
        )
        self.experiment_state.register_stateful(
            "score_improvement_monitor", self.score_improvement_monitor
        )
        self.best_metrics = StatefulAttribute()
        self.experiment_state.register_stateful("best_metrics", self.best_metrics)
        self.current_metrics = StatefulAttribute()
        self.experiment_state.register_stateful("current_metrics", self.current_metrics)

        setup_ckpt_dir(
            config.checkpoint_options.checkpoint_dir,
            config.checkpoint_options.symlink_target,
        )
        self.checkpoint_saver = CheckpointHelper(
            dir=config.checkpoint_options.checkpoint_dir,
            memory=config.checkpoint_options.save_k_best,
        )
        self.experiment_checkpoint_saver = CheckpointHelper(
            dir=config.checkpoint_options.checkpoint_dir
        )

        self._train_output_collector = OutputCollector()
        self._eval_output_collector = OutputCollector()

        self._setup_logging()

    def update_best_metrics(self, score):
        self.best_metrics.value = add_prefix(self.current_metrics.value, "best")

    def checkpoint_model(self, score):
        log(f"Saving new best model to checkpoints!")
        monitor_name = self.config.checkpoint_options.monitored_metric.replace("/", "_")
        epoch = self.counters.counters["epoch"]
        checkpoint_name = f"best_model_epoch{epoch}_{monitor_name}_{score:.3f}.ckpt"
        sd = self.model_state.state_dict()
        self.checkpoint_saver.save_checkpoint(sd, checkpoint_name)

    def _setup_data(self):
        config = self.config
        log(f"instantiating core datamodule <{config.core_dm._target_}>")
        self.core_dm = instantiate(config.core_dm)
        self.core_dm.setup()

        if self.config.patch_dm is not None:
            log(f"instantiating patch datamodule <{config.patch_dm._target_}>")
            self.patch_dm = instantiate(config.patch_dm)
            self.patch_dm.setup()
        else:
            self.config.patch_dm = None

    def _setup_models(self):
        config = self.config
        log(
            f"Instantiating pretrained feature extractor <{config.feat_extractor._target_}>."
        )
        self.feature_extractor = instantiate(config.feat_extractor)
        self.feature_extractor.to(self.config.device)
        self.feature_extractor.eval()
        self.model_state.register_stateful("feature_extractor", self.feature_extractor)
        assert isinstance(
            self.feature_extractor, FeatureExtractionProtocol
        ), f"Feature extractor must implement feature extraction protocol."

        log(
            f"Instantiating sequence classification model <{config.seq_model._target_}>."
        )
        self.seq_model = instantiate(config.seq_model)
        self.seq_model.to(self.config.device)
        if isinstance(self.seq_model, MustFitLatentSpace):
            self._fit_latent_space()
        self.model_state.register_stateful("seq_model", self.seq_model)

        features_dim = self.seq_model.features_dim
        if self.config.use_psa_as_feature:
            features_dim = features_dim + 1

        self.linear_layer = torch.nn.Linear(features_dim, 2).to(self.config.device)
        self.model_state.register_stateful("linear_layer", self.linear_layer)

        # also save config in model state
        config = {
            "feat_extractor": config_to_dict(self.config.feat_extractor),
            "seq_model": config_to_dict(self.config.seq_model),
        }
        config = StatefulAttribute(config)
        self.model_state.register_stateful("config", config)
        self.experiment_state.register_stateful("models", self.model_state)

    def _setup_optimization(self):

        config = self.config
        assert isinstance(self.seq_model, torch.nn.Module)
        [self.opt], [self.sched] = configure_optimizers(
            [
                {"params": self.seq_model.parameters()},
                {"params": self.linear_layer.parameters()},
            ],
            self.config.opt_config,
            num_scheduling_steps_per_epoch=len(self.core_dm.train_dataloader())
            / self.config.training_options.accumulate_grad_batches,
            num_epochs=self.config.training_options.num_epochs,
        )
        self.experiment_state.register_stateful("opt", self.opt)
        if self.sched is not None:
            self.sched = self.sched["scheduler"]
            self.experiment_state.register_stateful("sched", self.sched)

        log(
            f"Using optimizer <{self.opt.__class__.__name__}> and scheduler <{self.sched.__class__.__name__}> for main classfifier!"
        )

        self.feat_extractor_sched = None
        self.feat_extractor_opt = None
        if self.config.feat_extractor_opt_config is not None:
            [self.feat_extractor_opt], [
                self.feat_extractor_sched
            ] = configure_optimizers(
                [{"params": self.feature_extractor.parameters()}],
                self.config.feat_extractor_opt_config,
                num_scheduling_steps_per_epoch=len(self.core_dm.train_dataloader())
                / self.config.training_options.accumulate_grad_batches,
                num_epochs=self.config.training_options.num_epochs
                - config.start_training_feat_extractor_at_epoch,
            )
            self.experiment_state.register_stateful(
                "feat_extractor_opt", self.feat_extractor_opt
            )
            if self.feat_extractor_sched is not None:
                self.feat_extractor_sched = self.feat_extractor_sched["scheduler"]
                self.experiment_state.register_stateful(
                    "sched", self.feat_extractor_sched
                )
        log(
            f"Using optimizer <{self.feat_extractor_opt.__class__.__name__}> and scheduler <{self.feat_extractor_sched.__class__.__name__}> for feature extractor!"
        )

    def _setup_logging(self):
        config = self.config
        wandb.init(
            "train",
            project=self.config.logging_options.project,
            id=self.config.logging_options.run_id,
            name=config.logging_options.run_name,
            entity=config.logging_options.entity,
            config=config_to_dict(config),
        )

    @property
    def _should_train_feature_extractor(self):
        return (
            self.config.feat_extractor_opt_config is not None
            and self.counters.counters["epoch"]
            >= self.config.start_training_feat_extractor_at_epoch
        )

    def train_step(self, batch):
        patch, pos, label, metadata = batch
        psa = (metadata["psa"].to(self.config.device).float() - 6.8) / 5
        psa = torch.nan_to_num(psa)
        patch = patch.to(self.config.device)[0]
        pos = pos.to(self.config.device)[0]
        label = label.to(self.config.device)[0].long()

        with nullcontext() if self._should_train_feature_extractor else torch.no_grad():
            feats = self.feature_extractor.get_features(patch)

        embs = self.seq_model(feats, pos[:, 0], pos[:, 2])

        if self.config.use_psa_as_feature:
            embs = torch.concat([embs, psa], dim=-1)

        logits = self.linear_layer(embs)
        loss = (
            torch.nn.functional.cross_entropy(logits, label)
            / self.config.training_options.accumulate_grad_batches
        )
        if self.counters.counters["train_step"] % 100 == 0:
            self.log_dict(
                {
                    "train_loss": loss.item(),
                    "global_step": self.counters.counters["global_step"],
                }
            )
        loss.backward()
        if (
            self.counters.counters["train_step"]
            % self.config.training_options.accumulate_grad_batches
            == 0
        ):
            self.opt_step()

        patch, pos, label, metadata = batch
        self._train_output_collector.collect_batch(
            {
                "preds": logits.unsqueeze(0).softmax(-1),
                "labels": label.long(),
                **metadata,
            }
        )

        self.counters.increment("train_step")
        self.counters.increment("global_step")

    def eval_step(self, batch):
        patch, pos, label, metadata = batch
        psa = (metadata["psa"].to(self.config.device).float() - 6.8) / 5
        psa = torch.nan_to_num(psa)
        patch = patch.to(self.config.device)[0]
        pos = pos.to(self.config.device)[0]
        label = label.to(self.config.device)[0].long()

        with torch.no_grad():
            # feats = pretrained_vicreg.get_features(patch)
            # feats = pca(feats)
            feats = self.feature_extractor.get_features(patch)
            embs = self.seq_model(feats, pos[:, 0], pos[:, 2])
            if self.config.use_psa_as_feature:
                embs = torch.concat([embs, psa], dim=-1)
            logits = self.linear_layer(embs)

        patch, pos, label, metadata = batch
        self._eval_output_collector.collect_batch(
            {
                "preds": logits.unsqueeze(0).softmax(-1),
                "labels": label.long(),
                **metadata,
            }
        )

        self.counters.increment("global_step")

    def train_epoch(self):
        self.seq_model.train()
        self._train_output_collector.reset()
        for batch in tqdm(
            self.core_dm.train_dataloader(),
            desc=f"Training Epoch {self.counters.counters['epoch']}",
        ):
            self.train_step(batch)
        epoch_output = self._train_output_collector.compute()
        return epoch_output

    def eval_epoch(self, val_or_test="val"):
        self._eval_output_collector.reset()
        self.seq_model.eval()
        self.feature_extractor.eval()
        loader = (
            self.core_dm.val_dataloader()
            if val_or_test == "val"
            else self.core_dm.test_dataloader()
        )
        for batch in tqdm(
            loader, desc=f"{val_or_test} epoch {self.counters.counters['epoch']}"
        ):
            self.eval_step(batch)
        return self._eval_output_collector.compute()

    def epoch(self):

        log(f"Starting Epoch {self.counters.counters['epoch']}")
        # save snapshot at start of epoch
        self._save_experiment_state()

        metrics = {}
        out = self.train_epoch()
        metrics.update(add_prefix(self.compute_metrics(out), "train", sep="/"))
        out = self.eval_epoch("val")
        metrics.update(add_prefix(self.compute_metrics(out), "val", sep="/"))
        out = self.eval_epoch("test")
        metrics.update(add_prefix(self.compute_metrics(out), "test", sep="/"))
        metrics["epoch"] = self.counters.counters["epoch"]
        metrics["global_step"] = self.counters.counters["global_step"]
        self.log_dict(metrics)

        self.current_metrics.value = metrics
        score = metrics[self.config.checkpoint_options.monitored_metric]
        self.score_improvement_monitor.update(score)

        log(f"Epoch completed with validation score {score:.2f}.")

        self.log_dict(self.best_metrics.value)

        self.counters.increment("epoch")

    def compute_metrics(self, out):
        metrics = compute_center_and_macro_metrics(out, base_metrics)
        return metrics

    def _pre_compute_features(self, loader):
        out = []
        for batch in tqdm(loader):
            patch, *_ = batch
            patch = patch.to(self.config.device)[0]
            with torch.no_grad():
                feats = self.feature_extractor.get_features(patch)
            batch = feats, *_
            out.append(batch)
        return out

    def opt_step(self):
        self.opt.step()
        self.opt.zero_grad()
        if self.sched is not None:
            self.sched.step()
            self.log_dict(
                {
                    "main_lr": self.sched.get_last_lr()[-1],
                }
            )
        if self._should_train_feature_extractor:
            if not hasattr(self, "_has_given_feat_extractor_start_training_msg"):
                log(
                    f"Epoch {self.counters.counters['epoch']} reached. Started feature extractor training!"
                )
                self._has_given_feat_extractor_start_training_msg = True

            assert self.feat_extractor_opt is not None
            self.feat_extractor_opt.step()
            self.feat_extractor_opt.zero_grad()
            if self.feat_extractor_sched is not None:
                self.feat_extractor_sched.step()
                self.log_dict(
                    {
                        "backbone_lr": self.feat_extractor_sched.get_last_lr()[-1],
                    }
                )

        self.counters.increment("opt_step")

    def log_dict(self, dict):
        dict.update(self.counters.counters)
        wandb.log(dict)

    def _save_experiment_state(self):
        # if not os.path.isdir(self.config.checkpoint_options.checkpoint_dir):
        #    os.mkdir(self.config.checkpoint_options.checkpoint_dir)
        sd = self.experiment_state.state_dict()
        self.experiment_checkpoint_saver.save_checkpoint(sd, "experiment_state.ckpt")

    def _load_experiment_state(self, sd):
        self.experiment_state.load_state_dict(sd)

    def _fit_latent_space(self):

        if self.patch_dm is None:
            raise ValueError(
                f"Must be configured with patch datamodule to fit latent space."
            )
        all_train_patch_latents = []
        self.feature_extractor.eval()
        for patch, *_ in tqdm(
            self.patch_dm.train_dataloader(),
            desc="Extracting latents from all training patches",
        ):
            patch = patch.to(self.config.device)
            with torch.no_grad():
                feats = self.feature_extractor.get_features(patch)
            all_train_patch_latents.append(feats)
        all_train_patch_latents = torch.concat(all_train_patch_latents, 0)
        log("Fitting sequence model to latent space")
        self.seq_model.fit_latent_space(all_train_patch_latents)

    def train(self):
        while self.counters.counters["epoch"] < self.config.training_options.num_epochs:
            if self.early_stopping_monitor.should_early_stop():
                break
            self.epoch()

    def run(self):
        ckpt_path = self.experiment_checkpoint_saver.get_fname("experiment_state.ckpt")
        if os.path.exists(ckpt_path):
            log(f"Found checkpoint at {ckpt_path}. Loading...")
            sd = torch.load(ckpt_path)
            self._load_experiment_state(sd)

        log(f"Starting training!")
        self.train()
        self.log_dict(self.best_metrics.value)
        return self.best_metrics.value["best_test/micro_avg_auroc"]
