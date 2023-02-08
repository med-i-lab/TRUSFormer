from dataclasses import dataclass
import logging
import os

import torch
from matplotlib.pyplot import cla

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from typing import List, Optional
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ...utils import driver as utils
from ...typing import SupportsLoadingPretrainingCheckpoint

# from pytorch_lightning.callbacks import ModelCheckpoint, Checkpoint


log = utils.get_logger(__name__)


@dataclass
class ExperimentResults:
    score: Optional[float]
    ckpt: Optional[str]


class TrainingDriver:
    def __init__(self, config: DictConfig):
        self.config = config

    def setup(self):
        """
        Sets up models, datamodule, loggers, callbacks, etc.
        """

        # Set seed for random number generators in pytorch, numpy and python.random
        if (s := self.config.get("seed")) is not None:
            log.info(f"Seed {s} is being used to set the random state!")
            seed_everything(s, workers=True)

        torch.use_deterministic_algorithms(mode=True)

        self.ckpt_path = self.config.trainer.get("resume_from_checkpoint")

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{self.config.datamodule._target_}>")
        self.datamodule: LightningDataModule = instantiate(self.config.datamodule)

        # Init lightning model
        log.info(f"Instantiating model <{self.config.model._target_}>")
        self.lit_module: LightningModule = instantiate(self.config.model)

        # Init lightning callbacks
        self.callbacks: List[Callback] = []
        if "callbacks" in self.config:
            for _, cb_conf in self.config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    self.callbacks.append(instantiate(cb_conf))

        # Init lightning loggers
        self.loggers: List[LightningLoggerBase] = []
        if "logger" in self.config:
            for name, lg_conf in self.config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    if "wandb" in name:
                        from pytorch_lightning.loggers import WandbLogger
                        import wandb

                        config = OmegaConf.to_container(
                            self.config, resolve=True, throw_on_missing=True
                        )
                        lg_conf.pop("_target_")
                        self.loggers.append(WandbLogger(**lg_conf, config=config))
                    else:
                        self.loggers.append(instantiate(lg_conf))

        # Init lightning trainer
        log.info(f"Instantiating trainer <{self.config.trainer._target_}>")
        self.trainer: Trainer = instantiate(
            self.config.trainer,
            callbacks=self.callbacks,
            logger=self.loggers,
            _convert_="partial",
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=self.config,
            model=self.lit_module,
            datamodule=self.datamodule,
            trainer=self.trainer,
            callbacks=self.callbacks,
            logger=self.loggers,
        )

    def _find_best_checkpoint_and_score(self):
        # prefer to resume training from online eval checkpoint
        if hasattr(self.trainer, "online_eval_callback"):
            from ..lightning.callbacks.online_evaluation import OnlineEvaluation

            online_eval_callback: OnlineEvaluation = self.trainer.online_eval_callback
            optimized_metric = "val_auroc"
            score = online_eval_callback.best_val_metrics_global["auroc"]
            ckpt_path = online_eval_callback.checkpoint_paths["val_auroc"]
        else:
            assert self.trainer.checkpoint_callback is not None

            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            optimized_metric = self.trainer.checkpoint_callback.monitor
            score = self.trainer.checkpoint_callback.best_model_score

        return ckpt_path, score

    def train(self) -> ExperimentResults:
        # Train the model
        log.info("Starting training!")

        # try and load the last model checkpoint
        ckpt_dir = self.trainer.checkpoint_callback.dirpath
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

        log.info(f"Looking for checkpoint at {ckpt_path}")

        if os.path.exists(ckpt_path):
            log.info(f"Found checkpoint. Resuming Training!")
            last_ckpt = ckpt_path
        else:
            log.info(f"No checkpoint found. Starting training from scratch.")
            last_ckpt = None

        self.trainer.fit(
            model=self.lit_module, datamodule=self.datamodule, ckpt_path=last_ckpt
        )

        # Get metric score for hyperparameter optimization
        optimized_metric = self.config.get("optimized_metric")
        if optimized_metric and optimized_metric not in self.trainer.callback_metrics:
            raise Exception(
                "Metric for hyperparameter optimization not found! "
                "Make sure the `optimized_metric` in `hparams_search` config is correct!"
            )
        score = self.trainer.callback_metrics.get(optimized_metric)

        # Test the model
        if self.config.get("test"):
            checkpoint_path, score = self._find_best_checkpoint_and_score()
            log.info(f"Testing model ckpt at {checkpoint_path}, scored {score}.")
            log.info("Starting testing!")
            self.trainer.test(
                model=self.lit_module,
                datamodule=self.datamodule,
                ckpt_path=checkpoint_path,
            )

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=self.config,
            model=self.lit_module,
            datamodule=self.datamodule,
            trainer=self.trainer,
            callbacks=self.callbacks,
            logger=self.loggers,
        )

        # Print path to best checkpoint
        if not self.config.trainer.get("fast_dev_run") and self.config.get("train"):
            best_ckpt, score = self._find_best_checkpoint_and_score()
            log.info(f"Best model ckpt at {best_ckpt}, scored {score}.")
        else:
            best_ckpt = None

        # Return metric score for hyperparameter optimization
        return ExperimentResults(
            score,
            best_ckpt,
        )

    def run(self):
        self.setup()
        self.train()


class PretrainingDriver(TrainingDriver):
    def __init__(self, config):
        super().__init__(config)
        self.best_ckpt = None
        self.best_score = None

    def run(self):
        self.setup()
        out = self.train()
        self.best_ckpt = out.ckpt
        self.best_score = out.score


class FinetuningDriver(TrainingDriver):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.pretraining_checkpoint = None

    def port(self, prev_experiment):
        if not isinstance(prev_experiment, PretrainingDriver):
            raise RuntimeError(
                f"""attempted to port finetuning from pretraining experiment, but 
                                    {prev_experiment} if not a pretraining experiment."""
            )

        if prev_experiment.best_score is None:
            raise RuntimeError(
                f"""attempted to port finetuning from pretraining experiment, but 
                                    best checkpoint could not be found. has `run()` completed for pretraining experiment?"""
            )

        self.pretraining_checkpoint = prev_experiment.best_ckpt
        # self.pretraining_score = prev_experiment.best_score

    def setup(self):

        super().setup()

        if self.pretraining_checkpoint is not None:
            # should load pretraining checkpoint
            if not isinstance(self.lit_module, SupportsLoadingPretrainingCheckpoint):
                raise RuntimeError(
                    f"""Attempted to load pretraining checkpoint into finetune model,
                    but {self.pretraining_checkpoint.__class__} did not support checkpointing. 
                    """
                )

            self.lit_module.load_from_pretraining_ckpt(self.pretraining_checkpoint)


class PretrainAndFinetuneDriver:
    def __init__(
        self, pretrain_driver: PretrainingDriver, finetune_driver: FinetuningDriver
    ):
        self.pretraining_driver = pretrain_driver
        self.finetune_driver = finetune_driver

    def run(self):
        self.pretraining_driver.setup()
        self.pretraining_driver.train()
        self.finetune_driver.port(self.pretraining_driver)
        self.finetune_driver.setup()
        self.finetune_driver.train()
