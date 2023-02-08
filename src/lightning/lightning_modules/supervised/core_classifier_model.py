from contextlib import nullcontext
from dataclasses import dataclass
import itertools
from typing import Any, Optional, Tuple

import torchmetrics

from ..evaluation_base import EvaluationBase
from omegaconf import DictConfig, dictconfig
from copy import copy, deepcopy
from ....modeling.optimizer_factory import (
    OptimizerConfig,
    SchedulerOptions,
    configure_optimizers,
)
import logging
from ....typing import FeatureExtractionProtocol
from ....modeling.bert import TransformerConfig
from ....modeling.positional_embedding import GridPositionEmbedder2d
from torch.nn import Linear
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning
import torch
from einops import rearrange, repeat
from torchmetrics import AveragePrecision


NUM_CLASSES = 2


logger = logging.getLogger(__name__)


@dataclass
class TransformerCoreClassifierConfig:

    feature_extractor: Any
    sequence_model: Any = TransformerConfig()
    feature_extractor_lr: Optional[float] = None
    start_trainging_feature_extractor_at_epoch: Optional[int] = 10
    opt_config: OptimizerConfig = OptimizerConfig(
        scheduler_options=SchedulerOptions(scheduler_interval="epoch")
    )
    use_pos_embeddings: bool = True
    pos_embeddings_grid_shape: Tuple[int, int] = (28, 46)
    loss_weights: Optional[Tuple[int, int]] = None

    _target_: str = __name__ + ".TransformerCoreClassifier"
    _recursive_: bool = False


@dataclass
class TransformerCoreClassifierWithLearnableFeatureReductionConfig(
    TransformerCoreClassifierConfig
):

    feature_reducer_opt_cfg: OptimizerConfig = OptimizerConfig(
        scheduler_options=SchedulerOptions(scheduler_interval="epoch")
    )

    _target_: str = __name__ + ".TransformerCoreClassifierWithLearnableFeatureReduction"


class TransformerCoreClassifier(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: DictConfig,
        sequence_model: Any = TransformerConfig(),
        feature_extractor_lr: Optional[float] = None,
        start_trainging_feature_extractor_at_epoch: Optional[int] = 10,
        opt_config: OptimizerConfig = OptimizerConfig(
            scheduler_options=SchedulerOptions(scheduler_interval="epoch")
        ),
        use_pos_embeddings: bool = True,
        pos_embeddings_grid_shape=(28, 46),
        loss_weights=None,
    ):

        super().__init__()
        self.save_hyperparameters()

        from hydra.utils import instantiate

        logger.info(f"Instantiating backbone model <{feature_extractor._target_}>!")
        self.feature_extractor = instantiate(feature_extractor)
        assert isinstance(
            self.feature_extractor, FeatureExtractionProtocol
        ), f"Feature extractor must follow the FeatureExtractionProtocol"

        logger.info(f"Instantiating sequence model <{sequence_model._target_}>!")
        self.sequence_model = instantiate(sequence_model)
        assert isinstance(
            self.feature_extractor, FeatureExtractionProtocol
        ), f"Feature extractor must follow the FeatureExtractionProtocol"

        self.hidden_size = sequence_model.hidden_size

        self.feature_reducer = Linear(
            self.feature_extractor.features_dim, self.hidden_size
        )

        if use_pos_embeddings:
            self.pos_embeddings = GridPositionEmbedder2d(
                self.hidden_size, pos_embeddings_grid_shape
            )
        else:
            self.pos_embeddings = None

        self.cls_token = torch.nn.parameter.Parameter(torch.randn(1, self.hidden_size))

        self.classification_head = torch.nn.Linear(self.hidden_size, NUM_CLASSES)

        self.main_opt_config = opt_config
        assert (
            self.main_opt_config.scheduler_options.scheduler_interval == "epoch"
        ), f"This lightning module does not work with stepwise lr scheduling because it does not know the number of steps."
        self.feature_extractor_lr = feature_extractor_lr
        self.start_training_feature_extractor_at_epoch = (
            start_trainging_feature_extractor_at_epoch
        )

        self.loss_weight = loss_weights

        self.train_avg_prec = AveragePrecision(num_classes=2)
        self.val_avg_prec = AveragePrecision(num_classes=2)
        self.test_avg_prec = AveragePrecision(num_classes=2)

    def on_train_epoch_start(self) -> None:
        self._should_train_feature_extractor = (
            self.feature_extractor_lr is not None
        ) and (
            self.start_training_feature_extractor_at_epoch is None
            or self.current_epoch > self.start_training_feature_extractor_at_epoch
        )

        if not self._should_train_feature_extractor:
            self.feature_extractor.eval()

    def configure_optimizers(self):

        params1 = [
            {
                "name": "core_classifier",
                "params": itertools.chain(
                    self.sequence_model.parameters(),
                    [self.cls_token],
                    self.pos_embeddings.parameters()
                    if self.pos_embeddings is not None
                    else [],
                    self.classification_head.parameters(),
                ),
            },
            {
                "name": "feature_reducer",
                "params": self.feature_reducer.parameters(),
            },
        ]

        if self.feature_extractor_lr is not None:
            params1.append(
                {
                    "name": "feature_extractor",
                    "params": self.feature_extractor.parameters(),
                    "lr": self.feature_extractor_lr,
                },
            )

        return configure_optimizers(
            params1,
            self.main_opt_config,
            1,
            self.num_epochs,
        )

    @property
    def num_epochs(self):
        return self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        out = self._main_shared_step(batch, batch_idx, True)
        self.train_avg_prec(out["logits_for_core"], out["labels"])
        self.log("train_core_avg_prec", self.train_avg_prec)
        return out

    def _main_shared_step(self, batch, batch_idx, train=True):

        patches, pos, labels, metadata = batch

        labels = labels.long()

        n_cores, core_len, n_chans, h, w = patches.shape
        patches = patches[0]

        with torch.no_grad() if (
            self._should_train_feature_extractor and not train
        ) else nullcontext():
            features = self.feature_extractor.get_features(patches)

        core_len, n_feats = features.shape

        with torch.no_grad() if not train else nullcontext():
            reduced_features = self.feature_reducer(features)

        core_len, n_feats = reduced_features.shape
        assert n_feats == self.hidden_size

        if self.pos_embeddings is not None:
            pos_embeddings = self.pos_embeddings(pos[0, ..., 0], pos[0, ..., 2])
            reduced_features = reduced_features + pos_embeddings

        # class token is first row
        reduced_features = torch.concat([self.cls_token, reduced_features], dim=0)

        with torch.no_grad() if not train else nullcontext():
            # sequence model expects batch dim first, we need to add it
            contextualized_features = self.sequence_model(
                reduced_features.unsqueeze(0)
            ).last_hidden_state[0]

        contextualized_cls_token = contextualized_features[0]

        logits_for_core = self.classification_head(contextualized_cls_token).unsqueeze(
            0
        )

        loss = torch.nn.functional.cross_entropy(
            logits_for_core, labels, weight=self.loss_weight
        )

        return {
            "loss": loss,
            "logits_for_core": logits_for_core.detach(),
            "labels": labels,
            "metadata": metadata,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:

            # validation
            out = self._main_shared_step(batch, batch_idx, False)
            self.val_avg_prec(out["logits_for_core"], out["labels"])
            self.log("val_core_avg_prec", self.val_avg_prec)

        else:
            # test
            out = self._main_shared_step(batch, batch_idx, False)
            self.test_avg_prec(out["logits_for_core"], out["labels"])
            self.log("test_core_avg_prec", self.test_avg_prec)

        return out

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        out = self._main_shared_step(batch, batch_idx, False)
        self.test_avg_prec(out["logits_for_core"], out["labels"])
        self.log("test_core_avg_prec", self.test_avg_prec)

        return out


class TransformerCoreClassifierWithLearnableFeatureReduction(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: DictConfig,
        sequence_model: Any = TransformerConfig(),
        feature_extractor_lr: Optional[float] = None,
        start_trainging_feature_extractor_at_epoch: Optional[int] = 10,
        opt_config: OptimizerConfig = OptimizerConfig(
            scheduler_options=SchedulerOptions(scheduler_interval="epoch")
        ),
        feature_reducer_opt_cfg: OptimizerConfig = OptimizerConfig(
            scheduler_options=SchedulerOptions(scheduler_interval="epoch")
        ),
        use_pos_embeddings: bool = True,
        pos_embeddings_grid_shape=(28, 46),
        loss_weights=None,
    ):

        super().__init__()
        self.save_hyperparameters()

        from hydra.utils import instantiate

        logger.info(f"Instantiating backbone model <{feature_extractor._target_}>!")
        self.feature_extractor = instantiate(feature_extractor)
        assert isinstance(
            self.feature_extractor, FeatureExtractionProtocol
        ), f"Feature extractor must follow the FeatureExtractionProtocol"

        logger.info(f"Instantiating sequence model <{sequence_model._target_}>!")
        self.sequence_model = instantiate(sequence_model)
        assert isinstance(
            self.feature_extractor, FeatureExtractionProtocol
        ), f"Feature extractor must follow the FeatureExtractionProtocol"

        self.hidden_size = sequence_model.hidden_size

        self.feature_reducer = Linear(
            self.feature_extractor.features_dim, self.hidden_size
        )
        self.feature_reconstructor = Linear(
            self.hidden_size, self.feature_extractor.features_dim
        )

        if use_pos_embeddings:
            self.pos_embeddings = GridPositionEmbedder2d(
                self.hidden_size, pos_embeddings_grid_shape
            )
        else:
            self.pos_embeddings = None

        self.cls_token = torch.nn.parameter.Parameter(torch.randn(1, self.hidden_size))

        self.classification_head = torch.nn.Linear(self.hidden_size, NUM_CLASSES)

        self.main_opt_config = opt_config
        assert (
            self.main_opt_config.scheduler_options.scheduler_interval == "epoch"
        ), f"This lightning module does not work with stepwise lr scheduling because it does not know the number of steps."
        self.feature_extractor_lr = feature_extractor_lr
        self.start_training_feature_extractor_at_epoch = (
            start_trainging_feature_extractor_at_epoch
        )
        self.feature_reducer_opt_cfg = feature_reducer_opt_cfg
        assert (
            self.feature_reducer_opt_cfg.scheduler_options.scheduler_interval == "epoch"
        ), f"This lightning module does not work with stepwise lr scheduling because it does not know the number of steps."

        self.loss_weight = loss_weights

        self.train_avg_prec = AveragePrecision(num_classes=2)
        self.val_avg_prec = AveragePrecision(num_classes=2)
        self.test_avg_prec = AveragePrecision(num_classes=2)

    @property
    def num_epochs(self):
        return self.trainer.max_epochs

    def configure_optimizers(self):

        params1 = [
            {
                "name": "core_classifier",
                "params": itertools.chain(
                    self.sequence_model.parameters(),
                    [self.cls_token],
                    self.pos_embeddings.parameters()
                    if self.pos_embeddings is not None
                    else [],
                    self.classification_head.parameters(),
                ),
            },
        ]

        if self.feature_extractor_lr is not None:
            params1.append(
                {
                    "name": "feature_extractor",
                    "params": self.feature_extractor.parameters(),
                    "lr": self.feature_extractor_lr,
                },
            )

        (opt1,), (sched1,) = configure_optimizers(
            params1,
            self.main_opt_config,
            1,
            self.num_epochs,
        )

        params2 = [
            {
                "name": "feature_reducer",
                "params": itertools.chain(
                    self.feature_reducer.parameters(),
                    self.feature_reconstructor.parameters(),
                ),
            }
        ]

        (opt2,), (sched2,) = configure_optimizers(
            params2, self.feature_reducer_opt_cfg, 1, self.num_epochs
        )

        return [opt1, opt2], [sched1, sched2]

    def on_train_epoch_start(self) -> None:
        self._should_train_feature_extractor = (
            self.feature_extractor_lr is not None
        ) and (
            self.start_training_feature_extractor_at_epoch is None
            or self.current_epoch > self.start_training_feature_extractor_at_epoch
        )

        if not self._should_train_feature_extractor:
            self.feature_extractor.eval()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            out = self._main_shared_step(batch, batch_idx, True)
            self.train_avg_prec(out["logits_for_core"], out["labels"])
            self.log("train_core_avg_prec", self.train_avg_prec)
            return out
        else:
            return self._feature_reducer_training_step()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:

            # validation
            out = self._main_shared_step(batch, batch_idx, False)
            self.val_avg_prec(out["logits_for_core"], out["labels"])
            self.log("val_core_avg_prec", self.val_avg_prec)

        else:
            # test
            out = self._main_shared_step(batch, batch_idx, False)
            self.test_avg_prec(out["logits_for_core"], out["labels"])
            self.log("test_core_avg_prec", self.test_avg_prec)

    def _main_shared_step(self, batch, batch_idx, train=True):

        patches, pos, labels, metadata = batch

        labels = labels.long()

        n_cores, core_len, n_chans, h, w = patches.shape
        patches = patches[0]

        with torch.no_grad() if (
            self._should_train_feature_extractor and not train
        ) else nullcontext():
            features = self.feature_extractor.get_features(patches)

        core_len, n_feats = features.shape
        self._cached_features = torch.clone(features).detach()

        with torch.no_grad():
            reduced_features = self.feature_reducer(features)

        core_len, n_feats = reduced_features.shape
        assert n_feats == self.hidden_size

        if self.pos_embeddings is not None:
            pos_embeddings = self.pos_embeddings(pos[0, ..., 0], pos[0, ..., 2])
            reduced_features = reduced_features + pos_embeddings

        # class token is first row
        reduced_features = torch.concat([self.cls_token, reduced_features], dim=0)

        with torch.no_grad() if not train else nullcontext():
            # sequence model expects batch dim first, we need to add it
            contextualized_features = self.sequence_model(
                reduced_features.unsqueeze(0)
            ).last_hidden_state[0]

        contextualized_cls_token = contextualized_features[0]

        logits_for_core = self.classification_head(contextualized_cls_token).unsqueeze(
            0
        )

        loss = torch.nn.functional.cross_entropy(
            logits_for_core, labels, weight=self.loss_weight
        )

        return {
            "loss": loss,
            "logits_for_core": logits_for_core.detach(),
            "labels": labels,
            "metadata": metadata,
        }

    def _feature_reducer_training_step(self):
        features = self._cached_features
        reduced_features = self.feature_reducer(features)
        reconstructed_features = self.feature_reconstructor(reduced_features)

        mse_loss = torch.nn.functional.mse_loss(features, reconstructed_features)
        self.log("feature_reducer_mse", mse_loss)

        return mse_loss
