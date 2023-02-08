from typing import Any, Protocol, runtime_checkable, Optional
import torch


@runtime_checkable
class FeatureExtractionProtocol(Protocol):

    features_dim: int

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class SupportsLoadingPretrainingCheckpoint(Protocol):
    def load_from_pretraining_ckpt(self, ckpt: str):
        ...


@runtime_checkable
class SupportsDatasetNameConvention(Protocol):
    train_ds: Optional[Any]


@runtime_checkable
class Stateful(Protocol):
    def load_state_dict(self, state_dict):
        ...

    def state_dict(self, state_dict):
        ...
