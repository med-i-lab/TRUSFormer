"""
Hacky new dataset class which allows finer-grained patch selection
eg. take all patches within the needle region and some but not all
patches outside needle region.
"""

from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional
from ..core import Core, PatchViewConfig, PatchView
from tqdm.auto import tqdm
from src.data.exact.splits import (
    HasProstateMaskFilter,
    Splits,
    InvolvementThresholdFilter,
)
from src.data.dataloader_factory import LoaderConfig, build_dataloader
from src.data.exact.transforms import MultiTransform
from pytorch_lightning import LightningDataModule
import torch


@dataclass
class PatchSelectionOptions:
    patch_size_mm: tuple = 5, 5
    patch_strides_mm: tuple = 1, 1
    needle_region_overlap_threshold: float = 0.66
    prostate_region_overlap_threshold: float = 1.0
    discard_inside_needle: Optional[float] = None
    discard_inside_prostate: Optional[float] = None
    discard_outside_prostate: Optional[float] = None
    selection_seed = 0


class RFPatchesDataset(Dataset):
    def __init__(
        self,
        core_specifiers: list[Core],
        transform,
        patch_selection_options=PatchSelectionOptions(),
    ):
        self.core_specifiers = core_specifiers
        self.transform = transform
        self.patch_selection_options = patch_selection_options

        self._cores = []
        self._patch_views = []
        self._labels = []
        for core_specifier in tqdm(self.core_specifiers, desc="selecting patches"):

            core = Core.create_core(core_specifier)
            self._cores.append(core)

            patch_view = core.get_patch_view_from_config(
                PatchViewConfig(
                    patch_size=self.patch_selection_options.patch_size_mm,
                    patch_strides=self.patch_selection_options.patch_strides_mm,
                    needle_region_only=False,
                    prostate_region_only=False,
                    return_extras=("positions", "mask_intersections"),
                )
            )

            idx_inside_needle_and_prostate = []
            idx_in_prostate_not_needle = []
            idx_outside_prostate_and_needle = []
            for idx in range(len(patch_view)):
                mask_intersections = patch_view.mask_intersections[idx]
                if (
                    mask_intersections["prostate"]
                    < self.patch_selection_options.prostate_region_overlap_threshold
                ):
                    idx_outside_prostate_and_needle.append(idx)
                elif (
                    mask_intersections["needle"]
                    < self.patch_selection_options.needle_region_overlap_threshold
                ):
                    idx_in_prostate_not_needle.append(idx)
                else:
                    idx_inside_needle_and_prostate.append(idx)

            # now we downsample the patches:
            idx_to_keep = []
            import random

            rng = random.Random(self.patch_selection_options.selection_seed)
            if self.patch_selection_options.discard_inside_needle is not None:
                idx_to_keep.extend(
                    rng.sample(
                        idx_inside_needle_and_prostate,
                        int(
                            (1 - self.patch_selection_options.discard_inside_needle)
                            * len(idx_inside_needle_and_prostate)
                        ),
                    )
                )
            else:
                idx_to_keep.extend(idx_inside_needle_and_prostate)
            if self.patch_selection_options.discard_inside_prostate is not None:
                idx_to_keep.extend(
                    rng.sample(
                        idx_in_prostate_not_needle,
                        int(
                            (1 - self.patch_selection_options.discard_inside_prostate)
                            * len(idx_in_prostate_not_needle)
                        ),
                    )
                )
            else:
                idx_to_keep.extend(idx_in_prostate_not_needle)
            if self.patch_selection_options.discard_outside_prostate is not None:
                idx_to_keep.extend(
                    rng.sample(
                        idx_outside_prostate_and_needle,
                        int(
                            (1 - self.patch_selection_options.discard_outside_prostate)
                            * len(idx_outside_prostate_and_needle)
                        ),
                    )
                )
            else:
                idx_to_keep.extend(idx_outside_prostate_and_needle)

            # now create a new patch_view
            new_patch_view = PatchView(
                patch_view.base_grid,
                [patch_view.patch_positions[idx] for idx in idx_to_keep],
                [patch_view.mask_intersections[idx] for idx in idx_to_keep],
                return_extras=("positions", "mask_intersections"),
            )

            self._patch_views.append(new_patch_view)
            self._labels.extend(
                [core.metadata["grade"] != "Benign"] * len(new_patch_view)
            )

        # now create a mapping from index to patch and core
        self.idx_to_patch_and_core_idx = []
        for core_idx, core in enumerate(self._cores):
            for patch_idx in range(len(self._patch_views[core_idx])):
                self.idx_to_patch_and_core_idx.append((core_idx, patch_idx))

    def __len__(self):
        return len(self.idx_to_patch_and_core_idx)

    def __getitem__(self, idx):
        core_idx, patch_idx = self.idx_to_patch_and_core_idx[idx]
        core = self._cores[core_idx]
        patch, pos, mask_intersections = self._patch_views[core_idx][patch_idx]

        pos = torch.tensor(pos).long()
        label = self._labels[idx]
        label = torch.tensor(label).long()

        if self.transform is not None:
            patch = self.transform(patch)

        additional_info = core.metadata
        additional_info.update(
            {f"{k}_overlap": v for k, v in mask_intersections.items()}
        )

        return patch, pos, label, additional_info


class RFPatchesDataModule(LightningDataModule):
    def __init__(
        self,
        splits: Splits,
        train_transform,
        eval_transform,
        ssl_mode=True,
        loader_config=LoaderConfig(),
        minimum_involvement=0.4,
        patch_selection_options: PatchSelectionOptions = PatchSelectionOptions(),
    ):
        super().__init__()
        self.splits = splits
        self.splits.apply_filters(
            HasProstateMaskFilter(), InvolvementThresholdFilter(minimum_involvement)
        )
        self.loader_config = loader_config
        self.ssl_mode = ssl_mode
        self.train_transform = train_transform
        self.train_transform = (
            MultiTransform(self.train_transform, self.train_transform)
            if ssl_mode
            else self.train_transform
        )
        self.eval_transform = eval_transform
        self.eval_transform = (
            MultiTransform(self.eval_transform, self.eval_transform)
            if ssl_mode
            else self.eval_transform
        )
        self.patch_selection_options = patch_selection_options

    def setup(self, stage=None):
        self.train_ds = RFPatchesDataset(
            self.splits.get_train(),
            self.train_transform,
            self.patch_selection_options,
        )
        self.val_ds = RFPatchesDataset(
            self.splits.get_val(),
            self.eval_transform,
            self.patch_selection_options,
        )
        self.test_ds = RFPatchesDataset(
            self.splits.get_test(),
            self.eval_transform,
            self.patch_selection_options,
        )

        self.dataset_info = [
            {
                "split": "train",
                "num_patches": len(self.train_ds),
                "num_cores": len(self.train_ds._cores),
            },
            {
                "split": "val",
                "num_patches": len(self.val_ds),
                "num_cores": len(self.val_ds._cores),
            },
            {
                "split": "test",
                "num_patches": len(self.test_ds),
                "num_cores": len(self.test_ds._cores),
            },
        ]

        import pandas as pd

        self.dataset_info = pd.DataFrame(self.dataset_info).T

    def train_dataloader(self):
        return build_dataloader(
            self.train_ds, self.train_ds._labels, self.loader_config, True
        )

    def val_dataloader(self):
        return build_dataloader(
            self.val_ds,
            self.val_ds._labels,
            self.loader_config,
            False,
        )

    def test_dataloader(self):
        return build_dataloader(
            self.test_ds, self.test_ds._labels, self.loader_config, False
        )
