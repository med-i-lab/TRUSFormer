_DATASETS = {}


def register_dataset(f):
    _DATASETS[f.__name__] = f
    return f


def create_dataset(name, split="train"):
    return _DATASETS[name](split)


def list_datasets():
    return list(_DATASETS.keys())


@register_dataset
def exact_patches_ssl_all_centers_all_cores_all_patches_v0(split="train"):
    """
    Created by @pfrwilson
    ON 2023-02-04.

    Self-supervised dataset with all patches from all cores.
    normalized, and using crop-like augmentation. (SimCLR style)
    """

    from src.data.exact.splits import Splits
    from src.data import data_dir
    from torchvision import transforms as T
    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )

    def augment_patch(patch):
        return transform(patch).float(), transform(patch).float()

    splits = Splits(
        cohort_specifier="all",
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=False,
    )
    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=False
    )

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    return PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=augment_patch,
    )


@register_dataset
def exact_patches_sl_all_centers_balanced_ndl(split):
    from src.data.exact.splits import Splits, InvolvementThresholdFilter
    from src.data import data_dir
    from torchvision import transforms as T
    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    transform_ = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
        ]
    )
    transform = lambda x: transform_(x).float()

    splits = Splits(
        cohort_specifier=["UVA", "CRCEO", "PCC", "PMCC", "JH"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        merge_test_centers=True,
        merge_val_centers=True,
        undersample_benign_eval=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    return PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
    )

    