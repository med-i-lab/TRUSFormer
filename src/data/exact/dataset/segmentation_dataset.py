from ..core import Core
import os
from torch.utils.data import Dataset
from tqdm import tqdm


ANATOMICAL_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]


ANATOMICAL_LOCATIONS_INV = {name: idx for idx, name in enumerate(ANATOMICAL_LOCATIONS)}
from src.data import data_dir


class BModeSegmentationDataset(Dataset):
    def __init__(self, cores_list, transform=None, info_dict_transform=None):
        self.cores_list = cores_list

        self.transform = transform
        self.info_dict_transform = info_dict_transform

        self.cores = [
            Core.create_core(core_specifier) for core_specifier in self.cores_list
        ]
        for core in tqdm(
            self.cores, desc="Downloading b_modes and prostate masks if necessary"
        ):
            core: Core
            if not core.bmode_is_downloaded:
                core.download_bmode()
            if not core.prostate_mask_is_downloaded:
                core.download_prostate_mask()

        from exactvu.resources import metadata

        self.metadata = metadata().query("core_specifier in @self.cores_list")

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, idx):

        core_specifier = self.cores_list[idx]
        core = self.cores[idx]
        bmode = core.bmode
        seg = core.prostate_mask

        if self.transform:
            bmode, seg = self.transform(bmode, seg)

        metadata = core.metadata
        if self.info_dict_transform:
            metadata = self.info_dict_transform(metadata)

        return bmode, seg, metadata

    def plot_item(self, idx): 
        # temporarily change transform to none
        transform = self.transform
        self.transform = None

        import matplotlib.pyplot as plt
        bmode, seg, metadata = self[idx]

        # show bmode and seg side by side
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(bmode)
        ax[1].imshow(seg)

        # set title
        title = f"Core {metadata['core_specifier']}"
        title += f", {metadata['anatomical_location']}"
        title += f", {metadata['grade']}"

        fig.suptitle(title)

        # reset transform
        self.transform = transform