from ..image_server import get_ssh
import scp
from torch.utils.data import Dataset
import dotenv
import os
import numpy as np
import sys
import pandas as pd


PATCHES_PATH_ON_SERVER = "/med-i_data/Data/Amoon/HDH_patches/"


class HDHPatchesDataset(Dataset):
    def __init__(
        self, data_dir, core_indices, patch_transform=None, target_transform=None
    ):
        """
        data_dir: path to the directory where the data is stored. If the data is not downloaded,
        it will be downloaded to this directory. A folder named "HDH_patches" will be created in this directory.
        If the HDH_patches is already downloaded, it will be used.
        core_indices: list of core indices to use
        patch_transform: transform to apply to each patch
        target_transform: transform to apply to the label
        """

        self.data_dir = data_dir
        self.patch_transform = patch_transform
        self.target_transform = target_transform

        if not self._data_is_downloaded:
            self._download_data()

        self.all_patch_info = self._get_patch_info()
        self.all_core_info = self._parse_metadata()
        self.core_indices = core_indices
        self.data_table = (
            self.all_patch_info.set_index(["patient_id", "core_num_for_patient"])
            .join(
                self.all_core_info.set_index(["patient_id", "core_num_for_patient"]),
            )
            .query("core_idx in @core_indices")
            .reset_index()
        )

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        row = dict(self.data_table.iloc[index])
        fpath = row.pop("fpath")
        patch = np.load(fpath)
        patch = patch.astype(np.float32)
        label = row.pop("label")
        if self.patch_transform:
            patch = self.patch_transform(patch)
        if self.target_transform:
            label = self.target_transform(label)
        return patch, label, row

    def _get_patch_info(self):
        """Looks through the patches folder and saves the information about each patch in a dataframe"""
        patch_info = []
        for fpath in os.listdir(
            os.path.join(self.data_dir, "HDH_patches", "bm_coor_oriented")
        ):
            if not "data" in fpath:
                continue
            import re

            patient_id, core_id, patch_id = re.match(
                "P(\d+)_c(\d+)_p(\d+)_data.npy", fpath
            ).groups()
            patch_info.append(
                {
                    "patient_id": int(patient_id),
                    "core_num_for_patient": int(core_id),
                    "patch_num_for_core": int(patch_id),
                    "fpath": os.path.join(
                        self.data_dir, "HDH_patches", "bm_coor_oriented", fpath
                    ),
                }
            )
        patch_info = pd.DataFrame(patch_info)
        patch_info = patch_info.sort_values(
            by=["patient_id", "core_num_for_patient", "patch_num_for_core"]
        ).reset_index(drop=True)
        return patch_info

    def _parse_metadata(self):
        path_to_metadata = os.path.join(self.data_dir, "HDH_patches", "meta_data.npz")
        metadata = np.load(path_to_metadata, allow_pickle=True)
        core_infos = []
        for i in range(len(metadata["patient_id"])):
            core_info = {}
            core_info["patient_id"] = metadata["patient_id"][i]
            core_info["core_loc"] = metadata["core_locs"][i]
            core_info["gleason"] = metadata["gleasons"][i]
            core_info["core_idx"] = i
            core_info["core_num_for_patient"] = (
                np.where(metadata["patient_id"] == metadata["patient_id"][i])[0]
                .tolist()
                .index(i)
            )
            core_info["label"] = metadata["labels"][i]
            core_info["involvement"] = metadata["involvements"][i]
            core_info["prostate_volume"] = metadata["pr_volumes"][i]
            core_info["psa"] = metadata["psas"][i]

            core_infos.append(core_info)

        return pd.DataFrame(core_infos)

    @staticmethod
    def default_data_dir():
        dotenv.load_dotenv()
        dir = os.getenv("DATA_DIR")
        if dir is None:
            raise ValueError("DATA_DIR is not defined in .env file")

    @property
    def _data_is_downloaded(self):
        return "HDH_patches" in os.listdir(self.data_dir)

    def _progress(self, filename, size, sent):
        sys.stdout.write(
            "%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100)
        )

    def _download_data(self):
        print(
            "Downloading HDH Patches data. This may take a while... (about 10 minutes)"
        )
        transport = get_ssh().get_transport()
        scp_client = scp.SCPClient(transport, progress=self._progress)
        scp_client.get(PATCHES_PATH_ON_SERVER, self.data_dir, recursive=True)
