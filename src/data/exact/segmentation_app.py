import argparse
from typing import Optional

try:
    import pyroi
except:
    raise ModuleNotFoundError(
        f"to use the segmentaion app, pyroi is needed (https://github.com/pfrwilson/pyroi)"
    )

from pyroi.backend import SegmentationBackend
from tqdm import tqdm
import logging

from .server.segmentation import (
    list_available_prostate_segmentations,
    upload_prostate_mask,
)
from src.data import data_dir
from .core import Core
import os
from os.path import join
import numpy as np
from .preprocessing import to_bmode
from PIL import Image

logger = logging.getLogger("Segmentation App")


def setup_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


class ExactVuSegmentationBackend(SegmentationBackend):
    def __init__(self, core_specifiers: list[str], overwrite=False):

        self._setup_workdir()
        available_segmentations = list_available_prostate_segmentations()

        if not overwrite:
            self.core_specifiers = [
                specifier
                for specifier in core_specifiers
                if specifier not in available_segmentations
            ]
        else:
            self.core_specifiers = core_specifiers

        self.overwrite = overwrite

    def _setup_workdir(self):
        self.workdir = join(os.path.expanduser("~"), ".exact_roi")
        setup_dir(self.workdir)
        setup_dir(join(self.workdir, "bmodes"))
        setup_dir(join(self.workdir, "rois"))

    def get_path_to_roi(self, id):
        return os.path.join(self.workdir, "rois", f"{id}.png")

    def roi_is_saved(self, id) -> bool:
        return os.path.exists(self.get_path_to_roi(id))

    def setup(self):

        self.cores = [
            Core(specifier, os.path.join(data_dir(), "cores_dataset"))
            for specifier in self.core_specifiers
        ]

        for core in tqdm(self.cores, "downloading necessary bmodes"):
            if core.bmode is None:
                core.download_bmode()
            self._preprocess_and_store_bmode(core.bmode, core.specifier)

    def _preprocess_and_store_bmode(self, bmode, core_specifier):
        from skimage.transform import resize
        from skimage.exposure import rescale_intensity
        from skimage.color import gray2rgb

        bmode = resize(bmode, (500, 500))
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = gray2rgb(bmode)
        bmode = bmode * 255
        bmode = bmode.astype("uint8")
        path = join(self.workdir, "bmodes", f"{core_specifier}.jpg")
        Image.fromarray(bmode).save(path)

    def get_path_to_image(self, id) -> str:
        return join(self.workdir, "bmodes", f"{id}.jpg")

    def get_info_for_image(self, id) -> dict:
        if f"{id}.png" in os.listdir(join(self.workdir, "rois")):
            return {"saved": True}
        else:
            return {}

    def submit_roi(self, id, roi: np.ndarray) -> bool:

        fpath = self.get_path_to_roi(id)
        from PIL import Image

        im = Image.fromarray(roi.astype("bool"))
        im.save(fpath)

        return True

    def delete_roi(self, id) -> None:
        if os.path.exists(p := self.get_path_to_roi(id)):
            os.remove(p)

    def list_all_ids(self) -> list[str]:
        return self.core_specifiers

    def upload_prostate_mask(self, specifier):
        path = self.get_path_to_roi(specifier)
        # print(f"uploading mask at path {path}")
        upload_prostate_mask(specifier, fpath=path, overwrite=self.overwrite)

    def finish(self):
        rois_to_upload = [id for id in self.core_specifiers if self.roi_is_saved(id)]
        for specifier in tqdm(rois_to_upload, desc="Uploading ROIs"):
            self.upload_prostate_mask(specifier)

        # if not self.upload_immediately:


#
#    uploaded_segs = []
#    error = None
#
#    for specifier in tqdm(self.core_specifiers, "uploading rois to server"):
#        if self.cache_hits.get(specifier):
#            try:
#                self.upload_prostate_mask(
#                    specifier, self._load_cached_roi(specifier)
#                )
#                uploaded_segs.append(specifier)
#            except ValueError:
#                error = ValueError.args
#
#    print("The following segmentations were uploaded: ")
#    [print(f"\t{seg}") for seg in uploaded_segs]
#    if error is not None:
#        print(f"Caught error :\n{error}\n during upload")
#
#
