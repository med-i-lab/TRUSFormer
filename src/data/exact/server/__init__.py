import pickle
import errno
import functools
import logging
import dotenv
import paramiko
from typing import Literal, Optional
import mat73
from PIL import Image
from functools import lru_cache, wraps
from ..resources import metadata
from scipy import io
import numpy as np
import os
from ....utils import load_dotenv
from ....data.exact.resources import metadata

load_dotenv()

logged_in = False
ssh: paramiko.SSHClient
sftp: paramiko.SFTPClient


def login():

    if (username := os.getenv("SERVER_USERNAME")) is None:
        username = input("Enter username for image.cs.queensu.ca: ")

    if (password := os.getenv("SERVER_PASSWORD")) is None:
        password = input(f"Enter password for {username}@image.cs.queensu.ca: ")

    global ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("image.cs.queensu.ca", username=username, password=password)

    global sftp
    sftp = ssh.open_sftp()
    sftp.chdir("/med-i_data/Data/Exact_Ultrasound/data/full_data")

    global logged_in
    logged_in = True


def prompt_login(func):
    @functools.wraps(func)
    def logged_in_func(*args, **kwargs):

        logging.debug(f"login prompted by {func}")
        login()

        return func(*args, **kwargs)

    return logged_in_func


@prompt_login
def get_ssh():
    return ssh


@prompt_login
def get_sftp():
    return sftp


def __load(fname):
    assert sftp is not None
    sftp.get(fname, "tmp.mat")
    try:
        out = mat73.loadmat("tmp.mat")
    except OSError:
        out = io.loadmat("tmp.mat")
    os.remove("tmp.mat")
    return out


@prompt_login
def load_by_path(fname: str):
    """Loads the matlab file at the specified server location"""
    return __load(fname)


@prompt_login
def load_by_idx(idx: int):
    """Loads the core specified by its index in the exactvu.resources.metadata table"""
    path = metadata().loc[idx, "path_on_server"]  # type: ignore
    return __load(path)


@prompt_login
def load_by_core_specifier(core_specifier: str):
    """Loads the core specified by its core specifier"""
    match = metadata().query("core_specifier == @core_specifier")
    if len(match) == 0:
        raise KeyError(f"Core specifier {core_specifier} not found")

    return __load(match["path_on_server"].iloc[0])


@prompt_login
def load_by_info(center: str, patient_id: int, loc: str):
    specifier = f"{center}-{str(patient_id).zfill(4)}_{loc}"
    return load_by_core_specifier(specifier)


@lru_cache(maxsize=None)
def paths_to_prostate_masks(refresh=False):

    from src.data import data_dir
    import pickle

    fpath = os.path.join(data_dir(), "paths_to_prostate_masks.pkl")

    @prompt_login
    def _load_and_save_masks():

        masks_root = "/med-i_data/exact_prostate_segemnts/segmentations/segments"

        # load all the paths

        def to_core_specifier(s):
            p, loc, grade = s.split("_")
            return p + "_" + loc

        mask_paths = {}
        for dir in sftp.listdir(masks_root):

            path = f"{masks_root}/{dir}"

            try:
                if "label.png" in sftp.listdir(path):
                    mask_paths[to_core_specifier(dir)] = f"{path}/label.png"
            except IOError as e:
                if e.errno == errno.EACCES:
                    # if an IO error due to access restrictions happens just ignore it
                    pass
                else:
                    raise e

        # dump to file
        with open(fpath, "wb") as f:
            pickle.dump(mask_paths, f)

        return mask_paths

    if refresh or not os.path.isfile(fpath):
        return _load_and_save_masks()
    else:
        # load the cached file from disk
        with open(fpath, "rb") as f:
            return pickle.load(f)


def _load_prostate_mask_manual(core_specifier):
    """Loads one of our manually drawn prostate masks"""

    if not core_specifier in paths_to_prostate_masks().keys():
        raise KeyError(f"Mask not available for core {core_specifier}")

    path = paths_to_prostate_masks()[core_specifier]
    sftp.get(path, "label.png")

    mask = np.array(Image.open("label.png"))
    os.remove("label.png")

    return mask


# ======================
# new prostate masks (due to automatic segmentation module)

_NEW_PROSTATE_MASKS_ROOT = "/med-i_data/exact_prostate_segemnts/new_segmentations_paul"


@lru_cache()
def _prostate_masks_listdir():
    return sftp.listdir(_NEW_PROSTATE_MASKS_ROOT)


@prompt_login
def check_prostate_mask_exists(core_specifier, reload=False):
    if reload:
        _prostate_masks_listdir.cache_clear()

    return f"{core_specifier}__mask.png" in _prostate_masks_listdir()


@prompt_login
def load_prostate_mask(
    core_specifier, source: Literal["old_masks", "new_masks"] = "old_masks"
):

    if source == "old_masks":
        return _load_prostate_mask_manual(core_specifier)

    else:

        if not check_prostate_mask_exists(core_specifier):
            return None

        path = f"{_NEW_PROSTATE_MASKS_ROOT}/{core_specifier}__mask.png"
        sftp.get(path, "label.png")

        mask = np.array(Image.open("label.png"))

        return mask


@prompt_login
def add_prostate_mask(core_specifier, mask, overwrite=False):

    remote_path = f"{_NEW_PROSTATE_MASKS_ROOT}/{core_specifier}__mask.png"

    if check_prostate_mask_exists(core_specifier) and not overwrite:
        raise ValueError(
            f"Refusing to add prostate mask at {remote_path}. Mask already exists."
        )

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    mask.save("label.png")
    remote_path = f"{_NEW_PROSTATE_MASKS_ROOT}/{core_specifier}__mask.png"

    sftp.put("label.png", remote_path)


@prompt_login
def check_prostate_mask_probs_exists(core_specifier):
    return f"{core_specifier}__probs.npy" in sftp.listdir(_NEW_PROSTATE_MASKS_ROOT)


@prompt_login
def load_prostate_mask_probs(
    core_specifier,
):

    remote_path = f"{_NEW_PROSTATE_MASKS_ROOT}/{core_specifier}__probs.npy"

    if not check_prostate_mask_probs_exists(core_specifier):
        return None

    sftp.get(remote_path, "probs.npy")
    mask = np.load("probs.npy")

    return mask


@prompt_login
def add_prostate_mask_probs(core_specifier, probs, overwrite=True):

    remote_path = f"{_NEW_PROSTATE_MASKS_ROOT}/{core_specifier}__probs.npy"

    if check_prostate_mask_probs_exists(core_specifier) and not overwrite:
        raise ValueError(
            f"Refusing to add prostate mask at {remote_path}. Mask already exists."
        )

    probs.save("probs.npy")
    sftp.put("probs.npy", remote_path)


# =========================================


def add_label_tag(core_specifier):
    try:
        grade = metadata().query("core_specifier == @core_specifier")["grade"].iloc[0]
        return core_specifier + "_" + grade
    except IndexError:
        raise ValueError(f"Core specifier {core_specifier} not found in metadata table")
