from copyreg import pickle
import numpy as np
import pkg_resources
import pandas as pd
import pickle
import os
from ..image_server import get_sftp
from pathlib import Path


_RESOURCES = {}

# go to data folder in repository top level
RESOURCES_DIR = str(Path(__file__).parents[3].joinpath("data"))
REMOTE_RESOURCES_DIR = "/med-i_data/exact_prostate_segemnts/"


global _metadata
_metadata = None


def _get_resource(name, reader):
    if name in _RESOURCES:
        return _RESOURCES[name]

    if not name in os.listdir(RESOURCES_DIR):
        get_sftp().get(
            f"{REMOTE_RESOURCES_DIR}/{name}", os.path.join(RESOURCES_DIR, name)
        )

    _RESOURCES[name] = reader(os.path.join(RESOURCES_DIR, name))

    return _RESOURCES[name]


def _read_csv(fname):
    return pd.read_csv(fname, index_col=[0])


def _read_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def metadata():
    return _get_resource("metadata.csv", _read_csv)


def miccai_splits():
    return _get_resource("miccai_2022_patient_groups.csv", _read_csv)


def patient_test_sets():
    return _get_resource("patients_test_set_by_center.pkl", _read_pkl)


def crceo_428_splits():
    return _get_resource("crceo_428.csv", _read_csv)


def needle_mask():
    return _get_resource("needle_mask.npy", np.load)
