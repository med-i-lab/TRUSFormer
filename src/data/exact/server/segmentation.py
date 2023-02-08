from warnings import warn
from ...image_server import get_ssh, get_sftp

import numpy as np
from PIL import Image
import os
import stat
from . import add_label_tag
from functools import cache


SEGMENTATIONS_WORK_DIR = "/med-i_data/exact_prostate_segemnts/segmentations"
SEGMENTATIONS_DATA_DIR = "/med-i_data/exact_prostate_segemnts/segmentations/segments"


def _get_path_to_prostate_segmentation_workdir(core_specifier):
    """
    Returns the correct path to the prostate segmentation
    working directory on the image server
    """
    name = add_label_tag(core_specifier)
    return SEGMENTATIONS_DATA_DIR + "/" + name


def _make_prostate_segmentation_workdir(core_specifier):
    dirpath = _get_path_to_prostate_segmentation_workdir(core_specifier)
    get_sftp().mkdir(dirpath, mode=stat.S_IROTH | stat.S_IRWXU | stat.S_IRWXG)


def _get_path_to_prostate_segmentation(core_specifier):
    """
    Returns the correct path to the prostate
    segmentation on the image server for the given core specifier
    """
    return (
        _get_path_to_prostate_segmentation_workdir(core_specifier) + "/" + "label.png"
    )


@cache
def list_available_prostate_segmentations():
    """
    Returns a list of core specifiers for prostates which have segmentations available
    """

    in_, out, err = get_ssh().exec_command(
        "source /med-i_data/exact_prostate_segemnts/segmentations/list_all_segs.sh"
    )
    list_specifiers = out.read().decode().split()
    return [
        specifier.split("_")[0] + "_" + specifier.split("_")[1]
        for specifier in list_specifiers
    ]


def check_prostate_segmentation_is_available(core_specifier):
    """
    Checks whether a prostate segmentation mask is available for the
    specified core.
    """
    return core_specifier in list_available_prostate_segmentations()


def get_prostate_segmentation(core_specifier):
    """
    Returns the prostate segmentation for the given core
    specifier by downloading it from the server
    """
    if not check_prostate_segmentation_is_available(core_specifier):
        raise ValueError(
            f"Core {core_specifier} does not have a prostate segmentation."
        )

    path = _get_path_to_prostate_segmentation(core_specifier)
    get_sftp().get(path, "label.png")
    mask = np.array(Image.open("label.png"))
    os.remove("label.png")
    return mask


#
# def manually_inspect_available_prostate_segmentations() -> list:
#    """Manually looks through the file structure on the
#    server to see which prostate_segmentations are available"""
#
#    core_specifiers_with_segs = []
#    for path in tqdm(get_sftp_client().listdir(SEGMENTATIONS_DATA_DIR)):
#
#        core_specifier = get_core_specifier(path)
#
#        if "label.png" in get_sftp_client().listdir(f"{SEGMENTATIONS_DATA_DIR}/{path}"):
#            core_specifiers_with_segs.append(core_specifier)
#
#    return core_specifiers_with_segs
#
#
# def push_segmentations_lookup_table(table: list):
#    """Push the new table to the segmentations lookup."""
#
#    with LockingSFTPFile(f"{SEGMENTATIONS_LOOKUP_FPATH}", get_ssh_client()).open(
#        "wb"
#    ) as f:
#        f.write(json.dumps(table).encode())
#
#    # get_ssh_client().exec_command(
#    #    f"mv {SEGMENTATIONS_LOOKUP_FPATH}.tmp {SEGMENTATIONS_LOOKUP_FPATH}"
#    # )
#
#


def upload_prostate_mask(core_specifier, mask=None, fpath=None, overwrite=False):

    try:
        get_sftp().stat(_get_path_to_prostate_segmentation_workdir(core_specifier))
    except IOError:
        _make_prostate_segmentation_workdir(core_specifier)

    remote_path = _get_path_to_prostate_segmentation(core_specifier)

    if check_prostate_segmentation_is_available(core_specifier) and not overwrite:
        raise ValueError(
            f"Refusing to add prostate mask at {remote_path}. Mask already exists."
        )

    if fpath is not None:
        get_sftp().put(fpath, remote_path)

    else:
        assert mask is not None, "if not passing path, must pass ROI mask as argument"

        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype("bool"))

        mask.save("label.png")

        dir = _get_path_to_prostate_segmentation_workdir(core_specifier)
        if not os.path.basename(dir) in get_sftp().listdir(SEGMENTATIONS_DATA_DIR):
            get_sftp().mkdir(dir)

        get_sftp().put("label.png", remote_path)
