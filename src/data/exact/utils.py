import logging
from turtle import dot
from typing import Callable, List, Sequence
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from .preprocessing import DEFAULT_PREPROCESS_TRANSFORM
import os
from ...utils import load_dotenv
import torch


info = logging.getLogger(__name__).info


load_dotenv()


def download_and_preprocess_cores(
    root,
    core_specifiers: List[str],
    preprocess_iq_fn=DEFAULT_PREPROCESS_TRANSFORM,
    force_redownload=True,
    download_prostate_masks=False,
):
    from .core import Core

    if not os.path.isdir(root):
        os.mkdir(root)

    cores = []
    with tqdm(core_specifiers, desc="Downloading and preprocessing cores") as pbar:
        for specifier in pbar:

            try:
                core = Core(specifier, root)
                if core.rf is None or force_redownload:
                    core.download_and_preprocess_iq(preprocess_iq_fn)
                if core.prostate_mask is None and download_prostate_masks:
                    if not core.download_prostate_mask():
                        raise Warning(
                            f"download prostate mask was requested but unavailable for core {specifier}"
                        )

            except Exception as e:
                raise RuntimeError(
                    f"Exception occurred while processing core {specifier}"
                ) from e

            cores.append(core)

    return cores


import numpy as np
from itertools import product


def slice_intersection(x1, x2, x1_, x2_):
    """determines the length of intersection between the slice [x1:x2], [x1_:x2_]"""
    assert x2 > x1 and x2_ > x1_
    return max(min(x2, x2_) - (max(x1, x1_)), 0)


def sliding_window_grid(img_size, window_size, step_size):
    """
    Computes the correct indices for a sliding window of size window_size over
    and img of size img_size with stride lengths step_size.

    returns an array whose i, j'th entry contains the array
    [x1, x2, y1, y2]
    such that
    img[x1: x2, y1: y2]
    accesses the patch of the original image corresponding to the correct grid position.
    """

    axial_startpos = [
        i
        for i in range(0, img_size[0], step_size[0])
        if i + window_size[0] <= img_size[0]
    ]
    lateral_startpos = [
        i
        for i in range(0, img_size[1], step_size[1])
        if i + window_size[1] <= img_size[1]
    ]

    grid_h = len(axial_startpos)
    grid_w = len(lateral_startpos)
    out = np.zeros((grid_h, grid_w, 4))

    for i, j in product(range(grid_h), range(grid_w)):

        x1 = axial_startpos[i]
        x2 = x1 + window_size[0]
        y1 = lateral_startpos[j]
        y2 = y1 + window_size[1]

        out[i, j, :] = np.array([x1, x2, y1, y2])

    return out.astype(np.int32)


def collate_variable_length_tensors(batch: List[torch.Tensor]):
    """
    Stacks the variable length tensors along their first dimension and returns
    an "batch" and "ptr" tensor. The batch tensor indicates the batch origin for
    each input position and the ptr tensor indicates the start position in the stacked
    tensor for each batch.
    """

    lengths = [len(t) for t in batch]

    ptr = torch.cumsum(torch.tensor([0] + lengths), dim=0)[:-1]

    ind = []
    for i, length in enumerate(lengths):
        [ind.append(i) for _ in range(length)]
    ind = torch.tensor(ind)

    return torch.concat(batch, dim=0), ind, ptr


def unpack_stacked_batch(stacked_batch, batch_ind):

    indices = torch.sort(torch.unique(batch_ind))
    out = []
    for index in indices:
        out.append(stacked_batch[batch_ind == index])

    return out
