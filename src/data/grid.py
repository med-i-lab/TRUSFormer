from abc import abstractmethod, ABC
from typing import Tuple, Callable, Optional
import numpy as np
import shutil
import pickle

from torch import ShortTensor

from ..utils.preprocessing import split_into_patches_pixels
import os
from itertools import product
import numpy as np


class SubPatchAccessorMixin(ABC):
    """
    Class for data structured as a "grid" of subpatches. For when individual subpatches
    can be easily accessed (eg. because they are stored on the disk). Abstracts the
    logic of accessing and stitching together multiple subpatches into a larger patch into
    intuitive slice notation.

    To use this wrapper, subclass it and implement the "shape" property, representing the
    grid shape, the "subpatch_shape" property, representing the shape of a subpatch,
    and implement the method to access a single subpatch specified by its grid location i, j
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        ...

    @property
    @abstractmethod
    def subpatch_shape(self) -> Tuple[int]:
        ...

    @property
    def dtype(self):
        if getattr(self, "_dtype", None) is None:
            self._dtype = self.get_subpatch(0, 0).dtype
        return self._dtype

    @abstractmethod
    def get_subpatch(self, i, j) -> np.ndarray:
        ...

    def __getitem__(self, val):

        val = list(map(lambda i: slice(i, i + 1) if type(i) == int else i, val))
        i, j = val

        patch_indices = np.indices(self.shape)[:, i, j]
        grid_shape = patch_indices[0].shape
        grid_indices = np.indices(grid_shape)

        out = np.zeros((*grid_shape, *self.subpatch_shape)).astype(self.dtype)

        for i, j in zip(grid_indices[0].flatten(), grid_indices[1].flatten()):
            out[i, j] = self.get_subpatch(*patch_indices[:, i, j])

        import einops

        out = einops.rearrange(
            out,
            "grid_i grid_j patch_h patch_w ... -> (grid_i patch_h) (grid_j patch_w) ...",
        )

        return out

    def view(self, positions: np.ndarray):
        """
        Returns a view of this grid whose positions are specified by an array.
        The last axis of the array should be a 4 dimensional vector specifying
        x1, x2, y1, y2 that will be used to access slice  x1:x2, y1:y2 of the grid.

        For example, if passed the rank 3 array `positions` of shape [2, 2, 4],
        Then accessing the view at position [0, 1] will pull the slice
        x1:x2, y1:y2 from the grid , where x1, x2, y1, y2 = positions[0, 1].
        """

        class View:
            def __init__(self, grid, positions):
                self._grid = grid
                self._positions = positions

            def __getitem__(self, val):
                x1, x2, y1, y2 = self._positions[val]
                return self._grid[x1:x2, y1:y2]

        return View(self, positions)


class SavedSubPatchGrid(SubPatchAccessorMixin):
    def __init__(self, path_to_image, patch_height_pixels, patch_width_pixels):

        self.directory = os.path.join(os.path.dirname(path_to_image), "subpatches")
        self.path_to_image = path_to_image
        self.patch_height_pixels = patch_height_pixels
        self.patch_width_pixels = patch_width_pixels

        try:
            grid_metadata = self.read(os.path.join(self.directory, "grid_metadata.pkl"))
        except FileNotFoundError:
            grid_metadata = None
        if grid_metadata is None or grid_metadata["subpatch_shape"] != (
            self.patch_height_pixels,
            self.patch_width_pixels,
        ):
            self.create_grid()
        else:
            self._shape = grid_metadata["shape"]
            self._subpatch_shape = grid_metadata["subpatch_shape"]

    def create_grid(self):

        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        os.mkdir(self.directory)

        img = np.load(self.path_to_image)

        grid = split_into_patches_pixels(
            img, self.patch_height_pixels, self.patch_width_pixels
        )

        self._shape = grid.shape[:2]
        self._subpatch_shape = grid.shape[2:]

        for i, j in product(range(grid.shape[0]), range(grid.shape[1])):
            patch = grid[i, j]
            fpath = self.get_filename_for_patch(i, j)
            self.save(fpath, patch)

        self.save(
            os.path.join(self.directory, "grid_metadata.pkl"),
            {
                "shape": self._shape,
                "subpatch_shape": self._subpatch_shape,
            },
        )

    def save(self, filename, data):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def read(self, filename):
        with open(filename, "rb") as f:
            out = pickle.load(f)
        return out

    def get_filename_for_patch(self, i, j):
        return os.path.join(self.directory, f"{i}_{j}.pkl")

    def get_subpatch(self, i, j) -> np.ndarray:
        return self.read(self.get_filename_for_patch(i, j))

    @property
    def shape(self):
        return self._shape

    @property
    def subpatch_shape(self) -> Tuple[int]:
        return self._subpatch_shape


class PatchCoordinatesCalculator:
    def __init__(
        self,
        image_shape,
        patch_height_pixels,
        patch_width_pixels,
        discard_from="bottomleft",
    ):

        self.img_shape = image_shape
        self.patch_height_pixels = patch_height_pixels
        self.patch_width_pixels = patch_width_pixels
        self.discard_from = discard_from

    from functools import lru_cache

    @classmethod
    @lru_cache
    def get_calculator(
        cls, image_shape, patch_height_pixels, patch_width_pixels, discard_from
    ):
        return cls(image_shape, patch_height_pixels, patch_width_pixels, discard_from)

    @lru_cache
    def get_slice(self, i1, i2, j1, j2):

        dw = self.patch_width_pixels
        dh = self.patch_height_pixels

        w = int(self.img_shape[1] / dw)
        h = int(self.img_shape[0] / dh)

        discard_from = self.discard_from

        origin = [0, 0]
        if "bottom" in discard_from:
            origin[0] = 0
        elif "top" in discard_from:
            origin[0] = self.img_shape[0] - (h * dh)
        if "right" in discard_from:
            origin[1] = 0
        elif "left" in discard_from:
            origin[1] = self.img_shape[1] - (w * dw)

        slice = (
            dh * i1 + origin[0],
            dh * i2 + origin[0],
            dw * j1 + origin[1],
            dw * j2 + origin[1],
        )
        return slice

    def get_shape(self):

        dw = self.patch_width_pixels
        dh = self.patch_height_pixels

        w = int(self.img_shape[1] / dw)
        h = int(self.img_shape[0] / dh)

        return h, w


class InMemoryImagePatchGrid:
    def __init__(
        self, image, patch_height_pixels, patch_width_pixels, discard_from="bottomleft"
    ):
        self.image = image
        self.calculator = PatchCoordinatesCalculator.get_calculator(
            image.shape, patch_height_pixels, patch_width_pixels, discard_from
        )

    def get_slice(self, i1, i2, j1, j2):
        return self.calculator.get_slice(i1, i2, j1, j2)

    def __getitem__(self, val):
        i, j = val
        if type(i) == int:
            i = slice(i, i + 1)
        if type(j) == int:
            j = slice(j, j + 1)

        i1 = i.start if i.start is not None else 0
        i2 = i.stop if i.stop is not None else self.shape[0]
        j1 = j.start if j.start is not None else 0
        j2 = j.stop if j.stop is not None else self.shape[1]

        x1, x2, y1, y2 = self.get_slice(i1, i2, j1, j2)
        return self.image[x1:x2, y1:y2]

    @property
    def shape(self):
        return self.calculator.get_shape()

    @property
    def subpatch_shape(self) -> Tuple[int]:
        return self[0, 0].shape
