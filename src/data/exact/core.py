from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Tuple, overload, Optional
from skimage.transform import resize
from warnings import warn
from functools import cache

from .preprocessing import (
    patch_size_mm_to_pixels,
    split_into_patches,
    DEFAULT_PREPROCESS_TRANSFORM,
    split_into_patches_pixels,
)
from ...data.grid import (
    InMemoryImagePatchGrid,
    SavedSubPatchGrid,
    SubPatchAccessorMixin,
)
from itertools import product
import numpy as np
import os
from .server import load_by_core_specifier, load_prostate_mask
from .server.segmentation import get_prostate_segmentation
from .preprocessing import to_bmode
from enum import Enum, auto

import matplotlib.pyplot as plt


AXIAL_IMAGE_DEPTH = 28
LATERAL_IMAGE_WIDTH = 46


def mask_intersections_for_grid(img_shape, mask, subpatch_size_mm=(1, 1)):

    mask = resize(mask.astype("bool"), (img_shape[0], img_shape[1]))
    intersections = split_into_patches(mask, *subpatch_size_mm)
    patch_size = intersections[0, 0].size
    return intersections.sum(axis=(-1, -2)) / patch_size


def mask_intersections_for_rf_slices(mask, rf_slices, rf_shape):
    mask = resize(
        mask.astype("bool"),
        rf_shape,
    )
    intersections = np.zeros(rf_slices.shape[:-1])
    indices = product(*map(range, intersections.shape))
    for idx in indices:
        x1, x2, y1, y2 = rf_slices[idx]
        mask_slice = mask[x1:x2, y1:y2]
        intersection_ratio = mask_slice.sum() / mask_slice.size
        intersections[idx] = intersection_ratio

    return intersections


def grid_slices_to_rf_slices(
    grid_slices: np.ndarray, rf_shape, subpatch_size_mm=(1, 1)
):
    """
    Converts the given array of grid positions corresponding to slices of a base patch grid
    of shape specified by `subpatch_shape` to an array of slice positions corresponding to the
    rf image with the given shape
    """
    patch_size = patch_size_mm_to_pixels(rf_shape, *subpatch_size_mm)

    out = np.zeros_like(grid_slices)
    out[..., 0] = grid_slices[..., 0] * patch_size[0]
    out[..., 1] = grid_slices[..., 1] * patch_size[0]
    out[..., 2] = grid_slices[..., 2] * patch_size[1]
    out[..., 3] = grid_slices[..., 3] * patch_size[1]

    return out


def is_inside_mask(
    x1, x2, y1, y2, mask, subpatch_size_pixels, intersection_threshold=0.6
):
    """check if the patch specified by grid coordinates x1, x2, y1, y2 is inside the given mask,
    assuming the mask is divided into a grid of subpatches with size subpatch_size_pixels.

    The criterion for being inside the mask is an overlap of at least intersection_threshold
    with the mask."""

    intersections = split_into_patches_pixels(mask, *subpatch_size_pixels)
    patch_size = intersections[0, 0].size
    intersections = intersections.sum(axis=(1, -2)) / patch_size

    intersection = needle_intersections_grid[  # type:ignore
        x1:x2, y1:y2
    ]
    intersection = intersection.mean()

    return intersection >= intersection_threshold


class PatchView:
    """A view of a base grid that allows accessing the patches at the given positions"""

    def __init__(
        self,
        base_grid,
        patch_positions,
        mask_intersections=None,
        return_extras=["positions"],
    ):
        self.base_grid = base_grid
        self.patch_positions = patch_positions
        self.mask_intersections = mask_intersections
        self.return_extras = return_extras

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.patch_positions[idx]
        patch = self.base_grid[x1:x2, y1:y2]

        out = [patch]
        if "positions" in self.return_extras:
            out.append(self.patch_positions[idx])
        if "mask_intersections" in self.return_extras:
            if self.mask_intersections is None:
                raise ValueError(
                    "mask intersections are not available for this patch view"
                )
            out.append(self.mask_intersections[idx])

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    def plot_over_image(patch_view, image, extent, mask_name):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray", extent=extent)
        centers = []
        intersections = []
        for patch, pos, mask_intersections in patch_view:
            centers.append([(pos[2] + pos[3]) / 2, (pos[0] + pos[1]) / 2])
            intersections.append(mask_intersections[mask_name])
        centers = np.array(centers)
        ax.scatter(centers[:, 0], centers[:, 1], c=intersections, s=1)
        return fig, ax


class PatchBackends(Enum):
    INDIVIDUAL_SAVED_SUBPATCHES = auto()
    IN_MEMORY_IMAGE = auto()


@dataclass
class PatchViewConfig:
    """Configuration for the patch view

    Parameters
    ----------
    patch_size : Tuple[int, int]
        size of the patches in mm
    patch_strides : Tuple[int, int]
        strides of the patches in mm
    subpatch_size : Tuple[int, int]
        size of the subpatches in mm
    needle_region_only : Optional[bool]
        if True, only patches that are inside the needle region are returned.
    prostate_region_only : Optional[bool]
        if True, only patches that are inside the prostate region are returned.
    prostate_intersection_threshold : float
        the minimum overlap of a patch with the prostate region to be considered inside the prostate region
    needle_intersection_threshold : float
        the minimum overlap of a patch with the needle region to be considered inside the needle region
    return_extras : List[str]
        list of extra information to return. Currently supported: 'positions', 'mask_intersections
    """

    patch_size: Tuple[int, int] = (5, 5)
    patch_strides: Tuple[int, int] = (1, 1)
    subpatch_size: Tuple[int, int] = (1, 1)
    needle_region_only: Optional[bool] = None
    prostate_region_only: Optional[bool] = None
    prostate_intersection_threshold: float = 0.9
    needle_intersection_threshold: float = 0.6
    return_extras: Tuple[str] = ("positions",)

    def __post_init__(self):
        if self.prostate_region_only is None:
            self.prostate_region_only = self.prostate_intersection_threshold != 0.0
        if self.needle_region_only is None:
            self.needle_region_only = self.needle_intersection_threshold != 0.0


@dataclass
class CoreRCParams:
    patch_backend: PatchBackends = PatchBackends.IN_MEMORY_IMAGE
    cache_rf: bool = False
    cache_bmode: bool = True
    cache_prostate_mask: bool = False


class Core:

    _rc_params: CoreRCParams = CoreRCParams()
    _built_cores_lookup = {}
    _calling_constructor = False
    _has_been_warned = False

    @classmethod
    def create_core(cls, specifier, directory=None):
        if specifier in cls._built_cores_lookup:
            return cls._built_cores_lookup[specifier]
        else:
            cls._calling_constructor = True
            core = cls(specifier, root=directory)
            cls._calling_constructor = False
            cls._built_cores_lookup[specifier] = core
            return core

    @classmethod
    def set_rc_params(cls, rc_params: CoreRCParams):
        cls._rc_params = rc_params

    def __init__(
        self,
        specifier,
        root=None,
    ):
        if not self._calling_constructor and not self._has_been_warned:
            warn(
                """You appear to be calling Core constructor manually. It is preferable to call `Core.create_core(specifier)`
                    which will cache core instances and not double memory requirements for core objects!
                """
            )
            self._has_been_warned = True

        if root is None:

            root = self.default_data_dir()
            if not os.path.isdir(root):
                os.mkdir(root)

        self.directory = os.path.join(root, specifier)
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.specifier = specifier
        self._metadata = None
        self._rf = None
        self._bmode = None
        self._prostate_mask = None

    @property
    def pixel_spacing(self):
        axial_num_pixels, lateral_num_pixels = self.rf.shape
        return (
            AXIAL_IMAGE_DEPTH / axial_num_pixels,
            LATERAL_IMAGE_WIDTH / lateral_num_pixels,
        )

    @property
    def rf(self):
        if self._rf is not None:
            return self._rf
        if not self.rf_is_downloaded:
            self.download_and_preprocess_iq()
        rf = np.load(os.path.join(self.directory, "image.npy"), mmap_mode="r")
        if self._rc_params.cache_rf:
            self._rf = rf
        return rf

    @property
    def rf_is_downloaded(self):
        return os.path.isfile(os.path.join(self.directory, "image.npy"))

    @property
    def prostate_mask(self):
        if self._prostate_mask is not None:
            return self._prostate_mask
        if not self.prostate_mask_is_downloaded:
            self.download_prostate_mask()
        mask = np.load(os.path.join(self.directory, "prostate_mask.npy"))
        if self._rc_params.cache_prostate_mask:
            self._prostate_mask = mask
        return mask

    @property
    def prostate_mask_is_downloaded(self):
        path = os.path.join(self.directory, "prostate_mask.npy")
        return os.path.isfile(path)

    def download_prostate_mask(self):
        path = os.path.join(self.directory, "prostate_mask.npy")
        mask = get_prostate_segmentation(self.specifier)
        np.save(path, mask)  # type:ignore

        # if we are manually downloading a new prostate mask,
        # we should delete the stored intersection grids
        for fname in os.listdir(self.directory):
            if "prostate_mask_grid_subpatch_size_" in fname:
                os.remove(os.path.join(self.directory, fname))
        return True

    @property
    def bmode(self):
        if self._bmode is not None:
            return self._bmode
        if not self.bmode_is_downloaded:
            self.download_bmode()
        bmode = np.load(os.path.join(self.directory, "bmode.npy"))
        if self._rc_params.cache_bmode:
            self._bmode = bmode
        return bmode

    @property
    def bmode_is_downloaded(self):
        return os.path.isfile(os.path.join(self.directory, "bmode.npy"))

    def download_bmode(self):
        if not self.rf_is_downloaded:
            iq = load_by_core_specifier(self.specifier)
            rf = DEFAULT_PREPROCESS_TRANSFORM(iq)
        else:
            rf = self.rf

        bmode = to_bmode(rf)
        np.save(os.path.join(self.directory, "bmode.npy"), bmode)

    @property
    def needle_mask(self):
        from .resources import needle_mask

        return needle_mask()

    @property
    def metadata(self):
        if self._metadata is None:
            from .resources import metadata

            _metadata = metadata()
            _metadata = _metadata.query(f"core_specifier == @self.specifier")
            self._metadata = dict(_metadata.iloc[0])

        return self._metadata

    @property
    def has_prostate_mask(self):
        from src.data.exact.server.segmentation import (
            list_available_prostate_segmentations,
        )

        return self.specifier in list_available_prostate_segmentations()

    def get_patient_object(self):
        from .patient import ExactPatient

        return ExactPatient(self.metadata["patient_specifier"])

    def get_needle_mask_intersections(self, subpatch_size):
        fpath = os.path.join(
            self.directory,
            f"needle_mask_grid_subpatch_size_{subpatch_size[0]}-{subpatch_size[1]}.npy",
        )
        try:
            return np.load(fpath)
        except FileNotFoundError:
            intersections = self.compute_patch_intersections(
                self.needle_mask, subpatch_size
            )
            np.save(fpath, intersections)
            return intersections

    def get_prostate_mask_intersections(self, subpatch_size):

        fpath = os.path.join(
            self.directory,
            f"prostate_mask_grid_subpatch_size_{subpatch_size[0]}-{subpatch_size[1]}.npy",
        )

        try:
            return np.load(fpath)
        except FileNotFoundError:
            intersections = self.compute_patch_intersections(
                self.prostate_mask, subpatch_size
            )
            np.save(fpath, intersections)
            return intersections

        else:
            intersections = self.compute_patch_intersections(
                self.prostate_mask, subpatch_size
            )
            np.save(fpath, intersections)
            return intersections

    def download_and_preprocess_iq(
        self, iq_preprocessor_fn=DEFAULT_PREPROCESS_TRANSFORM
    ):
        iq = load_by_core_specifier(self.specifier)
        image = iq_preprocessor_fn(iq)
        self.rf = image

        image_fpath = os.path.join(self.directory, "image.npy")
        np.save(image_fpath, image)

    def get_grid_view(self, subpatch_size_mm=(1, 1)):

        assert self.rf is not None, "Image not downloaded. "
        subpatch_size_pixels = patch_size_mm_to_pixels(self.rf.shape, *subpatch_size_mm)

        if self._rc_params.patch_backend == PatchBackends.IN_MEMORY_IMAGE:

            grid = InMemoryImagePatchGrid(
                self.rf,
                subpatch_size_pixels[0],
                subpatch_size_pixels[1],
            )

            return grid

        elif self._rc_params.patch_backend == PatchBackends.INDIVIDUAL_SAVED_SUBPATCHES:

            grid = SavedSubPatchGrid(
                os.path.join(self.directory, "image.npy"), *subpatch_size_pixels
            )
            return grid

        else:
            raise ValueError(f"backend {self._rc_params.patch_backend} not supported.")

    def compute_patch_intersections(self, mask, subpatch_size_mm):
        assert (img := self.rf) is not None, "Image not loaded for core."
        return mask_intersections_for_grid(img.shape, mask, subpatch_size_mm)

    def get_patch_view_from_config(self, config: PatchViewConfig):
        if isinstance(config, PatchViewConfig):
            config = asdict(config)
        return self.get_patch_view(**config)

    @cache
    def get_patch_view(
        self,
        patch_size=(5, 5),
        patch_strides=(1, 1),
        subpatch_size=(1, 1),
        needle_region_only=False,
        prostate_region_only=False,
        prostate_intersection_threshold=0.6,
        needle_intersection_threshold=0.6,
        return_extras=["positions"],
    ) -> PatchView:

        should_compute_needle_mask_intersections = (
            needle_region_only or "mask_intersections" in return_extras
        )
        should_compute_prostate_mask_intersections = (
            prostate_region_only or "mask_intersections" in return_extras
        )
        if not self.has_prostate_mask:
            if prostate_region_only:
                raise ValueError(
                    f"""Requested prostate region only, but no prostate mask available for core {self.specifier}"""
                )
            else:
                should_compute_prostate_mask_intersections = False

        base_grid = self.get_grid_view(subpatch_size)

        h, w = base_grid.shape

        axial_startpos = [
            i for i in range(0, h, patch_strides[0]) if i + patch_size[0] <= h
        ]
        lateral_startpos = [
            i for i in range(0, w, patch_strides[1]) if i + patch_size[1] <= w
        ]

        patch_positions = []
        mask_intersections = []

        if should_compute_needle_mask_intersections:
            needle_intersections_grid = self.get_needle_mask_intersections(
                subpatch_size
            )
        else:
            needle_intersections_grid = None

        if should_compute_prostate_mask_intersections:
            prostate_intersections_grid = self.get_prostate_mask_intersections(
                subpatch_size
            )
        else:
            prostate_intersections_grid = None

        for i, j in product(axial_startpos, lateral_startpos):

            intersections = {}
            if should_compute_needle_mask_intersections:

                intersection = needle_intersections_grid[  # type:ignore
                    i : i + patch_size[0], j : j + patch_size[1]
                ]
                intersection = intersection.mean()
                intersections["needle"] = intersection

                if needle_region_only and intersection < needle_intersection_threshold:
                    continue

            if should_compute_prostate_mask_intersections:

                intersection = prostate_intersections_grid[  # type:ignore
                    i : i + patch_size[0], j : j + patch_size[1]
                ]
                intersection = intersection.mean()
                intersections["prostate"] = intersection

                if (
                    prostate_region_only
                    and intersection < prostate_intersection_threshold
                ):
                    continue

            patch_positions.append((i, i + patch_size[0], j, j + patch_size[0]))
            mask_intersections.append(intersections)

        return PatchView(base_grid, patch_positions, mask_intersections, return_extras)

    def get_sliding_window_view(
        self, window_size=(5, 5), step_size=(1, 1), subpatch_size_mm=(1, 1)
    ):
        assert self.rf is not None, "Download rf first"

        grid = self.get_grid_view(subpatch_size_mm)

        assert isinstance(
            grid, SubPatchAccessorMixin
        ), f"Backend {type(grid)} not supported for this operation."

        from .utils import sliding_window_grid

        out = {}

        grid_slices = sliding_window_grid(grid.shape, window_size, step_size)

        out["view"] = grid.view(grid_slices)
        out["positions"] = grid_slices
        rf_slices = grid_slices_to_rf_slices(
            grid_slices, self.rf.shape, subpatch_size_mm
        )
        if self.needle_mask is not None:
            out["needle_region_intersections"] = mask_intersections_for_rf_slices(
                self.needle_mask, rf_slices, self.rf.shape
            )

        if self.prostate_mask is not None:
            out["prostate_region_intersections"] = mask_intersections_for_rf_slices(
                self.needle_mask, rf_slices, self.rf.shape
            )

        return out

    def filter_inside_needle_region(
        self, positions: Iterable, subpatch_size=(1, 1), threshold=0.65
    ):
        """
        Given the iterable of positions, returns a list of the positions that are inside the
        needle region for this core, according to the threshold specified. The patch will be considered
        inside if the fraction of its pixels lying inside the mask meets or exceeds the threshold.
        """

        filtered = []
        needle_intersections_grid = self.get_needle_mask_intersections(subpatch_size)

        for position in positions:

            x1, x2, y1, y2 = position

            intersection = needle_intersections_grid[  # type:ignore
                x1:x2, y1:y2
            ]
            intersection = intersection.mean()

            if intersection >= threshold:
                filtered.append(position)

        return filtered

    def filter_inside_prostate_region(
        self, positions, subpatch_size=(1, 1), threshold=0.9
    ):
        """
        Given the iterable of positions, returns a list of the positions that are inside the
        needle region for this core, according to the threshold specified. The patch will be considered
        inside if the fraction of its pixels lying inside the mask meets or exceeds the threshold.
        """
        filtered = []
        needle_intersections_grid = self.get_needle_mask_intersections(subpatch_size)

        for position in positions:

            x1, x2, y1, y2 = position

            intersection = needle_intersections_grid[  # type:ignore
                x1:x2, y1:y2
            ]
            intersection = intersection.mean()

            if intersection >= threshold:
                filtered.append(position)

        return filtered

    def clear_data(self):
        import shutil

        shutil.rmtree(self.directory)

    @staticmethod
    def sample_core():
        """Returns a sample core object to speed up development"""
        from .splits import get_splits, filter_splits, HasProstateMaskFilter

        train, val, test = filter_splits(
            get_splits("UVA600", True, 26), HasProstateMaskFilter()
        )
        return Core(train[0])

    @staticmethod
    def default_data_dir():
        """
        default location of a top-level data directory ('~/data'),
        or read from DATA environment variable.
        """
        if not os.environ.get("DATA"):
            print("Environment variable DATA not set")
        else:
            return os.environ["DATA"]

        root = input("Enter data root: ")
        if not os.path.isdir(root):
            raise ValueError(f"root {root} is not a directory")
        os.environ["DATA"] = root

        return root
