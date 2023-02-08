from albumentations import (
    Equalize,
    GaussianBlur,
    GridDistortion,
    HorizontalFlip,
    Rotate,
    RandomBrightnessContrast,
    Compose,
)
import torch
import skimage
from typing import Tuple
import numpy as np
from torchvision.transforms import Normalize
from skimage.transform import resize
from dataclasses import dataclass


@dataclass
class SegmentationTransformConfig:
    _target_: str = __name__ + ".SegmentationTransform"
    use_augmentations: bool = True
    equalize_hist: bool = False
    random_rotate: bool = True
    grid_distortion: bool = True
    horizontal_flip: bool = True
    gaussian_blur: bool = True
    random_brightness: bool = True
    to_tensor: bool = True
    out_size: Tuple[int, int] = (512, 512)


class SegmentationTransform:
    """
    A preproccessor for segmentation data which transforms the raw numpy arrays
    to torch tensors, optionally including various data augmentations.
    """

    def __init__(
        self,
        use_augmentations=True,
        equalize_hist=False,
        random_rotate=True,
        grid_distortion=True,
        horizontal_flip=True,
        gaussian_blur=True,
        random_brightness=True,
        to_tensor=True,
        out_size=(512, 512),
    ):

        self.eq = Equalize(always_apply=True) if equalize_hist else None
        self.out_size = out_size

        self.normalize = Normalize((0.5,), (0.25,))

        self.use_augmentations = use_augmentations
        self.to_tensor = to_tensor

        augmentations = []
        if random_rotate:
            import cv2

            augmentations.append(Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT))
        if grid_distortion:
            augmentations.append(GridDistortion())
        if horizontal_flip:
            augmentations.append(HorizontalFlip())
        if gaussian_blur:
            augmentations.append(GaussianBlur())
        if random_brightness:
            augmentations.append(RandomBrightnessContrast())
        self.augmentations = Compose(augmentations)

    def __call__(self, bmode, seg):

        bmode = resize(bmode, self.out_size)
        seg = resize(seg, self.out_size)
        seg = seg.astype("uint8")

        bmode = skimage.exposure.rescale_intensity(bmode, out_range="uint8")

        if self.eq:
            bmode, seg = self.__apply_transform(self.eq, bmode, seg)

        if self.use_augmentations:
            bmode, seg = self.__apply_transform(self.augmentations, bmode, seg)

        if self.to_tensor:
            bmode = skimage.exposure.rescale_intensity(bmode, out_range="float32")
            bmode = np.expand_dims(bmode, axis=0)
            bmode = torch.tensor(bmode)
            bmode = self.normalize(bmode)
            seg = torch.tensor(seg).long()

        return bmode, seg

    def __apply_transform(self, T, bmode, seg):
        out = T(image=bmode, mask=seg)
        return out["image"], out["mask"]
