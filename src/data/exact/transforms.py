from dataclasses import asdict, dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union
import numpy as np
from torchvision import transforms as T
from .ultrasound_augs import UltrasoundArrayAugmentation
from skimage.transform import resize
import torch


PIXEL_MEAN = 0.24708273
PIXEL_STD = 848.8191


@dataclass
class UltrasoundAugsConfig:
    _target_: str = "src.data.exact.transforms.UltrasoundArrayAugmentation"

    random_phase_shift: bool = False
    random_phase_distort: bool = False
    random_phase_distort_strength: float = 0.1
    random_phase_distort_freq_limit: float = 0.3
    random_envelope_distort: bool = False
    random_envelope_distort_strength: float = 0.2
    random_envelope_distort_freq_limit: float = 0.1
    random_bandstop: bool = False
    random_bandstop_width: float = 0.1
    random_freq_stretch: bool = False
    random_freq_stretch_range: Tuple[float, float] = (0.98, 1.0)


@dataclass
class TensorAugsConfig:
    _target_: str = "src.data.exact.transforms.TensorImageAugmentation"

    prob: float = 0.5
    random_erasing: bool = False
    random_invert: bool = True
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = True
    random_affine_translation: Tuple[float, float] = (0.0, 0.0)
    random_affine_rotation: int = 0
    random_affine_shear: Tuple[int, int, int, int] = (0, 0, 0, 0)
    random_resized_crop: bool = True
    random_resized_crop_scale: Tuple[float, float] = (0.5, 1)


@dataclass
class NormConfig:
    _target_: str = "src.data.exact.transforms.Normalize"

    mode: str = "instance"
    type: str = "min-max"
    truncate: bool = True


@dataclass
class TransformConfig:
    _target_: str = "src.data.exact.transforms.TransformV3"

    norm: NormConfig = NormConfig()
    tensor_transform: Optional[TensorAugsConfig] = TensorAugsConfig()
    us_augmentation: Optional[UltrasoundAugsConfig] = None
    out_size: Tuple[int, int] = (256, 256)


# def register_configs():


class Normalize:
    def __init__(
        self,
        mode: Literal["instance", "global"] = "instance",
        type: Literal["z-score", "min-max"] = "min-max",
        truncate: bool = True,
    ):
        """
        Creates a function which normalizes patches of RF data, to be called on torch.tensor objects
        mode: whether to normalize the image based on global properties
            or properties of the specific patch.
        type: whether to use z-score normalization, or min-max normalization
        truncate: If selected, truncates pixel values that are more than 4 standard
        deviations from the mean. If using min-max normalization, this happens
        BEFORE the normalization step which stops the dynamic range from being compressed
        too much due to pixel outliers.
        """

        self.mode = mode
        self.type = type
        self.truncate = truncate

    def __call__(self, img: torch.Tensor):

        mean = PIXEL_MEAN if self.mode == "global" else img.mean()
        std = PIXEL_STD if self.mode == "global" else img.std()

        img = (img - mean) / std

        if self.truncate:
            img = torch.clamp(img, -4, 4)

        if self.type == "z-score":
            return img
        else:
            return (img - img.min()) / (img.max() - img.min())


class TensorImageAugmentation:
    def __init__(
        self,
        prob=0.2,
        random_erasing=True,
        random_invert=True,
        random_horizontal_flip=True,
        random_vertical_flip=True,
        random_affine_translation=[0.2, 0.2],
        random_affine_rotation=0,
        random_affine_shear=[0, 0, 0, 0],
        random_resized_crop=False,
        random_resized_crop_scale=[0.7, 1],
        out_size=(256, 256),
    ):

        augs = []

        if random_erasing:
            augs.append(
                T.RandomErasing(
                    p=prob, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=0.5
                )  # type:ignore
            )
        if random_invert:
            augs.append(T.RandomInvert(p=prob))
        if random_horizontal_flip:
            augs.append(T.RandomHorizontalFlip(p=prob))
        if random_vertical_flip:
            augs.append(T.RandomVerticalFlip(p=prob))

        augs.append(
            T.RandomAffine(
                degrees=random_affine_rotation,
                translate=random_affine_translation,
                shear=random_affine_shear,
                fill=0.5,  # type:ignore
            )
        )

        if random_resized_crop:
            augs.append(
                T.RandomResizedCrop(tuple(out_size), scale=random_resized_crop_scale)
            )

        self.aug = T.Compose(augs)

    def __call__(self, img: torch.Tensor):
        return self.aug(img)


class ToTensor:
    def __init__(self, resize_to=(256, 256)):
        self.resize_to = resize_to
        self.to_tensor = T.ToTensor()

    def __call__(self, array: np.ndarray):
        array = resize(array, self.resize_to)
        return self.to_tensor(array).float()


class Transform:
    def __init__(
        self,
        use_augmentations: bool = True,
        create_pairs: bool = False,
        random_phase_shift=False,
        random_phase_distort=False,
        random_phase_distort_strength=0.1,
        random_phase_distort_freq_limit=0.3,
        random_envelope_distort=False,
        random_envelope_distort_strength=0.2,
        random_envelope_distort_freq_limit=0.1,
        random_bandstop=False,
        random_bandstop_width=0.1,
        random_freq_stretch=False,
        random_freq_stretch_range=(0.98, 1.0),
        normalize_mode: Literal["instance", "global"] = "instance",
        normalize_type: Literal["z-score", "min-max"] = "min-max",
        normalize_truncate: bool = True,
        prob: float = 0.2,
        random_erasing: bool = True,
        random_invert: bool = True,
        random_horizontal_flip: bool = True,
        random_vertical_flip: bool = True,
        random_affine_translation: Union[float, Tuple[float, float]] = (0.2, 0.2),
        random_affine_rotation: int = 0,
        random_affine_shear: List[int] = [0, 0, 0, 0],
        random_resized_crop: bool = False,
        random_resized_crop_scale: Union[float, Tuple[float, float]] = (0.7, 1),
        out_size=(256, 256),
    ):
        """Preprocessing transform for RF data patches"""

        if isinstance(random_affine_translation, float):
            random_affine_translation = tuple([random_affine_translation] * 2)

        if isinstance(random_resized_crop_scale, float):
            random_resized_crop_scale = tuple([random_resized_crop_scale, 1])

        self.to_tensor = ToTensor(resize_to=out_size)

        self.normalize = Normalize(
            normalize_mode,
            normalize_type,
            truncate=normalize_truncate,
        )

        self.ultrasound_augmentations = (
            UltrasoundArrayAugmentation(
                random_phase_shift=random_phase_shift,
                random_phase_distort=random_phase_distort,
                random_phase_distort_strength=random_phase_distort_strength,
                random_phase_distort_freq_limit=random_phase_distort_freq_limit,
                random_envelope_distort=random_envelope_distort,
                random_envelope_distort_strength=random_envelope_distort_strength,
                random_envelope_distort_freq_limit=random_envelope_distort_freq_limit,
                random_bandstop=random_bandstop,
                random_bandstop_width=random_bandstop_width,
                random_freq_stretch=random_freq_stretch,
                random_freq_stretch_range=random_freq_stretch_range,
            )
            if use_augmentations
            else None
        )

        self.tensor_augmentations = (
            TensorImageAugmentation(
                prob,
                random_erasing=random_erasing,
                random_invert=random_invert,
                random_horizontal_flip=random_horizontal_flip,
                random_vertical_flip=random_vertical_flip,
                random_affine_translation=random_affine_translation,
                random_affine_rotation=random_affine_rotation,
                random_affine_shear=random_affine_shear,
                random_resized_crop=random_resized_crop,
                random_resized_crop_scale=random_resized_crop_scale,
                out_size=out_size,
            )
            if use_augmentations
            else None
        )

        self.create_pairs = create_pairs

    def __transform(self, array):

        if self.ultrasound_augmentations:
            array = self.ultrasound_augmentations(array)
        array = self.to_tensor(array)  # type:ignore
        array = self.normalize(array)
        if self.tensor_augmentations:
            array = self.tensor_augmentations(array)
        return array

    def __call__(self, array):
        if self.create_pairs:
            return self.__transform(array), self.__transform(array)
        else:
            return self.__transform(array)


def target_transform(label: bool):
    return torch.tensor(label).long()


# class TransformV2:
#    def __init__(self, config: TransformConfig):
#        self.normalize = Normalize(**asdict(config.norm_config))
#        if config.tensor_augs_config is not None:
#            self.tensor_augs = TensorImageAugmentation(
#                **asdict(config.tensor_augs_config), out_size=config.out_size
#            )
#        else:
#            self.tensor_augs = None
#        self.to_tensor = ToTensor(config.out_size)
#        if config.us_augs_config is not None:
#            self.ultrasound_augs = UltrasoundArrayAugmentation(
#                **asdict(config.us_augs_config)
#            )
#        else:
#            self.ultrasound_augs = None
#
#    def __call__(self, img):
#        if self.ultrasound_augs:
#            img = self.ultrasound_augs(img)
#        img = self.to_tensor(img)  # type:ignore
#        img = self.normalize(img)
#        if self.tensor_augs:
#            img = self.tensor_augs(img)
#        return img


class TransformV3:
    def __init__(
        self,
        out_size: Tuple[int, int] = (256, 256),
        norm: Callable = Normalize(),
        tensor_transform: Optional[Callable] = None,
        us_augmentation: Optional[Callable] = None,
    ):

        self.normalize = norm
        self.out_size = out_size
        self.tensor_transform = tensor_transform
        self.us_augmentation = us_augmentation
        self.to_tensor = ToTensor(self.out_size)

    def __call__(self, img):
        if self.us_augmentation:
            img = self.us_augmentation(img)
        img = self.to_tensor(img)  # type:ignore
        img = self.normalize(img)
        if self.tensor_transform:
            img = self.tensor_transform(img)
        return img


class MultiTransform:
    def __init__(self, *transforms):
        self._transforms = transforms

    def __call__(self, img):
        return tuple([t(img) for t in self._transforms])


_BASELINE_TENSOR_AUGS = TensorAugsConfig(
    prob=0.5,
)

_CROPS_TENSOR_AUGS = TensorAugsConfig(
    prob=0.5,
    random_invert=True,
    random_vertical_flip=False,
    random_affine_translation=(0, 0),
    random_resized_crop=True,
    random_resized_crop_scale=(0.5, 1),
)

_STANDARD_US_AUGS = UltrasoundAugsConfig(
    random_phase_shift=True,
    random_envelope_distort=True,
    random_freq_stretch=True,
    random_envelope_distort_strength=0.5,
)


# class PrebuiltConfigs:
#
#    BASELINE = TransformConfig(tensor_augs_config=_BASELINE_TENSOR_AUGS)
#
#    BASELINE_PLUS_US = TransformConfig(
#        tensor_augs_config=_BASELINE_TENSOR_AUGS,
#        us_augs_config=_STANDARD_US_AUGS,
#    )
#    CROPS = TransformConfig(tensor_augs_config=_CROPS_TENSOR_AUGS)
#
#    CROPS_PLUS_US = TransformConfig(
#        tensor_augs_config=_CROPS_TENSOR_AUGS,
#        us_augs_config=_STANDARD_US_AUGS,
#    )
#
#    US_AUGS_ONLY = TransformConfig(us_augs_config=_STANDARD_US_AUGS)
#
#    NO_AUGS = TransformConfig()
