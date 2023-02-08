from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
)

from warnings import warn


CIFAR10_NORM = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_NORM = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
STL10_NORM = (0.4467, 0.4398, 0.4066), (0.2603, 0.2564, 0.2712)
IMAGENET_NORM = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class SimCLRAugmentationsForNaturalImages:
    """Default SimCLR augmentations"""

    def __init__(self, image_size=32, mean=None, std=None):
        if mean is None or std is None:
            # use imagenet mean and std
            warn("No mean and std provided, using imagenet mean and std")
            mean, std = IMAGENET_NORM

        self.image_size = image_size
        self.augmentations = Compose(
            [
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(),
                ColorJitter(0.8, 0.8, 0.8, 0.2),
                GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        return self.augmentations(x)


class MultiViewTransform:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class ToTensorAndNormalize:
    def __init__(self, mean, std):
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, x):
        return self.normalize(self.to_tensor(x))
