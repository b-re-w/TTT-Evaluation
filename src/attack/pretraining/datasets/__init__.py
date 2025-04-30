from .imagenet1k import *
from .cifars import *

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from torchvision import transforms, utils
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


# Image Visualizer
def show_sample_grid(self, mean, std):
    images, targets = next(iter(self))
    grid_images = utils.make_grid(images, nrow=8, padding=10)
    np_image = np.array(grid_images).transpose((1, 2, 0))
    de_norm_image = np_image * std + mean
    plt.figure(figsize=(10, 10))
    plt.imshow(de_norm_image)

DataLoader.show_sample_grid = show_sample_grid


def dataset_split(dataset, split=0.1, new_transforms=None, random_seed=42):
    """
    Splits the dataset into train and validation sets.
    :param dataset: The dataset to split.
    :param split: The fraction of the dataset to use for validation.
    :param new_transforms: Optional new transforms to apply to the validation set.
    :param random_seed: The random seed for reproducibility.
    :return: A tuple of (train_dataset, val_dataset).
    """
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=split,
        random_state=random_seed,
        shuffle=True,
        stratify=dataset.targets if hasattr(dataset, 'targets') else None
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = deepcopy(dataset)
    if new_transforms:
        val_dataset.transform = new_transforms
    val_dataset = Subset(val_dataset, val_indices)

    return train_dataset, val_dataset


class DatasetHolder:
    def __init__(self, config, train, test, valid=None, attack=None):
        self.config = config
        self.train = train
        self.test = test
        self.valid = valid
        self.attack = attack
        self.num_classes = len(train.classes) if hasattr(train, 'classes') else len(test.classes)

    def split_train_valid(self, split=0.1, random_seed=42):
        if self.valid is None:
            self.train, self.valid = dataset_split(self.train, split, self.config.resizing, random_seed)
        else:
            raise ValueError("Validation set already exists. Use a different split value.")

    def split_train_attack(self, split=0.1, random_seed=42):
        if self.attack is None:
            self.train, self.attack = dataset_split(self.train, split, self.config.resizing, random_seed)
        else:
            raise ValueError("Attack set already exists. Use a different split value.")

    def __repr__(self):
        return f"{self.config.name.upper().replace('-', '')}DatasetHolder(train={len(self.train)}, test={len(self.test)}, valid={len(self.valid) if self.valid else 'None'}, attack={len(self.attack) if self.attack else 'None'})"


class DatasetConfig(dict):
    def __init__(self, name, size, norm, epoch, augmenter=None, resizer=None):
        super().__init__()
        self.name = name
        self.size = size
        self.norm = norm
        self.epoch = epoch
        self.augmentation = augmenter(size, norm) if augmenter else self.augmenter(size, norm)
        self.resizing = resizer(size, norm) if resizer else self.resizer(size, norm)

        self.FOR_SWIN = None

    def __repr__(self):
        return f"DatasetConfig(size={self.size}, norm={self.norm})"

    @classmethod
    def augmenter(cls, size, norm):
        return transforms.Compose([
            transforms.RandomResizedCrop(size),  # Resize Image
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),  # Convert Image to Tensor
            transforms.Normalize(**norm)  # Normalization
        ])

    @classmethod
    def resizer(cls, size, norm):
        return transforms.Compose([
            transforms.Resize(size),  # Resize Image
            transforms.ToTensor(),  # Convert Image to Tensor
            transforms.Normalize(**norm)  # Normalization
        ])

    @classmethod
    def centered_resizer(cls, size1, size2, norm):
        return transforms.Compose([
            transforms.Resize(size1),  # Resize Image
            transforms.CenterCrop(size2),  # Center Crop Image
            transforms.ToTensor(),  # Convert Image to Tensor
            transforms.Normalize(**norm)  # Normalization
        ])


CIFAR10Config = DatasetConfig(
    name=CIFAR10.dataset_name,
    size=32,
    norm=dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    epoch=200
)
CIFAR10Config.FOR_SWIN = DatasetConfig(CIFAR10Config.name, 224, CIFAR10Config.norm, CIFAR10Config.epoch)

CIFAR100Config = DatasetConfig(
    name=CIFAR100.dataset_name,
    size=32,
    norm=dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    epoch=1000
)
CIFAR100Config.FOR_SWIN = DatasetConfig(CIFAR100Config.name, 224, CIFAR100Config.norm, CIFAR100Config.epoch)

IMAGENET1KConfig = DatasetConfig(
    name=ImageNet1K.dataset_name,
    size=224,
    norm=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    resizer=lambda s, n: DatasetConfig.centered_resizer(256, s, n),
    epoch=1000
)
