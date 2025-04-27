from torchvision.datasets import CIFAR10 as _CIFAR10, CIFAR100 as _CIFAR100
from typing import Union, Optional, Callable
from pathlib import Path


class CIFAR10(_CIFAR10):
    dataset_name = "CIFAR10"

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super().__init__(Path(root)/self.dataset_name, train, transform, target_transform, download)


class CIFAR100(_CIFAR100):
    dataset_name = "CIFAR100"

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super().__init__(Path(root)/self.dataset_name, train, transform, target_transform, download)
