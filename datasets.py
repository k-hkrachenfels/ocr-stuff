import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError


import numpy as np

from PIL import Image

import torch
import torchvision
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class DigitDataset(VisionDataset):
    """Dataset for digit recognition
    Args:
        root (string): Root directory of dataset 
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.root = root
        self.train = train  
        self.data, self.targets = self._load_data()

    def _load_data(self):   
        targets = []
        label_file_name = f"{self.root}/labels.txt"
        if os.path.exists(label_file_name):
            with open(label_file_name) as label_file:
                for line in label_file:
                    targets.append(int(line))
            dataset_len = len(targets)
        else:
            # no labels in case we apply model for inference
            dataset_len = 9
            targets=[0]*9


        data = []
        for i in range(dataset_len):
            img = Image.open(f"{self.root}/img_{i}.png")
            data.append(img)

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
