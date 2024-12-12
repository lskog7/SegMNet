# |-------------------------------------------|
# | DATASET CLASS FOR TRAINING AND EVALUATION |
# |-------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# Libraries:
from pathlib import Path
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple

# Local modules:
...

# Module-specific logging template:
logging.basicConfig(level=logging.INFO, format="MODULE->[dataset.py]: %(message)s")


# Basic class for the KiTS23 Dataset.
# Mostly used for training and validation after it.
# Generally useless now.
class Kits23Dataset(Dataset):
    # Dataset class needs to know only a few things to go:
    # 1. csv_path - path to a CSV file where image paths and labels are stored.
    # 2. images_dir - path to a folder with all images.
    # 3. transform - just a transformation function.
    # 4. split - train, val or test split.
    # 5. part - how much of a dataset to use.
    def __init__(
            self,
            csv_path: str | Path,
            images_dir: str | Path,
            transform=None,
            split: str = "train",
            part: float = 1.0,
    ):
        """
        Basic class for KiTS23 dataset.
        Mostly used for training and validation after it.
        Generally useless now.
        Args:
            csv_path: Path to a CSV file where image paths and labels are stored.
            images_dir: Path to a folder with all images.
            transform: Just a transformation function.
            split: Train, val or test split.
            part: How much of a dataset to use.
        """

        # Check if all folders exist:
        assert Path(csv_path).exists(), f"MODULE->[dataset.py]: {csv_path} does not exist"
        assert Path(images_dir).exists(), f"MODULE->[dataset.py]: {images_dir} does not exist"
        assert (Path(images_dir) / "img").exists(), f"MODULE->[dataset.py]: {images_dir}/img does not exist"
        assert (Path(images_dir) / "seg").exists(), f"MODULE->[dataset.py]: {images_dir}/seg does not exist"

        # Check if split is right:
        assert split in ["train", "val", "test"], f"MODULE->[dataset.py]: {split} is not valid"

        # Define initial local variables:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.split = split
        self.part = part

        # Define paths to images and segmentations:
        self.img_dir = self.images_dir / "img"
        self.seg_dir = self.images_dir / "seg"

        # Open DataFrame and get information about data:
        # 1. Read the whole DataFrame:
        self.df_full = pd.read_csv(self.csv_path, encoding="utf-8", index_col=0)

        # 2. Get the desired part of a DataFrame:
        self.df_part = self.df_full.iloc[:int(self.df_full.shape[0] * self.part)]

        # 3. Get the max index of cases:
        self.max_case = self.df_part["case"].max()

        # Get a DataFrame split:
        # 80% for train, 10% for both val and test.
        if self.split == "train":
            self.dataset_split = self.df_part[
                self.df_part["case"] < int(self.max_case * 0.8)
                ]
        elif self.split == "val":
            self.dataset_split = self.df_part[
                (self.df_part["case"] >= int(self.max_case * 0.8))
                & (self.df_part["case"] < int(self.max_case * 0.9))
                ]
        else:
            self.dataset_split = self.df_part[
                self.df_part["case"] >= int(self.max_case * 0.9)
                ]

    # Get length of dataset:
    def __len__(self) -> int:
        """
        Basic function to get the length of a dataset.
        Returns:
            Length of the dataset.
        """

        return self.dataset_split.shape[0]

    # Main function for getting an item from a dataset:
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Basic function to get a dataset item.
        Args:
            index: Index of the item to get.

        Returns:
            (Image, Segmentation) pair.
        """

        # Get paths to image and segmentation.
        img_path = Path(self.dataset_split.iloc[index]["img_path"])
        seg_path = Path(self.dataset_split.iloc[index]["seg_path"])

        # Load both image and segmentation (with torch):
        # They are saved as torch tensors so we use torch.load():
        img = torch.load(img_path, weights_only=True).unsqueeze(0)
        seg = torch.load(seg_path, weights_only=True).unsqueeze(0)

        # Check if image and segmentations shapes match:
        assert img.shape == seg.shape, f"{img.shape} != {seg.shape}"

        # Apply transforms if they are provided:
        if self.transform:
            img, seg = self.transform(img, seg)

        # Images must have torch.float32 format.
        # But segmentations must be integer (or torch.long)
        img = img.to(dtype=torch.float32)
        seg = seg.to(dtype=torch.long)

        return img, seg
