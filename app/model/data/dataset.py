# Define basic libs
from pathlib import Path
from os.path import splitext, isfile, join, exists
from os import listdir
import logging
import random

# Define non-torch libs
import numpy as np
from PIL import Image
import pandas as pd

# Define torch-based libs
import torch
from torch.utils.data import Dataset

# Define local libs
...


def load_image(filename):
    """_summary_

    :param filename: _description_
    :type filename: _type_
    :return: _description_
    :rtype: _type_
    """
    ext = splitext(filename)[1]
    if ext == ".npy":
        return Image.fromarray(np.load(filename)).convert("L")
    elif ext in [".pt", ".pth"]:
        return Image.fromarray(torch.load(filename).numpy()).convert("L")
    else:
        return Image.open(filename).convert("L")


class KitsSlicesDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = "",
        transform=None,
        split: str = "train",
        random_state: int | None = None,
    ):
        """_summary_

        :param images_dir: _description_
        :type images_dir: str
        :param mask_dir: _description_
        :type mask_dir: str
        :param scale: _description_, defaults to 1.0
        :type scale: float, optional
        :param mask_suffix: _description_, defaults to ""
        :type mask_suffix: str, optional
        :param transform: _description_, defaults to None
        :type transform: _type_, optional
        :param split: _description_, defaults to "train"
        :type split: str, optional
        :param percent: _description_, defaults to 100
        :type percent: int, optional
        :raises RuntimeError: _description_
        """

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.split = split
        self.random_state = random_state

        assert 0 < scale <= 1, f"Assertion: Scale must be between 0 and 1, got {scale}"
        assert split in [
            "train",
            "val",
            "test",
            "all",
        ], f"Assertion: Split must be in ['train', 'val', 'test', 'all'], got {split}"

        # Сбор идентификаторов файлов
        self.raw_ids = [
            splitext(file)[0]
            for file in listdir(images_dir)[
                : int(len(listdir(images_dir)))
            ]
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]

        if not self.raw_ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        # Фиксация random state
        if self.random_state is not None:
            torch.random.seed(random_state)

        # Перемешивание идентификаторов
        random.shuffle(self.raw_ids)

        # Разделение данных на выборки
        if self.split == "train":
            self.ids = self.raw_ids[: int(len(self.raw_ids) * 0.8)]
        elif self.split == "val":
            self.ids = self.raw_ids[
                int(len(self.raw_ids) * 0.8) : int(len(self.raw_ids) * 0.9)
            ]
        elif self.split == "test":
            self.ids = self.raw_ids[int(len(self.raw_ids) * 0.9) :]
        else:
            self.ids = self.raw_ids

        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """_summary_

        :param idx: _description_
        :type idx: _type_
        :return: _description_
        :rtype: _type_
        """
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(img_file) == 1
        ), f"Assertion: Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Assertion: Either no mask or multiple masks found for the ID {name}: {mask_file}"

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert (
            img.size == mask.size
        ), f"Assertion: Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        # Apply transformations if provided
        if self.transform:
            img, mask = self.transform.transform(img, mask)

        return img, mask
        # return (
        #     torch.as_tensor(img).float().contiguous(),
        #     torch.as_tensor(mask).long().contiguous(),
        # )


class Kits23Dataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        root_dir: str | Path,
        transform=None,
        split: str = "train",
        part: float = 1.0,
    ):

        # Проверка на очевидность
        assert Path(csv_file).exists(), f"{csv_file} does not exist"
        assert Path(root_dir).exists(), f"{root_dir} does not exist"
        assert split in ["train", "val", "test"], f"{split} is not valid"
        assert (Path(root_dir) / "img").exists(), f"{root_dir}/img does not exist"
        assert (Path(root_dir) / "seg").exists(), f"{root_dir}/seg does not exist"

        # Инициализация параметров
        self.csv_path = Path(csv_file)
        self.part = part
        self.full_df = pd.read_csv(self.csv_path, encoding="utf-8", index_col=0)
        self.dataset_df = self.full_df.iloc[:int(self.full_df.shape[0] * self.part)]
        self.dataset_dir = Path(root_dir)
        self.img_dir = self.dataset_dir / "img"
        self.seg_dir = self.dataset_dir / "seg"
        self.transform = transform
        self.split = split
        self.max_case = self.dataset_df["case"].max()

        # Отделяем кусок датафрейма
        if self.split == "train":
            self.dataset_split = self.dataset_df[
                self.dataset_df["case"] < int(self.max_case * 0.8)
            ]
        elif self.split == "val":
            self.dataset_split = self.dataset_df[
                (self.dataset_df["case"] >= int(self.max_case * 0.8))
                & (self.dataset_df["case"] < int(self.max_case * 0.9))
            ]
        else:
            self.dataset_split = self.dataset_df[
                self.dataset_df["case"] >= int(self.max_case * 0.9)
            ]

    def __len__(self):
        return self.dataset_split.shape[0]

    def __getitem__(self, index):
        img_path = self.dataset_split.iloc[index]["img_path"]
        seg_path = self.dataset_split.iloc[index]["seg_path"]

        img = torch.load(img_path, weights_only=True).unsqueeze(0)
        seg = torch.load(seg_path, weights_only=True).unsqueeze(0)

        assert img.shape == seg.shape, f"{img.shape} != {seg.shape}"

        if self.transform:
            img, seg = self.transform(img, seg)

        return img.float().contiguous(), seg.long().contiguous()