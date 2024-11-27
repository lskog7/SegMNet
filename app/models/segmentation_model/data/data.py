from pathlib import Path
import os
import torch
import nibabel as nib
from torchvision.transforms import v2
import numpy as np
from torchvision import tv_tensors
from PIL import Image
import matplotlib.pyplot as plt
from typing import List


def _check_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path {path} does not exist")


def _nifti_totensor(path: str | Path) -> torch.Tensor:
    path = _check_path(path)
    img = nib.load(path)
    return torch.tensor(img.get_fdata())


def _image_totensor(path: str | Path) -> torch.Tensor:
    path = _check_path(path)
    ext = os.path.splitext(path)[1]
    if ext == ".npy":
        return Image.fromarray(np.load(path)).convert("L")
    elif ext in [".pt", ".pth"]:
        return Image.fromarray(torch.load(path, weights_only=True).numpy()).convert("L")
    else:
        return Image.open(path).convert("L")


def _show(img: torch.Tensor):
    plt.imshow(img[len(img[:, 0, 0]) // 2], cmap="gray")
    plt.axis("off")
    plt.show()


def _show_all(tensors: List[torch.Tensor]):
    # Определяем количество тензоров
    num_tensors = len(tensors)

    # Определяем размер сетки для отображения
    cols = int(np.ceil(np.sqrt(num_tensors)))  # Количество столбцов
    rows = int(np.ceil(num_tensors / cols))  # Количество строк

    # Создаем фигуру и оси
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Превращаем axes в одномерный массив для удобства
    axes = axes.flatten()

    for i, tensor in enumerate(tensors):
        # Отображаем тензор
        axes[i].imshow(
            tensor[len(tensor[:, 0, 0]) // 2], cmap="gray"
        )  # Используем серую цветовую карту для изображений
        axes[i].axis("off")  # Отключаем оси

    # Отключаем оси для оставшихся пустых подграфиков
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()  # Убираем лишние отступы
    plt.show()  # Отображаем


def _apply_windowing(img: torch.Tensor, W: int = 400, L: int = 50, mask: bool = False):
    # Создаем копию массива, чтобы не изменять оригинал
    modified_img = img.clone().detach()

    if not mask:

        # Вычисляем верхний и нижний уровни серого
        upper_grey_level = L + (W / 2)
        lower_grey_level = L - (W / 2)

        # Заменяем значения ниже минимума на минимум
        modified_img[modified_img < lower_grey_level] = lower_grey_level

        # Заменяем значения выше максимума на максимум
        modified_img[modified_img > upper_grey_level] = upper_grey_level

    return modified_img


def _get_transform():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=(512, 512),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.Normalize(mean=[-82.4897], std=[96.2965]),
            _Normalize(),
        ]
    )


class _Normalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized_tensor = tensor.clone().detach()
        # print(type(normalized_tensor))
        if not isinstance(normalized_tensor, tv_tensors._mask.Mask):
            # Проверяем, чтобы избежать деления на ноль
            if normalized_tensor.max() - normalized_tensor.min() > 0:
                normalized_tensor = tv_tensors._image.Image(
                    (normalized_tensor - normalized_tensor.min())
                    / (normalized_tensor.max() - normalized_tensor.min())
                )
        return normalized_tensor
