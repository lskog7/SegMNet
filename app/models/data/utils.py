# |-----------------------------|
# | UTILITIES TO WORK WITH DATA |
# |-----------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# Libraries:
from pathlib import Path
import torch
import nibabel as nib
from torchvision.transforms import v2
from typing import Tuple, Callable
import logging

# Local modules:
...

# Module-specific logging template:
# logging.basicConfig(level=logging.INFO, format="MODULE->[utils.py]: %(message)s")


def _check_path(path: str | Path) -> Path:
    """
    Function to check if a given path is valid (i.e., it exists).

    Args:
        path (str | Path): The path to check, which can be a string or a Path object.

    Returns:
        Path: The validated path as a Path object.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """

    path = Path(path)

    # Check if the path exists:
    if path.exists():
        return path

    # Raise an error if the path does not exist:
    raise FileNotFoundError(f"MODULE->[utils.py]: Path {path} does not exist.")


def _load_nifti(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to load a NIfTI image file and convert it to PyTorch tensors.

    Args:
        path (str | Path): The path to the NIfTI file, which can be a string or a Path object.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor representation of the NIfTI image data.
            - A tensor representation of the affine transformation matrix.
    """

    # Check if the provided path is valid and exists:
    path = _check_path(path)  # Call the _check_path function to validate the path

    # Load the NIfTI image using nibabel:
    img = nib.load(path)  # Use nibabel to load the NIfTI file

    # Convert the image data and affine matrix to PyTorch tensors:
    array = torch.tensor(img.get_fdata(),
                         dtype=torch.float32)  # Get the image data as a NumPy array and convert it to a tensor
    affine = torch.tensor(img.affine, dtype=torch.float32)  # Convert the affine matrix to a tensor

    # Return the image data tensor and the affine tensor:
    return array, affine  # Return both tensors as a tuple


def _get_window_fn(W: int = 400, L: int = 50, mask: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Function to create a windowing transformation function.

    Args:
        W (int): The width of the window (default is 400).
        L (int): The level of the window (default is 50).
        mask (bool): If True, apply a mask instead of windowing (default is False).

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that applies windowing to an image tensor.
    """

    def window_fn(img: torch.Tensor) -> torch.Tensor:
        """
        Applies windowing to the input image tensor.

        Args:
            img (torch.Tensor): The input image tensor to which windowing will be applied.

        Returns:
            torch.Tensor: The modified image tensor after applying windowing.
        """
        # Create a copy of the array to avoid modifying the original
        modified_img = img.clone().detach()

        if not mask:
            # Calculate the upper and lower grey levels
            upper_grey_level = L + (W / 2)
            lower_grey_level = L - (W / 2)

            # Replace values below the minimum with the minimum
            modified_img[modified_img < lower_grey_level] = lower_grey_level

            # Replace values above the maximum with the maximum
            modified_img[modified_img > upper_grey_level] = upper_grey_level

        return modified_img  # Return the modified image tensor

    return window_fn  # Return the windowing function


def _get_inference_transform_fn():
    """
    Function to create a transformation pipeline for inference.

    Returns:
        v2.Compose: A composed transformation function that applies a series of transformations to an image.
    """

    return v2.Compose(
        [
            v2.ToImage(),  # Convert the input data to an image format (if not already in image format)
            v2.ToDtype(torch.float32, scale=True),
            # Convert the image to a float32 tensor and scale pixel values to [0, 1]
            v2.Resize(
                size=(512, 512),  # Resize the image to 512x512 pixels
                interpolation=v2.InterpolationMode.BILINEAR,  # Use bilinear interpolation for resizing
                antialias=True,  # Apply antialiasing to reduce aliasing artifacts
            ),
            v2.Normalize(mean=[-82.4897], std=[96.2965]),
            # Normalize the image with specified mean and standard deviation
        ]
    )
