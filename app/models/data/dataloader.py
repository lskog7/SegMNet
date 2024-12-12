# |-----------------------------------------------|
# | SPECIAL CLASS FOR DATA INTERACTION WITHIN APP |
# |-----------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|


# Libraries:
from pathlib import Path
import torch
import logging
from typing import Tuple, List
from torchvision.transforms.v2.functional import resize
import numpy as np

# Local modules:
from app.constants.constants import TMP_DIR
from app.models.data.utils import _get_inference_transform_fn, _get_window_fn, _load_nifti

# Module-specific logging template:
logging.basicConfig(level=logging.INFO, format="MODULE->[dataloader.py]: %(message)s")


# Base dataloader class.
# It is used for loading, saving, opening NIFTI, convertion to torch.Tensors, etc.
class DataLoader:
    def __init__(self):
        """
        Initializes the Loader class, setting up the temporary directory,
        transformation functions, and a dictionary to store affine matrices.
        """

        self.tmp_dir = TMP_DIR  # Set the temporary directory (assumed to be defined elsewhere)
        self.transform = _get_inference_transform_fn()  # Get the inference transformation function
        self.window = _get_window_fn()  # Get the windowing function

        # Dictionary for storing file names and their affine matrices
        self.affine_dict = {}

        # Dictionary to store initial sizes of an image
        self.sizes_dict = {}

    @staticmethod
    def load(nifti_path: str | Path):
        # Load the NIfTI image and its affine matrix
        image, affine = _load_nifti(nifti_path)  # Call the function to load the NIfTI file

        # Extract the image name from the path
        image_name = Path(nifti_path).name

        return image, affine, image.shape[1:]  # Return the loaded image tensor

    def window(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply windowing to the input image tensor.

        Args:
            image (torch.Tensor): The input image tensor to which windowing will be applied.

        Returns:
            torch.Tensor: The modified image tensor after applying windowing.
        """

        return self.window(image)  # Call the windowing function and return the result

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply transformations to the input image tensor.

        Args:
            image (torch.Tensor): The input image tensor to be transformed.

        Returns:
            torch.Tensor: The transformed image tensor.
        """

        return self.transform(image)  # Call the transformation function and return the result

    @staticmethod
    def resize_back(image: torch.Tensor | np.ndarray, size: List[int]):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        return resize(image, size).numpy()
