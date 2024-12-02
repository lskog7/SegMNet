# |-----------------------------------------------|
# | SPECIAL CLASS FOR DATA INTERACTION WITHIN APP |
# |-----------------------------------------------|

# TODO: Add mimetypes-like autocheck of file extensions.
# TODO: Check _image_totensor, _nifti_totensor, _apply_windowing.

from pathlib import Path
import nibabel as nib
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
import logging

from app.constants import TMP_PATH
from app.models.segmentation_model.data.utils import _image_totensor, _nifti_totensor, _Normalize, _check_path, \
    _get_transform, _apply_windowing


# Base dataloader class.
# It is used for loading, saving, opening NIFTIs convertion to torch.Tensors, etc.
# The only necessary parameter is TMP_FOLDER
class Loader:
    def __init__(self, tmp_path: str | Path = TMP_PATH):
        self.tmp_path = tmp_path
        self.tensor_fn = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transform = _get_transform()
        self.windowing_fn = _apply_windowing
        # TODO: Add documentation for affine_dict.
        self.affine_dict = {}

        if not self.tmp_path.exists():
            self.tmp_path.mkdir(parents=True, exist_ok=True)

    # TODO: Add fallback to check file extension.
    def load_nifti(self, nifti_path: str | Path):
        """
        Load 3D NIFTI image and its affine.
        Args:
            nifti_path:

        Returns:
            image
            affine
        """
        nifti_path = _check_path(nifti_path)
        if ".nii" in nifti_path.suffixes:
            try:
                nifti = nib.load(nifti_path)
            except FileNotFoundError:
                logging.error(f"File not found: {nifti_path}")
                raise ValueError(f"NIFTI file not found: {nifti_path}")
            except Exception as e:
                logging.error(f"Unexpected error while loading NIFTI: {e}")
                raise ValueError("Failed to load NIFTI file")

            file_info = {"shape": nifti.header.get_data_shape(),
                         "dtype": nifti.header.get_data_dtype(),
                         "affine": nifti.affine}
            self.affine_dict[str(nifti_path.name)] = file_info
            return nifti.get_fdata(), nifti.affine
        else:
            raise ValueError(f"Unsupported file type: {nifti_path.suffix}")

    @staticmethod
    def load_image(image_path: str | Path):
        """
        Load single 2D image.
        Args:
            image_path:

        Returns:

        """
        image_path = _check_path(image_path)
        if image_path.suffix.lower() in ['.npy', '.npz']:
            return Image.fromarray(np.load(image_path)).convert('L')
        elif image_path.suffix.lower() in ['.pt', '.pth']:
            return Image.fromarray(torch.load(image_path, weights_only=True).numpy()).convert('L')
        elif image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return Image.open(image_path).convert('L')
        else:
            raise ValueError(f'Unsupported file type: {image_path.suffix}')

    def convert_to_tensor(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """
        Convert 2D/3D image ot torch.Tensor.
        Args:
            image:

        Returns:

        """
        return self.tensor_fn(image)

    # TODO: Add information that only medical images are allowed.
    def apply_windowing(self, nifti: torch.Tensor) -> torch.Tensor:
        """
        Applies special windowing to CT image. Allows to see kidneys much better.
        Use only after transformation to torch.Tensor.

        Args:
            nifti:

        Returns:

        """
        assert nifti.ndim == 3, f"NIFTI file must be 3D dimensions, got {nifti.ndim}"
        return self.windowing_fn(nifti)

    def preporcess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Makes image similar to training data.
        Use only after transformation to torch.Tensor and windowing.

        Args:
            image:

        Returns:

        """
        return self.transform(image)
