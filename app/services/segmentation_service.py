# |----------------------------------------------|
# | SEGMENTATION SERVICE DESCRIBES PROCESS LOGIC |
# |----------------------------------------------|

import logging
import nibabel as nib
from pathlib import Path
import numpy as np

from app.constants import ONNX_MODEL, TMP_PATH
from app.models.segmentation_model import Loader, Inference

DataLoader = Loader(TMP_PATH)
SegmentationModel = Inference(ONNX_MODEL)


def process_nifti(input_path: str | Path, output_path: str | Path) -> np.ndarray:
    input_path = Path(input_path)
    output_path = Path(output_path)
    try:
        # Load file from the input path:
        nifti_image, affine = DataLoader.load_nifti(input_path)
        base_size = nifti_image.shape[1:]
        logging.info(f"nifti image base shape: {nifti_image.shape}")

        if len(nifti_image) == 0 or len(affine) == 0:
            raise ValueError(f"Zero size of nifti image or its affine:\n"
                             f"Nifti Image shape: {nifti_image.shape}\n"
                             f"Affine shape: {affine.shape}")

        # Preprocess input file:
        #   1. Convert 3D CT image to torch.Tensor and save size:
        nifti_tensor = DataLoader.convert_to_tensor(np.transpose(nifti_image, (1,2,0)))
        logging.info(f"nifti_tensor shape: {nifti_tensor.shape}")
        #   2. Apply windowing:
        nifti_tensor_w = DataLoader.apply_windowing(nifti_tensor)
        logging.info(f"nifti_tensor_w shape: {nifti_tensor_w.shape}")
        #   3. Apply train-like transformations (normalizations):
        nifti_tensor_wn = DataLoader.transform(nifti_tensor_w)
        logging.info(f"nifti_tensor_wn shape: {nifti_tensor_wn.shape}")

        # Do segmentation:
        segmentation = SegmentationModel.predict_nifti(nifti_tensor_wn.unsqueeze(1)).detach().cpu()

        # TODO: Check this function, I think it can work not as I expect.
        # Return to initial (base) size:
        segmentation_base_sized = SegmentationModel.to_base_size(segmentation, base_size=base_size)

        if segmentation.shape[0] != nifti_image.shape[0]:
            raise ValueError("Mismatch between input and output shape in segmentation!")

        # Save the result to TMP_PATH:
        nib.save(nib.Nifti1Image(segmentation.numpy().astype(np.float32), affine), str(output_path))

        # Return the result
        return segmentation

    except FileNotFoundError:

        logging.error(f"File not found: {input_path}")
        raise ValueError(f"NIFTI file not found: {input_path}")

    except ValueError as ve:

        logging.error(f"Invalid NIFTI file format: {input_path}. Error: {ve}")
        raise ve

    except Exception as e:

        logging.error(f"Unexpected error during NIFTI loading: {e}")
        raise
