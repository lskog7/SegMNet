# |----------------------------------------------|
# | SEGMENTATION SERVICE DESCRIBES PROCESS LOGIC |
# |----------------------------------------------|

import time
import logging
import nibabel as nib
from pathlib import Path

from app.constants import ONNX_MODEL, TMP_PATH
from app.models.segmentation_model import Loader, Inference

DataLoader = Loader(TMP_PATH)
SegmentationModel = Inference(ONNX_MODEL)


def process_nifti(input_path: str | Path, output_path: str | Path, model):
    try:
        # Load NIfTI file:
        nifti_img = nib.load(str(input_path))
        logging.info(f"NIfTI file {input_path.name} loaded successfully.")

        # Simulate segmentation logic:
        time.sleep(5)  # Placeholder for real processing
        result_data = nifti_img.get_fdata()  # Placeholder for model inference
        nib.save(nib.Nifti1Image(result_data, nifti_img.affine), str(output_path))
        logging.info(f"Segmentation result saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        raise
