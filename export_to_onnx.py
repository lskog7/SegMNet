# |---------------------------------------|
# | CONVERT MODEL FROM CHECKPOINT TO ONNX |
# |---------------------------------------|

# First, import the necessary libraries:
#   1. General libraries:
from pathlib import Path
import logging

#   2. Local libraries:
from app.model.deeplab.model import DeepLab
from app.constants.constants import CHECKPOINT, ONNX_MODEL
from app.models.segmentation_model.export_to_onnx import ckpt_to_onnx

if __name__ == "__main__":
    ckpt_path = Path("external_files/20241126_105820_deeplabv3plus_efficientnet-b5_100_16_0.01.ckpt")
    ckpt_to_onnx(ckpt_path, ONNX_MODEL)
    logging.info("ONNX model exported successfully!")
