# |---------------------------------------|
# | CONVERT MODEL FROM CHECKPOINT TO ONNX |
# |---------------------------------------|

# First, import the necessary libraries:
#   1. General libraries:
import torch
from pathlib import Path
import logging

#   2. Local libraries:
from app.models.segmentation_model import DeepLab
from app.constants import CHECKPOINT, ONNX_MODEL


# Define a function to export the model to ONNX:
def ckpt_to_onnx(ckpt_path: str | Path, onnx_path: str | Path) -> None:
    # Read model from checkpoint:
    model = DeepLab.load_from_checkpoint(ckpt_path)
    model.cpu()
    model.eval()

    # Define example input (batch size is 16 as it was during training):
    batch_size = 4
    x = torch.randn(batch_size, 1, 512, 512)
    y = model(x)

    # Define export options for enabling dynamic shapes of the input:
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            onnx_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    logging.info("Exported ONNX model to {}".format(onnx_path))
