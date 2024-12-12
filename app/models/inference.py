# |----------------------------------------------|
# | BASE CLASS FOR INFERENCE MODEL IN PRODUCTION |
# |----------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# TODO: Rewrite to pure torch

# Libraries:
import numpy as np
import torch
import logging
import onnxruntime
from tqdm import tqdm

# Local modules:
from app.constants.constants import ONNX_MODEL

# Module-specific logging template:
logging.basicConfig(level=logging.INFO, format="MODULE->[inference.py]: %(message)s")


# Define base class for segmentation inference in production:
class Inference:
    def __init__(self):
        self.onnx_model = ONNX_MODEL
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    def __repr__(self) -> str:
        return f"ModelRepresentation(name=UNet++, encoder=EfficientNet-b0)"

    def predict(
            self, image: torch.Tensor, return_logits: bool = False
    ) -> np.ndarray:
        """
        Predicts the output for a batch of images using an ONNX model.

        Args:
            image (torch.Tensor): A tensor of shape (B, C, H, W) where B is the batch size,
                                  C is the number of channels, H is the height, and W is the width.
            return_logits (bool): If True, returns the raw logits; if False, returns the predicted classes.

        Returns:
            np.ndarray: An array of predicted classes or logits, depending on the value of return_logits.
        """

        # Ensure the input image tensor has 4 dimensions
        assert image.ndim == 4, f"MODULE->[inference.py]: Image must have 4 dimensions like (B, C, H, W). Got {image.shape}."

        logging.info(f"Predicting image with shape {image.shape}")

        # Prepare lists to collect logits and predictions
        logits_list = []
        predictions_list = []

        # Convert the entire batch of images to a NumPy array to minimize conversion overhead
        image_np = image.numpy()

        # Iterate over each image in the batch
        for index in tqdm(range(image.shape[0])):
            # Prepare the input for the ONNX model
            ort_inputs = {
                self.ort_session.get_inputs()[0].name: image_np[index:index + 1],  # Use slicing to get a single image
            }

            # Run the model and get the logits
            logits = torch.tensor(self.ort_session.run(None, ort_inputs)[0])

            # Store logits or predictions based on the return_logits flag
            if return_logits:
                logits_list.append(logits)
            else:
                pred = logits.argmax(1)  # Get the predicted class by finding the index of the max logit
                predictions_list.append(pred)

        # Return the concatenated results as a NumPy array
        if return_logits:
            return torch.cat(logits_list, dim=0).numpy()  # Convert logits to NumPy array before returning
        else:
            return torch.cat(predictions_list, dim=0).numpy()  # Convert predictions to NumPy array before returning