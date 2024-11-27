# |----------------------------------------------|
# | BASE CLASS FOR INFERENCE MODEL IN PRODUCTION |
# |----------------------------------------------|

# TODO: Comment all the methods.

# First, import the necessary libraries:
#   1. General libraries:
import numpy as np
import torch
from pathlib import Path
import torchvision.tv_tensors as tv
import logging
import onnxruntime
from tqdm import tqdm
from pydantic import BaseModel

#   2. Local libraries:
from .data import (
    _image_totensor,
    _nifti_totensor,
    _get_transform,
    _apply_windowing,
)


# Deine base class for model inference in production:
class Inference:
    """
    The Inference class serves as a tool for performing logical inference and solving logical problems.

    Methods:
    - __init__(self): Initializes an instance of the Inference class.
    - add_clause(self, clause): Adds a new clause to the knowledge base.
    - add_clauses(self, clauses): Adds a list of clauses to the knowledge base.
    - solve(self): Solves a logical problem using the resolution method.
    - is_consistent(self): Checks if the knowledge base is consistent.
    - is_entailed(self, clause): Checks if a clause is entailed by the knowledge base.
    """

    def __init__(self, onnx_model: str | Path):
        """
        Initializes the Inference object with the specified ONNX model.

        Args:
            onnx_model (str | Path): The path to the ONNX model.

        Attributes:
            onnx_model (str | Path): The path to the ONNX model.
            ort_session (onnxruntime.InferenceSession): The ONNX model session.
            input_name (str): The name of the model's input layer.
            transform (callable): A function for transforming input data.
        """

        self.onnx_model = onnx_model
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.transform = _get_transform()
        self.initialized = True
        self.name = "SegMNet_v0.1"
        self.encoder = "efficientnet-b5"
        self.decoder = "deeplabv3+"

    def __repr__(self) -> str:
        """
        The __repr__ method returns a string representation of the Inference object.

        Returns:
            str: A string representation of the Inference object.
        """
        return f"ModelRepresentation(name={self.name}, encoder={self.encoder}, decoder={self.decoder})"

    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Predicts the output for a given image tensor.

        Args:
            image (torch.Tensor): A 4-dimensional tensor representing the image.

        Returns:
            np.ndarray: The predicted class indices for the image.
        """

        assert image.ndim == 4
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(image)}
        return self.ort_session.run(None, ort_inputs)[0].argmax(1).squeeze(0)

    def predict_nifti(
        self, nifti: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        """
        Predicts the output for a given NIFTI tensor.

        Args:
            nifti (torch.Tensor): A 4-dimensional tensor representing the NIFTI image.

        Returns:
            np.ndarray: The predicted class indices for the NIFTI image.
        """

        assert nifti.ndim == 4
        logging.info(f"Predicting NIFTI image with shape: {nifti.shape}")
        if return_logits:
            logits_list = []
        else:
            predictions_list = []
        for index in tqdm(range(nifti.shape[0])):
            ort_inputs = {
                self.ort_session.get_inputs()[0].name: self.to_numpy(
                    nifti[index].unsqueeze(0)
                )
            }
            logits = torch.tensor(self.ort_session.run(None, ort_inputs)[0])
            if return_logits:
                logits_list.append(logits)
            else:
                pred = logits.argmax(1)
                predictions_list.append(pred)
        if return_logits:
            return torch.cat(logits_list, dim=0)
        else:
            return torch.cat(predictions_list, dim=0)

    def predict_logits_nifti(self, nifti: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load_image(self, image_path: str | Path) -> torch.Tensor:
        """
        Loads an image from the specified path and applies the necessary transformations.

        Args:
            image_path (str | Path): The path to the image file.

        Returns:
            torch.Tensor: A tensor representation of the loaded image.
        """

        logging.info(f"Loading image from {image_path}")
        return (
            self.transform(tv._image.Image(_image_totensor(image_path)))
            .unsqueeze(0)
            .contiguous()
        )

    def load_nifti_img(self, nifti_path: str | Path) -> torch.Tensor:
        """
        Loads a NIFTI image from the specified path and applies the necessary transformations.

        Args:
            nifti_path (str | Path): The path to the NIFTI file.

        Returns:
            torch.Tensor: A tensor representation of the loaded NIFTI image.
        """

        logging.info(f"Loading image from NIFTI file: {nifti_path}")
        return (
            self.transform(
                _apply_windowing(tv._image.Image(_nifti_totensor(nifti_path)))
            )
            .unsqueeze(1)
            .contiguous()
        )

    @staticmethod
    def load_nifti_seg(nifti_path: str | Path) -> torch.Tensor:
        """
        Loads a segmentation from a NIFTI file.

        Args:
            nifti_path (str | Path): The path to the NIFTI segmentation file.

        Returns:
            torch.Tensor: A tensor representation of the loaded segmentation.
        """

        logging.info(f"Loading segmnentation from NIFTI file: {nifti_path}")
        return _nifti_totensor(nifti_path).long().contiguous()

    @staticmethod
    def to_numpy(tensor):
        """
        Converts a PyTorch tensor to a NumPy array.

        This method handles tensors that require gradients by detaching them from the computation graph
        before conversion. It also ensures that the tensor is moved to the CPU if it is on a GPU.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            np.ndarray: The converted NumPy array.
        """

        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )
