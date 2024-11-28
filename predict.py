# |---------------------------------|
# | MAKE A PREDICTION ON NIFTI FILE |
# |---------------------------------|

# First, import the necessary libraries:
#   1. General libraries:
import os
import torch
import logging
from pathlib import Path

#   2. Local libraries:
from app.models.segmentation_model import Inference
from app.models.segmentation_model import dice_score

#   3. Paths (optional):
# from app.constants.constants import TEST_NIFTI_IMG, TEST_NIFTI_SEG, ONNX_MODEL

# For GitHub project I do not want to use local paths. 
# Therefore, I will use the following paths:
_IMG_NIFTI_PATH_ = Path("external_files/imaging.nii.gz")
_SEG_NIFTI_PATH_ = Path("external_files/segmentation.nii.gz")
_ONNX_MODEL_ = Path("external_files/seg_model.onnx")

#------------------------------------------------------------------------------
# TEST CASE
size = (1, 1, 512, 512)
x = torch.randn(size, device="cpu", requires_grad=False)
y_true = torch.randint(0, 4, size, device="cpu", requires_grad=False)
#------------------------------------------------------------------------------

# Define logging parameters:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

# Define the inference class.
# It is the main class of the project.
# Responsible for:
#   1. Loading the best ONNX model.
#   2. Loading the test NIFTI file to torch.Tensor.
#   3. Applying transformations to the torch.Tensor.
#   4. Making predictions.
inf = Inference(_ONNX_MODEL_) 

# Load the target NIFTI image (x) and corresponding segmentation (y_true).
x = inf.load_nifti_img(_IMG_NIFTI_PATH_) # torch.Tensor [B, 1, 512, 512]
y_true = inf.load_nifti_seg(_SEG_NIFTI_PATH_) # torch.Tensor [B, 1, 512, 512]
logging.info(f"Unique values in y_true: {torch.unique(y_true)}")

# Predicting segmenation mask (y_pred).
logits = inf.predict_nifti(x, return_logits=True) # torch.Tensor [B, 4, 512, 512]

# Calculate overall dice score.
dice = dice_score(logits, y_true) # Accepts only torch.Tensors
logging.info(f"Dice score: {dice}")

# Calculate separate binary dice scores.
#   1. Kidney dice score.
kidney_dice = dice_score(logits, y_true, case="kidney")
logging.info(f"Kidney dice score: {kidney_dice}")

#   2. Tumor dice score.
tumor_dice = dice_score(logits, y_true, case="tumor")
logging.info(f"Tumor dice score: {tumor_dice}")

#   3. Cyst dice score.
cyst_dice = dice_score(logits, y_true, case="cyst")
logging.info(f"Cyst dice score: {cyst_dice}")
