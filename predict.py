# |---------------------------------|
# | MAKE A PREDICTION ON NIFTI FILE |
# |---------------------------------|

# First, import the necessary libraries:
#   1. General libraries:
import torch
import logging

#   2. Local libraries:
from app.models.segmentation_model import Inference
from app.models.segmentation_model import dice_score
from app.constants import TEST_NIFTI_IMG, TEST_NIFTI_SEG, ONNX_MODEL, COLOR_MAP

#------------------------------------------------------------------------------
# TEST CASE
size = (4, 1, 512, 512)
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
inf = Inference(ONNX_MODEL) 

# Load the target NIFTI image (x) and corresponding segmentation (y_true).
x = inf.load_nifti_img(TEST_NIFTI_IMG) # torch.Tensor [B, 1, 512, 512]
y_true = inf.load_nifti_seg(TEST_NIFTI_SEG) # torch.Tensor [B, 1, 512, 512]
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

if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    
    target_slices = [idx for idx in range(y_true.shape[0]) if y_true[idx, :, :][y_true[idx, :, :] == 2].long().sum() > 0]
    logging.info(f"Target slices are: {target_slices}")
    
    n = target_slices[len(target_slices) // 2]
    
    plt.title(f"GT segmentation. Slice: {n}")
    plt.imshow(x.detach().cpu().squeeze()[n], cmap='gray')
    plt.imshow(y_true.detach().cpu().squeeze()[n], cmap=COLOR_MAP, alpha=0.5)
    plt.axis("off")
    plt.show()

    plt.title(f"Predicted segmentation. Slice: {n}")
    plt.imshow(x.detach().cpu().squeeze()[n], cmap='gray')
    plt.imshow(logits.argmax(1).detach().cpu().squeeze()[n], cmap=COLOR_MAP, alpha=0.5)
    plt.axis("off")
    plt.show()