# |--------------------------|
# | PROJECT GLOBAL CONSTANTS |
# |--------------------------|

# First, import the necessary libraries:
from pathlib import Path
from matplotlib.colors import ListedColormap

# Model weights and checkpoints paths:
CHECKPOINT = Path("external_files/best_deeplabv3plus_mobilenet_v2_100_32_0.01.ckpt")
ONNX_MODEL = Path("external_files/seg_model.onnx")

# Test and example cases paths:
TEST_NIFTI_IMG = Path("external_files/imaging20.nii.gz")
TEST_NIFTI_SEG = Path("external_files/segmentation20.nii.gz")

# Useful utilities:
COLOR_MAP = ListedColormap(["black", "green", "red", "magenta"])
