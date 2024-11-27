# |--------------------------|
# | PROJECT GLOBAL CONSTANTS |
# |--------------------------|

# First, import the necessary libraries:
from pathlib import Path
from matplotlib.colors import ListedColormap

_BASE_PATH_ = Path(__file__).resolve().parents[3] / "external_files"

# Model weights and checkpoints paths:
CHECKPOINT = _BASE_PATH_ / "best_deeplabv3plus_mobilenet_v2_100_32_0.01.ckpt"
ONNX_MODEL = _BASE_PATH_ / "seg_model.onnx"

# Test and example cases paths:
TEST_NIFTI_IMG = _BASE_PATH_ / "imaging20.nii.gz"
TEST_NIFTI_SEG = _BASE_PATH_ / "segmentation20.nii.gz"

# # Useful utilities:
COLOR_MAP = ListedColormap(["black", "green", "red", "magenta"])


if __name__ == "__main__":
    print(f"Base path: {_BASE_PATH_}")
    print(f"Model checkpoint file: {CHECKPOINT}. Exists -> {CHECKPOINT.exists()}")
    print(f"ONNX model file: {ONNX_MODEL}. Exists -> {ONNX_MODEL.exists()}")
    print(f"Test image path: {TEST_NIFTI_IMG}. Exists -> {TEST_NIFTI_IMG.exists()}")
    print(
        f"Test segmentation path: {TEST_NIFTI_SEG}. Exists -> {TEST_NIFTI_SEG.exists()}"
    )
