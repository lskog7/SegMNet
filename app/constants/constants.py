# |--------------------------|
# | PROJECT GLOBAL CONSTANTS |
# |--------------------------|

# First, import the necessary libraries:
from pathlib import Path
from matplotlib.colors import ListedColormap

_BASE_PATH_ = Path(__file__).resolve().parents[3]

# Model weights and checkpoints paths:
CHECKPOINT = _BASE_PATH_ / "external_files" / "best_deeplabv3plus_mobilenet_v2_100_32_0.01.ckpt"
ONNX_MODEL = _BASE_PATH_ / "external_files" / "dlv3p_enb5_76e.onnx"
TMP_PATH = _BASE_PATH_ / ".app_tmp"

# Result save path:
RESULT_PATH = TMP_PATH / "result"

# Test and example cases paths:
TEST_NIFTI_IMG = _BASE_PATH_ / "external_files" / "imaging.nii.gz"
TEST_NIFTI_SEG = _BASE_PATH_ / "external_files" / "segmentation.nii.gz"

# Useful utilities:
COLOR_MAP = ListedColormap(["black", "green", "red", "magenta"])


if __name__ == "__main__":
    print(_BASE_PATH_)
