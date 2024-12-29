# |--------------------------|
# | PROJECT GLOBAL CONSTANTS |
# |--------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# Libraries:
from pathlib import Path
from matplotlib.colors import ListedColormap
import logging

# Local modules:
...

# Module-specific logging template:
# logging.basicConfig(level=logging.INFO, format="MODULE->[constants.py]: %(message)s")

# Basic directory of the project.
# In fact, it is just ./SegMNet folder related to your file hierarchy:
_BASE_DIR_ = Path(__file__).resolve().parents[2]

# Temporary path for all files if the app:
TMP_DIR = _BASE_DIR_ / "tmp"

# Path to the .onnx model file:
MODELS_DIR = _BASE_DIR_ / "models"

# Path to the folder with inputs and outputs:
INPUT_DIR = TMP_DIR / "input"
OUTPUT_DIR = TMP_DIR / "output"

# Useful utilities:
# Color map for matplotlib:
COLOR_MAP = ListedColormap(["black", "green", "red", "magenta"])

# Make a list of all necessary paths:
_DIRS_ = [TMP_DIR, MODELS_DIR, INPUT_DIR, OUTPUT_DIR]

# All necessary files must be created during the first start of the app:
if len(_DIRS_):
    all_ok = True
    missing = []
    for d in _DIRS_:
        if not d.exists():
            all_ok = False
            missing.append(d)
    if not all_ok:
        logging.info(">>> SegMNet Builder: Found missing temporary directories. Creating them.")
        for d in missing:
            logging.info(f">>> SegMNet Builder: Creating {d}.")
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.info(f">>> SegMNet Builder: Creating {d} failed. App building stopped.")
                raise
    else:
        logging.info(">>> SegMNet Builder: All temporary directories found. Continuing building the app.")