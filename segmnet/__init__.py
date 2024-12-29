# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|


from . import constants

from .constants import ONNX_MODEL, OUTPUT_DIR, COLOR_MAP, INPUT_DIR, TMP_DIR, MODELS_DIR

__all__ = [
    "constants",
    "ONNX_MODEL",
    "OUTPUT_DIR",
    "MODELS_DIR",
    "COLOR_MAP",
    "INPUT_DIR",
    "TMP_DIR"
]
