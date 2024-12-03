from app.models.segmentation_model import ckpt_to_onnx
from pathlib import Path

ckpt_path = Path("/Users/fffgson/Desktop/Diploma/Code/segment_v0.1/external_files/2024_11_27_11_58_02_deeplabv3plus_efficientnet-b5_100_16_0.005.ckpt")

onnx_path = Path("/Users/fffgson/Desktop/Diploma/Code/segment_v0.1/external_files/dlv3p_enb5_76e.onnx")

ckpt_to_onnx(ckpt_path, onnx_path)