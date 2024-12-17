# |--------------------------------------------|
# | MAIN FILE OF A SEGMENT APP BASED ON GRADIO |
# |--------------------------------------------|

import gradio as gr
import numpy as np
import torch
import logging
import cv2
import segmentation_models_pytorch as smp
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from pathlib import Path
from PIL import Image
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="MODULE->[app.py]: %(message)s")

# Load the segmentation model
model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", in_channels=1, classes=4)
model.load_state_dict({
    ".".join(k.split(".")[1:]): v for k, v in torch.load(
        "/Users/fffgson/Desktop/Diploma/Code/segment_v0.1/SegMNet/demo/UNetPP_2024-12-16.pth",
        weights_only=True
    ).items()
})
model.eval()


# Functions
def predict_one_array(image: np.ndarray) -> np.ndarray:
    """Process a single image and return the segmentation output."""
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = F.resize(image, [512, 512], interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    image = F.normalize(image, mean=[0.5], std=[0.5])  # Adjusted for grayscale
    with torch.inference_mode():
        output = model(image)
    output = np.argmax(output, axis=1)
    return output.squeeze().detach().cpu().numpy()


def load_image(image_path: str | Path) -> np.ndarray:
    """Load image from various formats."""
    image_path = Path(image_path)
    if image_path.suffix in ['.pt', '.pth']:
        tensor = torch.load(image_path)
        return tensor.numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
    elif image_path.suffix == '.npy':
        return np.load(image_path)
    elif image_path.suffix == '.npz':
        data = np.load(image_path)
        return next(iter(data.values()))
    elif image_path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
        with Image.open(image_path) as img:
            return np.array(img.convert("L"))  # Convert to grayscale
    else:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")


def predict(image_path: str | Path) -> np.ndarray:
    """Predict segmentation for single or multi-slice images."""
    image = load_image(image_path)
    if np.ndim(image) == 2:  # Single 2D image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = [predict_one_array(image[idx:idx+1]) for idx in range(image.shape[0])]
    return np.array(predictions)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply segmentation mask to the image with specific colors for kidney and tumor."""
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    # Convert the grayscale image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create an empty color mask
    color_mask = np.zeros_like(image_rgb, dtype=np.uint8)

    # Highlight kidneys (class 1) in green
    color_mask[mask == 1] = [0, 255, 0]  # Green in BGR format

    # Highlight tumors (class 2) in red
    color_mask[mask == 2] = [255, 0, 0]  # Red in BGR format

    # Overlay the color mask on the image
    result = cv2.addWeighted(image_rgb, 0.7, color_mask, 0.3, 0)

    return result


def save_output(image: np.ndarray, output_folder: str) -> str:
    """Save the processed output as PNG or NPY."""
    os.makedirs(output_folder, exist_ok=True)
    if image.ndim == 2:
        output_path = os.path.join(output_folder, "output_image.png")
        Image.fromarray(image).save(output_path)
        return output_path
    elif image.ndim == 3:
        output_path = os.path.join(output_folder, "output_image.npy")
        np.save(output_path, image)
        return output_path


def process_image(image_path: str):
    """Main function to process the input image."""
    image = load_image(image_path)
    prediction = predict(image_path)
    if image.ndim == 2:  # Single image
        masked_image = apply_mask(image, prediction[0])
        save_path = save_output(masked_image, "output")
    else:  # Multi-slice image
        max_slice_index = np.argmax(np.sum(prediction == 2, axis=(1, 2)))
        masked_image = apply_mask(image[max_slice_index], prediction[max_slice_index])
        save_path = save_output(prediction, "output")
    return masked_image, save_path


# Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("## SegMNet: Fast yet precise kidney tumor segmentator")
    gr.Markdown("Upload an image, process the segmentation, and download the results.")

    # Input and output sections
    with gr.Row():
        input_image = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="numpy", label="Segmentation Output")

    with gr.Row():
        process_button = gr.Button("Process Segmentation")  # Button to trigger segmentation
        output_file = gr.File(label="Download Segmented Image")


    # Function to process segmentation
    def on_process(image_path):
        """Process the image when the button is clicked."""
        masked_image, save_path = process_image(image_path)
        return masked_image, save_path


    # Connect button click to the processing function
    process_button.click(on_process, inputs=input_image, outputs=[output_image, output_file])

demo.launch()