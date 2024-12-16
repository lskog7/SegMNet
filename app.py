# |--------------------------------------------|
# | MAIN FILE OF A SEGMENT APP BASED ON GRADIO |
# |--------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

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

logging.basicConfig(level=logging.INFO, format="MODULE->[app.py]: %(message)s")

# Load the segmentation model
model = smp.Unet(encoder_name="efficientnet-b0", in_channels=1, classes=4)  # Set in_channels to 1 for grayscale
model.eval()

def predict_one_array(image: np.ndarray) -> np.ndarray:
    image = cv2.normalize(
        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = F.resize(image, [512, 512], interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    image = F.normalize(image, mean=[0.5], std=[0.5])  # Adjusted for grayscale
    with torch.inference_mode():
        output = model(image)
    output = (np.argmax(output, axis=1))
    output = output.squeeze().detach().cpu().numpy()
    return output

def load_image(image_path: str | Path) -> np.ndarray:
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
            # Convert to grayscale
            img = img.convert("L")  # Convert to grayscale
            return np.array(img)
    else:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")

def predict(image_path: str | Path) -> np.ndarray:
    image = load_image(image_path)
    assert np.ndim(image) >= 2
    if np.ndim(image) == 2:
        image = np.expand_dims(image, axis=0)
    predictions = []
    for idx in range(image.shape[0]):
        img = image[idx: idx + 1]
        pred = predict_one_array(img)
        predictions.append(pred)
    predictions = np.array(predictions)
    assert len(predictions) > 0
    if len(predictions) == 1:
        return predictions[0]
    else:
        return predictions

def save_segmented_image(segmentation: np.ndarray) -> str:
    # Convert the segmentation output to an image
    segmented_image = Image.fromarray((segmentation * 255).astype(np.uint8))
    output_path = "tmp/segmented_image.png"
    segmented_image.save(output_path)
    return output_path


def save_full_prediction(prediction: np.ndarray, output_folder: str) -> str:
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    if prediction.ndim == 2:
        # Save as a single PNG if it's a 2D array
        output_path = os.path.join(output_folder, "segmented_image.png")
        Image.fromarray((prediction * 255).astype(np.uint8)).save(output_path)
        return output_path
    elif prediction.ndim == 3:
        # Save each slice as a separate PNG
        for idx in range(prediction.shape[0]):
            output_path = os.path.join(output_folder, f"segmented_image_{idx}.png")
            Image.fromarray((prediction[idx] * 255).astype(np.uint8)).save(output_path)
        return f"Saved {prediction.shape[0]} images to {output_folder}"
    else:
        raise ValueError("Prediction must be a 2D or 3D numpy array.")


def get_max_sum_slice(prediction: np.ndarray) -> np.ndarray:
    if prediction.ndim == 2:
        return prediction
    max_sum_index = np.argmax(np.sum(prediction==2))  # Sum over height and width
    return prediction[max_sum_index]


# Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("## SegMNet: Fast yet precise kidney tumor segmentator")
    gr.Markdown("Upload an image to see its segmentation and download the segmented image.")

    with gr.Row():
        input_image = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="numpy", label="Segmentation")
    with gr.Row():
        save_button = gr.Button("Save Segmentation")

    output_file = gr.File(label="Download Segmented Image")


    def process_image(image_path):
        prediction = predict(image_path)  # Get the prediction
        max_slice = get_max_sum_slice(prediction)  # Get the slice with the maximum sum
        save_full_prediction(prediction, "tmp/segmented_images")  # Save the full prediction
        return max_slice, save_full_prediction(prediction, "tmp/segmented_images")


    input_image.change(process_image, inputs=input_image, outputs=[output_image, output_file])

demo.launch()
