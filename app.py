# |--------------------------------------------|
# | MAIN FILE OF A SEGMNET APP BASED ON GRADIO |
# |--------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# Libraries:
import gradio as gr
import numpy as np
import nibabel as nib
import os
import uuid  # For generating unique file names
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm  # For progress bar
import shutil

# Local modules:
from app import DataLoader, Inference

# Module-specific logging template:
logging.basicConfig(level=logging.INFO, format="MODULE->[app.py]: %(message)s")

# Initialize data loader and model inference instances:
loader = DataLoader()
model = Inference()

# Cache for storing processed slices:
slice_cache = {}


def get_slice(image, seg, axis, index):
    if image is None or seg is None:
        return None

    slice_key = f"{axis}-{index}"
    if slice_key in slice_cache:
        return slice_cache[slice_key]

    if axis == "x":
        img_slice = image[index, :, :]
        seg_slice = seg[index, :, :]
    elif axis == "y":
        img_slice = image[:, index, :]
        seg_slice = seg[:, index, :]
    else:  # axis == "z"
        img_slice = image[:, :, index]
        seg_slice = seg[:, :, index]

    # Overlay segmentation on image:
    fig, ax = plt.subplots()
    ax.imshow(img_slice.T, cmap="gray", origin="lower")
    ax.imshow(seg_slice.T, cmap="Reds", alpha=0.5, origin="lower")
    ax.axis("off")

    # Save to buffer:
    buf = f"slice_{slice_key}.png"
    plt.savefig(buf, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    slice_cache[slice_key] = buf

    return buf


def process_and_predict(nifti_file):
    segmented_path = os.path.join(loader.tmp_dir, f"segmentation.nii.gz")

    image, affine, base_size = loader.load(nifti_file)
    image = loader.window(image)
    image = loader.transform(image)

    predictions = model.predict(image.unsqueeze(1)).astype(np.float32)

    # Save the segmented image as a new NIfTI file
    segmented_image = nib.Nifti1Image(predictions, affine)
    nib.save(segmented_image, segmented_path)

    return segmented_path


def visualize_slice(segmented_file, axis, index):
    """
    Visualize a slice of the segmented image.

    Args:
        segmented_file (str): Path to the segmented NIfTI file.
        axis (str): Axis to slice ('x', 'y', 'z').
        index (int): Index of the slice.

    Returns:
        str: Path to the PNG image of the slice overlay.
    """
    # Load the segmented image:
    segmented = nib.load(segmented_file).get_fdata()

    # Load the original image:
    original_file = segmented_file.replace("segmented", "original")
    original = nib.load(original_file).get_fdata()

    return get_slice(original, segmented, axis, index)


def download_segmented_file(segmented_file):
    """
    Provide the segmented NIfTI file for download.

    Args:
        segmented_file (str): Path to the segmented file.

    Returns:
        str: Path to the file for download.
    """
    return segmented_file


# Gradio app interface:
with gr.Blocks() as demo:
    gr.Markdown('# SegMNet: Fast and Precise Kidney Tumor Segmentation')

    with gr.Row():
        upload = gr.File(label="Upload .nii.gz File")
        # segment_btn = gr.Button("Run Segmentation")
        # axis = gr.Radio(["x", "y", "z"], label="Select Axis", value="z")
        # index = gr.Slider(minimum=0, maximum=100, step=1, label="Select Slice Index")
    with gr.Row():
        segment_btn = gr.Button("Run Segmentation")
    with gr.Row():
        # visualize_btn = gr.Button("Visualize Slice")
        download_btn = gr.Button("Download Segmented File")

    # output_image = gr.Image(label="Segmented Slice")
    output_file = gr.File(label="Download .nii.gz")
    # status_text = gr.Textbox(label="Status", interactive=False)

    # Logic:
    segmented_file = gr.State()

    upload.change(fn=None, inputs=[], outputs=[])  # Remove automatic segmentation on upload
    segment_btn.click(fn=process_and_predict, inputs=[upload], outputs=[segmented_file])
    # visualize_btn.click(fn=visualize_slice, inputs=[segmented_file, axis, index], outputs=[output_image])
    download_btn.click(fn=download_segmented_file, inputs=[segmented_file], outputs=[output_file])

demo.launch()

