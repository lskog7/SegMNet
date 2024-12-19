import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple
import time
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch


@st.cache_resource
class SimpleModel:
    def __init__(self, input_size: Tuple[int, int] = (256, 256)):
        self.w = np.random.randn(*input_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w


def show(x: np.ndarray) -> None:
    plt.imshow(x, cmap='jet')
    plt.axis('off')
    plt.show()


def load_data(uploaded_file: UploadedFile) -> np.ndarray:
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            return image_array

        elif file_extension in ['npy', 'npz']:
            data = np.load(uploaded_file).astype(np.float32)
            return data

        elif file_extension in ['pt', 'pth']:
            data = torch.load(uploaded_file, weights_only=True)
            return data.float().numpy()

        else:
            st.error("Unsupported file type. Please upload an image or a valid data file.")
            return None
    else:
        return None


st.header("SegMNet")
st.subheader("A Fast yet Precise Kidney Tumor Segmentation App!")

uploaded_file = st.file_uploader(
    label="Choose an image file:",
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png", ".webp", ".npy", ".npz", ".pt", ".pth"],
)
if uploaded_file is not None:
    image = load_data(uploaded_file)
    if image is not None:
        if np.ndim(image) == 3:
            ...
    else:
        st.error("Please upload an image or a valid data file.")