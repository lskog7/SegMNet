import streamlit as st
import numpy as np
from PIL import Image

from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch
import nibabel as nib
from typing import Optional, Tuple, Any, List
import cv2
import logging
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

# logging.basicConfig(level=logging.INFO, format="MODULE->[segmnet.py]: %(message)s")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", in_channels=1, classes=3)
# model.load_state_dict({
#     ".".join(k.split(".")[1:]): v for k, v in torch.load(
#         "demo/unetpp_enb0_3cls.pth",
#         weights_only=True
#     ).items()
# })
# model.eval()
# model.to(device)
#
#
# class _ImgDim(Enum):
#     TWO_DIM: int = 2
#     THREE_DIM: int = 3
#
#
# def _show(x: np.ndarray) -> None:
#     plt.imshow(x, cmap='jet')
#     plt.axis('off')
#     plt.show()
#
#
# @st.cache_data
# def _load_data(uploaded_file: UploadedFile) -> Optional[np.ndarray]:
#     data = None
#     if uploaded_file is not None:
#         file_extension = uploaded_file.name.split('.')[-1].lower()
#
#         if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
#             result = np.array(Image.open(uploaded_file).convert('L')).astype(np.float32)
#             return result
#
#         elif file_extension in ['npy', 'npz']:
#             result = np.load(uploaded_file).astype(np.float32)
#             return result
#
#         elif file_extension in ['pt', 'pth']:
#             result = torch.load(uploaded_file, weights_only=True).to(dtype=torch.float32).numpy()
#             return result
#
#         elif file_extension in ['nii', 'nii.gz']:
#             result = nib.load(uploaded_file).get_fdata().astype(np.float32)
#             return result
#
#         else:
#             st.error("Unsupported file type. Please upload an image or a valid data file.")
#             return None
#     else:
#         return None
#
#
# @st.cache_data
# def _preprocess(input: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
#     input = cv2.resize(input, target_size, interpolation=cv2.INTER_LINEAR)
#     output = cv2.normalize(input, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     return output
#
#
# @st.cache_data
# def _postprocess(input: np.ndarray) -> np.ndarray:
#     output = cv2.normalize(input, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     return output
#
#
# def _process_data(model: nn.Module, input: np.ndarray, dim: ImgDim = ImgDim.TWO_DIM) -> Optional[np.ndarray]:
#     output = None
#     if dim == ImgDim.THREE_DIM:
#         ...
#     elif dim == ImgDim.TWO_DIM:
#         input = preprocess(input)
#         output = model.forward(input)
#         return output
#     return output


# TODO: составить список переменных, функций и классов
logging.basicConfig(
    level=logging.INFO,
    format="MODULE->[app.py]: %(message)s"
)

__IMAGE_SIZE__: Tuple[int, int] = (512, 512)
__DEVICE__: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__NUM_CLASSES__: int = 3
__IN_CHANNELS__: int = 1
__ENCODER_NAME__: str = "efficientnet-b0"

model: nn.Module = smp.UnetPlusPlus(
    encoder_name=__ENCODER_NAME__,
    in_channels=__IN_CHANNELS__,
    classes=__NUM_CLASSES__
)
errors: List[str] = []


def load_image(
        uploaded_file: UploadedFile
) -> Optional[np.ndarray]:
    result = None
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
            try:
                result = np.array(Image.open(uploaded_file).convert('L')).astype(np.float32)
                return result
            except Exception as e:
                st.error(e)
                return None
        elif file_extension in ['npy', 'npz']:
            try:
                result = np.load(uploaded_file).astype(np.float32)
                return result
            except Exception as e:
                st.error(e)
                return None
        elif file_extension in ['pt', 'pth']:
            try:
                result = torch.load(uploaded_file, weights_only=True).to(dtype=torch.float32).numpy()
                return result
            except Exception as e:
                st.error(e)
                return None
        elif file_extension in ['nii', 'nii.gz']:
            try:
                result = nib.load(uploaded_file).get_fdata().astype(np.float32)
                return result
            except Exception as e:
                st.error(e)
                return None
        else:
            st.error("Unsupported file type. Please upload an image or a valid data file.")
            return None
    else:
        st.error("Uploaded file is None.")
        return None


@st.cache_data
def preprocess(
        image: np.ndarray,
        ndim: int = 2
) -> Optional[torch.Tensor]:
    result = None
    if image is not None:
        if ndim == 2:  # (H, W)
            norm_slice = cv2.normalize(
                image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )
            img_tensor = torch.tensor(
                img_norm,
                dtype=torch.float32
            )
            img_tensor_resized = F.resize(
                img_tensor,
                size=[__IMAGE_SIZE__[0], __IMAGE_SIZE__[1]],
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
            img_tensor_resized_norm = F.normalize(
                img_tensor_resized,
                mean=[0.5],
                std=[0.5],
            )
            result = img_tensor_resized_norm.unsqueeze(0).unsqueeze(0)
            return result
        elif ndim == 3:  # (N, H, W)
            processed_slices = []
            st.progress(0, text="Preprocessing 3D image.")
            for idx in range(image.shape[0]):
                norm_slice = cv2.normalize(
                image[idx],
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )


        else:
            st.error("Unsupported image dimension.")
            return None
    else:
        st.error("Image is None.")
        return None


def predict(
        image: np.ndarray,
        model: nn.Module,
        ndim: int = 2
) -> Optional[np.ndarray]:
    result = None
    if image is not None and model is not None:
        if ndim == 2:  # (H, W)
            ...
        elif ndim == 3:  # (N, H, W)
            ...
        else:
            st.error("Unsupported image dimension.")
            return None
    else:
        if image is None and model is not None:
            st.error("Image is None.")
        elif model is None and image is not None:
            st.error("Model is None.")
        else:
            st.error("Both image and model are None.")
        return None


@st.cache_data
def postprocess(
        prediction: np.ndarray,
        ndim: int = 2
) -> Optional[np.ndarray]:
    result = None
    if image is not None:
        if ndim == 2:  # (H, W)
            ...
        elif ndim == 3:  # (N, H, W)
            ...
        else:
            st.error("Unsupported image dimension.")
            return None
    else:
        st.error("Image is None.")
        return None


@st.cache_data
def combine_result(
        image: np.ndarray,
        prediction: np.ndarray,
        ndim: int = 2
) -> Optional[np.ndarray]:
    result = None
    if image is not None and prediction is not None:
        if ndim == 2:  # (H, W)
            ...
        elif ndim == 3:  # (N, H, W)
            ...
        else:
            st.error(f"Unsupported image dimension. Image must have 2 or 3 dimensions, got {ndim}.")
            return None
    else:
        if image is None and prediction is not None:
            st.error("Image is None.")
        elif prediction is None and image is not None:
            st.error("Prediction is None.")
        else:
            st.error("Both image and prediction are None.")
        return None


# TODO: просто прописать логику приложения, используя функции, которые написаны выше
# TODO: написать про то, что можно загружать, а что нельзя (вообще о том, что должны быть серые изображения, если загружаются в нампае или торче)
# STREAMLIT LOGIC
st.header("SegMNet")
st.subheader("Fast and precise kidney tumor segmentator")
st.write(
    "SegMNet is a segmentation segmnet built with PyTorch and Streamlit that processes CT scans and finds kidney and kidney tumors if present.")
st.write("1. Upload an image")
st.write("2. Click ""Segmentation"" button")
st.write("3. Lookup the result")
st.write("4. Download the result in the desired format")

uploaded_file = st.file_uploader(
    label="Choose an image file:",
    accept_multiple_files=False,
    type=[".jpg", ".jpeg", ".png", ".webp", ".npy", ".npz", ".pt", ".pth", ".nii", ".nii.gz"],
)

if uploaded_file is not None:
    image = load_image(uploaded_file)
    if image is not None:
        ndim = np.ndim(image)
        if ndim == 3:
            st.write(f"Loaded {ndim}-dimensional image")
            st.image(image, use_column_width=True, caption="Input image")
            st.write("Preprocessing image...")
            image = preprocess(image, ndim=ndim)
            st.write("Predicting...")
            prediction = predict(image, model, ndim=ndim)
            st.write("Postprocessing prediction...")
            prediction = postprocess(prediction, ndim=ndim)
            result = combine_result(image, prediction, ndim=ndim)
            st.write(result, unsafe_allow_html=True, caption="Result sample")

        else:
            st.error("Image must be 2- or 3-dimensional.")

    else:
        st.error("Please upload an image or a valid data file.")

else:
    st.error("No image selected.")
