# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|


# Import libraries:
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import settings
import logging
from app.models.segmentation_model import Inference
from pathlib import Path
import nibabel as nib
from app.constants import ONNX_MODEL, TEST_NIFTI_IMG, TEST_NIFTI_SEG


# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")


# Define FastAPI functions:
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Initializing model from ONNX file: {ONNX_MODEL}")
    model = Inference(ONNX_MODEL)
    app.state.model = model
    logging.info(f"Model loaded: {model}.")
    yield
    logging.info("Deleting model from memory.")
    del app.state.model


# Define FastAPI itself:
app = FastAPI(lifespan=lifespan)


# Define endpoints:
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/login")
async def login(): ...


@app.get("/register")
async def register(): ...


@app.get("/settings")
async def settings(): ...


@app.get("/settings/user")
async def user_settings(): ...


@app.get("/settings/segmentation")
async def segmentation_settings(): ...


@app.post("/segmentation/")
async def segmentation(file_path: str | Path):
    x = app.state.model.load_nift_img(file_path)
    y_pred = app.state.model.predict_nifti(x).detach().cpu().numpy()
    y_pred.save("")

    output_file = "output.nii.gz"
    output_nii = nib.Nifti1Image(output_tensor.numpy(), x.affine)
    nib.save(output_nii, output_file)

    return {"output_file": "file saved"}


@app.get("/load")
async def load(): ...


@app.get("/result")
async def result(): ...
