# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|
import tempfile

# Import libraries:
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Union
import shutil
import torch
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import nibabel as nib

from app.config import settings
from app.models.segmentation_model import Inference
from app.constants import ONNX_MODEL, TMP_PATH

# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")


# Define FastAPI functions:
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Initializing model from ONNX file: {ONNX_MODEL}")
    app.state.model = Inference(ONNX_MODEL)
    logging.info(f"Model loaded: {app.state.model}.")
    yield
    logging.info("Deleting model from memory.")
    del app.state.model


# Define FastAPI itself:
app = FastAPI(lifespan=lifespan)

# Create (or not) TMP_PATH:
TMP_PATH.mkdir(parents=True, exist_ok=True)


# Define endpoints:
#   1. Root endpoint for the project (same main page):
@app.get("/")
async def root():
    return {"message": "Hello World"}

#   2. Segmentation endpoint:
@app.post("/segmentation")
async def segment_nifti(file: UploadFile = File(...)):
    # Check file format:
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported.")

    # Temporary path to save file:
    temp_file_path = TMP_PATH / file.filename

    try:
        # Save uploaded file:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File {file.filename} has been saved to {temp_file_path}.")

        # Load file to memory:
        nifti_img = ...
    except Exception as e:
        ...

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

@app.get("/load")
async def load(): ...


@app.get("/result")
async def result(): ...
