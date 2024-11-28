# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|


# Import libraries:
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import settings
import logging
from app.models.segmentation_model import Inference
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