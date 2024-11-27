# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|


# Import librarires:
from fastapi import FastAPI

# from app.routers import auth, segmentation, user_settings
from contextlib import asynccontextmanager
from app.config import settings
import logging
from app.models.segmentation_model import Inference
from app.constants import ONNX_MODEL, TEST_NIFTI_IMG, TEST_NIFTI_SEG


# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

# Include routers:
# app.include_router(auth.router, prefix="/auth", tags=["auth"])
# app.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])
# app.include_router(user_settings.router, prefix="/user", tags=["user settings"])


# Define FastAPI functions:
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Initiaziling model from ONNX file: {ONNX_MODEL}")
    model = Inference(ONNX_MODEL)
    logging.info(f"Model loaded: {model}.")
    yield
    logging.info("Deleting model from memory.")
    del app.state.model


# Define FastAPI itself:
app = FastAPI(lifespan=lifespan)
