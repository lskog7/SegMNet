# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|


from fastapi import FastAPI
# from app.routers import auth, segmentation, user_settings
from contextlib import asynccontextmanager
from app.config import settings
import logging
from app.models.segmentation_model import Inference
from app.constants import ONNX_MODEL, TEST_NIFTI_IMG, TEST_NIFTI_SEG


# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

# Define FastAPI itself:
app = FastAPI()

# Include routers:
# app.include_router(auth.router, prefix="/auth", tags=["auth"])
# app.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])
# app.include_router(user_settings.router, prefix="/user", tags=["user settings"])


# Initialize segmentation model:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model when the application starts
    inf = Inference(ONNX_MODEL)
    logging.info(f"Model status: {inf.initialized}, ort session: {inf.ort_session}")
    print("Model initialized.")
    yield
    # Clean up resources when the application shuts down
    del app.state.model
    print("Model cleaned up.")
