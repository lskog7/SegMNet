# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|

# Import libraries:
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from contextlib import asynccontextmanager
import logging

# Add locals libs:
from app.models.segmentation_model import Inference
from app.constants import ONNX_MODEL, TMP_PATH
from app.routers.segmentation_router import SegmentationRouter
from app.services.segmentation_service import DataLoader, SegmentationModel, process_nifti

# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

# Create (or not) TMP_PATH:
TMP_PATH.mkdir(parents=True, exist_ok=True)


# Define FastAPI functions:
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Initializing model from ONNX file: {ONNX_MODEL}")
    app.state.model = SegmentationModel
    logging.info(f"Model loaded: {app.state.model}.")
    yield
    logging.info("Deleting model from memory.")
    del app.state.model


# Define FastAPI itself:
app = FastAPI(title="Kidney Tumor Segmentation App (SegMNet)", lifespan=lifespan)

# Add middleware:
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Add routers:
app.include_router(SegmentationRouter)


# Define endpoints:
#   1. Root endpoint for the project (same main page):
@app.get("/")
async def root():
    return {"message": "Welcome to the Kidney Tumor Segmentation App!"}


APP_HOST = "0.0.0.0"
APP_PORT = 7777

if __name__ == "__main__":
    import uvicorn

    # uvicorn app:app --reload --host 0.0.0.0 --port 7777
    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=True)
