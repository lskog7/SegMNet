from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import logging
import os

from app.constants import TMP_PATH
from app.services.segmentation_service import process_nifti

SegmentationRouter = APIRouter(prefix="/segmentation", tags=["Segmentation"])

@SegmentationRouter.post("/upload/")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_path = os.path.join(TMP_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_path = os.path.join(TMP_PATH, f"result_{file.filename}")
    background_tasks.add_task(process_nifti, file_path, result_path)

@SegmentationRouter.post("/")
async def segment_nifti(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        model=Depends(lambda: app.state.model),  # Access the model from app state
):
    # Check file format:
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported.")

    # Define paths:
    temp_file_path = TMP_PATH / file.filename
    result_file_path = TMP_PATH / f"result_{file.filename}"

    try:
        # Save uploaded file:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File {file.filename} saved to {temp_file_path}.")

        # Start background segmentation task:
        background_tasks.add_task(process_nifti, temp_file_path, result_file_path, model)
        return {"message": "Segmentation started.", "result_path": str(result_file_path)}
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during file processing.")


@SegmentationRouter.get("/download/{filename}")
def download_result(filename: str):
    result_path = TMP_PATH / filename
    if result_path.exists():
        return FileResponse(result_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found.")
