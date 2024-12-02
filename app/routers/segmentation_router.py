from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
import shutil
import logging

from app.constants import TMP_PATH
from app.services.segmentation_service import process_nifti

SegmentationRouter = APIRouter(prefix="/segmentation", tags=["Segmentation"])


@SegmentationRouter.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Загружает файл на сервер без запуска процесса сегментации.
    """
    # Проверка формата файла:
    if not file.filename.endswith(".nii.gz"):
        logging.error(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="Only .nii.gz files are supported.")

    # Определяем путь для сохранения файла:
    file_path = TMP_PATH / file.filename

    try:
        # Сохраняем файл:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File {file.filename} uploaded and saved to {file_path}.")
        return {"message": "File uploaded successfully.", "file_path": str(file_path)}
    except Exception as e:
        logging.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail="File upload failed.")


@SegmentationRouter.post("/process/")
async def process_file(filename: str, background_tasks: BackgroundTasks):
    """
    Запускает процесс сегментации для загруженного файла.
    """
    # Пути для входного и результирующего файлов:
    file_path = TMP_PATH / filename
    result_path = TMP_PATH / f"result_{filename}"

    # Проверяем, существует ли файл:
    if not file_path.exists():
        logging.error(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        # Добавляем задачу сегментации в фоновый процесс:
        background_tasks.add_task(process_nifti, file_path, result_path)
        logging.info(f"Segmentation task started for {filename}.")
        return {"message": "Segmentation started.", "result_path": str(result_path)}
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        raise HTTPException(status_code=500, detail="Segmentation process failed.")


@SegmentationRouter.get("/download/{filename}")
def download_result(filename: str):
    """
    Позволяет скачать результирующий файл сегментации.
    """
    result_path = TMP_PATH / filename

    # Проверяем существование файла:
    if not result_path.exists():
        logging.error(f"Requested file not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        logging.info(f"File {filename} found and ready for download.")
        return FileResponse(result_path, filename=filename)
    except Exception as e:
        logging.error(f"Error during file download: {e}")
        raise HTTPException(status_code=500, detail="Error during file download.")
