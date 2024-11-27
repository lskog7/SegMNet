# from fastapi import APIRouter, UploadFile, File
# from app.services.segmentation_service import process_segmentation

# router = APIRouter()

# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     result = await process_segmentation(file)
#     return {"result": result}
