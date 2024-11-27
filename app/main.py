# |-----------------------|
# | MAIN FASTAPI APP FILE |
# |-----------------------|


from fastapi import FastAPI
from app.routers import auth, segmentation, user_settings
from contextlib import asynccontextmanager
from app.config import settings
import logging


# Define Logging Basic Config:
logging.basicConfig(level=logging.INFO, format="CONSOLE: %(message)s")

# Define FastAPI itself:
app = FastAPI()

# Include routers:
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])
app.include_router(user_settings.router, prefix="/user", tags=["user settings"])


# Initialize segmentation model:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model when the application starts
    # model = SegmentationModel(model_path="path_to_your_model.pth")
    # app.state.model = model
    print("Model initialized.")
    yield
    # Clean up resources when the application shuts down
    del app.state.model
    print("Model cleaned up.")
