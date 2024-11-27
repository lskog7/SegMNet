from pydantic import BaseModel
import torch

class Segmentation(BaseModel):
    segmentation: torch.Tensor