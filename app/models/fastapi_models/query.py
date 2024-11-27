from pydantic import BaseModel
import torch

class Image(BaseModel):
    image: torch.Tensor