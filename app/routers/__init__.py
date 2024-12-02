from . import auth_router
from . import segmentation_router
from . import user_settings_router

from .segmentation_router import SegmentationRouter

__all__ = ["auth_router", "segmentation_router", "user_settings_router", "SegmentationRouter"]