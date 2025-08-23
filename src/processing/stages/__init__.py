"""Processing stages for image pipeline."""

from .resize import ResizeStage
from .crop import CropStage
from .adjust import AdjustmentStage
from .edges import EdgeEnhancementStage

__all__ = ["ResizeStage", "CropStage", "AdjustmentStage", "EdgeEnhancementStage"]
