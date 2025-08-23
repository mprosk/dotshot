"""Image processing module for DotShot."""

from .pipeline import ProcessingPipeline
from .dithering import DitheringProcessor

__all__ = ["ProcessingPipeline", "DitheringProcessor"]
