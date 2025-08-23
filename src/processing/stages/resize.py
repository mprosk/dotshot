"""Resize processing stage."""

import logging
from typing import Tuple
from PIL import Image, ImageOps

from config.settings import SETTINGS


class ResizeStage:
    """Handles image resizing for printer compatibility."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Resize image to target dimensions.
        
        Args:
            image: Input PIL Image
            **kwargs: Optional override parameters
            
        Returns:
            Resized PIL Image
        """
        target_width = kwargs.get("target_width", self.settings.PRINTER_WIDTH_PIXELS)
        target_height = kwargs.get("target_height", self.settings.PRINTER_HEIGHT_PIXELS)
        maintain_aspect = kwargs.get("maintain_aspect_ratio", self.settings.MAINTAIN_ASPECT_RATIO)
        algorithm = kwargs.get("resize_algorithm", self.settings.RESIZE_ALGORITHM)
        
        target_size = (target_width, target_height)
        original_size = image.size
        
        self.logger.debug(f"Resizing from {original_size} to target {target_size}")
        
        # Get PIL resampling filter
        resample_filter = self._get_resample_filter(algorithm)
        
        if maintain_aspect:
            # Use PIL's thumbnail method which maintains aspect ratio
            resized_image = image.copy()
            resized_image.thumbnail(target_size, resample_filter)
            
            # Create final image with target dimensions and center the resized image
            final_image = Image.new('RGB', target_size, (255, 255, 255))  # White background
            
            # Calculate position to center the image
            x_offset = (target_width - resized_image.width) // 2
            y_offset = (target_height - resized_image.height) // 2
            
            final_image.paste(resized_image, (x_offset, y_offset))
            result_image = final_image
            
        else:
            # Resize directly to target size (may distort aspect ratio)
            result_image = image.resize(target_size, resample_filter)
        
        self.logger.debug(f"Resize completed: {original_size} -> {result_image.size}")
        return result_image
    
    def _get_resample_filter(self, algorithm: str) -> Image.Resampling:
        """Get PIL resampling filter from algorithm name.
        
        Args:
            algorithm: Algorithm name string
            
        Returns:
            PIL resampling filter
        """
        algorithm_map = {
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "LANCZOS": Image.Resampling.LANCZOS,
            "HAMMING": Image.Resampling.HAMMING,
            "BOX": Image.Resampling.BOX
        }
        
        return algorithm_map.get(algorithm.upper(), Image.Resampling.LANCZOS)
    
    def get_stage_name(self) -> str:
        """Get the name of this processing stage."""
        return "resize"
    
    def configure(self, **kwargs) -> None:
        """Configure this stage with new parameters."""
        if "target_width" in kwargs:
            self.settings.PRINTER_WIDTH_PIXELS = kwargs["target_width"]
        if "target_height" in kwargs:
            self.settings.PRINTER_HEIGHT_PIXELS = kwargs["target_height"]
        if "maintain_aspect_ratio" in kwargs:
            self.settings.MAINTAIN_ASPECT_RATIO = kwargs["maintain_aspect_ratio"]
        if "resize_algorithm" in kwargs:
            self.settings.RESIZE_ALGORITHM = kwargs["resize_algorithm"]
