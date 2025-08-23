"""Crop processing stage."""

import logging
from typing import Tuple, Optional
from PIL import Image

from config.settings import SETTINGS


class CropStage:
    """Handles image cropping for optimal composition."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Crop image based on specified parameters.
        
        Args:
            image: Input PIL Image
            **kwargs: Optional override parameters
            
        Returns:
            Cropped PIL Image
        """
        crop_to_center = kwargs.get("crop_to_center", self.settings.CROP_TO_CENTER)
        crop_margin = kwargs.get("crop_margin", self.settings.CROP_MARGIN)
        crop_box = kwargs.get("crop_box", None)  # (left, top, right, bottom)
        
        original_size = image.size
        
        if crop_box:
            # Use explicit crop box
            cropped_image = image.crop(crop_box)
            self.logger.debug(f"Cropped using explicit box: {crop_box}")
        
        elif crop_to_center:
            # Crop to center with margin
            cropped_image = self._crop_to_center(image, crop_margin)
            
        else:
            # No cropping, return original
            cropped_image = image
            self.logger.debug("No cropping applied")
        
        self.logger.debug(f"Crop completed: {original_size} -> {cropped_image.size}")
        return cropped_image
    
    def _crop_to_center(self, image: Image.Image, margin: float) -> Image.Image:
        """Crop image to center with specified margin.
        
        Args:
            image: Input PIL Image
            margin: Margin as fraction of image dimension (0.0 to 0.5)
            
        Returns:
            Center-cropped PIL Image
        """
        width, height = image.size
        
        # Calculate crop margins in pixels
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Ensure margins don't exceed image dimensions
        margin_x = min(margin_x, width // 4)
        margin_y = min(margin_y, height // 4)
        
        # Calculate crop box (left, top, right, bottom)
        left = margin_x
        top = margin_y
        right = width - margin_x
        bottom = height - margin_y
        
        crop_box = (left, top, right, bottom)
        
        self.logger.debug(f"Center crop box: {crop_box}, margin: {margin}")
        return image.crop(crop_box)
    
    def _crop_to_aspect_ratio(self, image: Image.Image, target_ratio: float) -> Image.Image:
        """Crop image to match target aspect ratio.
        
        Args:
            image: Input PIL Image
            target_ratio: Target width/height ratio
            
        Returns:
            Aspect-ratio cropped PIL Image
        """
        width, height = image.size
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to target ratio
        
        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            margin = (width - new_width) // 2
            crop_box = (margin, 0, width - margin, height)
        else:
            # Image is too tall, crop height
            new_height = int(width / target_ratio)
            margin = (height - new_height) // 2
            crop_box = (0, margin, width, height - margin)
        
        self.logger.debug(f"Aspect ratio crop: {current_ratio:.2f} -> {target_ratio:.2f}, box: {crop_box}")
        return image.crop(crop_box)
    
    def get_stage_name(self) -> str:
        """Get the name of this processing stage."""
        return "crop"
    
    def configure(self, **kwargs) -> None:
        """Configure this stage with new parameters."""
        if "crop_to_center" in kwargs:
            self.settings.CROP_TO_CENTER = kwargs["crop_to_center"]
        if "crop_margin" in kwargs:
            self.settings.CROP_MARGIN = kwargs["crop_margin"]
    
    def suggest_crop_box(self, image: Image.Image, target_aspect_ratio: Optional[float] = None) -> Tuple[int, int, int, int]:
        """Suggest optimal crop box for an image.
        
        Args:
            image: Input PIL Image
            target_aspect_ratio: Optional target aspect ratio (width/height)
            
        Returns:
            Suggested crop box as (left, top, right, bottom)
        """
        if target_aspect_ratio:
            cropped = self._crop_to_aspect_ratio(image, target_aspect_ratio)
            # Calculate the crop box that would produce this result
            width, height = image.size
            new_width, new_height = cropped.size
            
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            return (left, top, right, bottom)
        else:
            # Use center crop with default margin
            return self._get_center_crop_box(image, self.settings.CROP_MARGIN)
    
    def _get_center_crop_box(self, image: Image.Image, margin: float) -> Tuple[int, int, int, int]:
        """Get center crop box coordinates."""
        width, height = image.size
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        return (margin_x, margin_y, width - margin_x, height - margin_y)
