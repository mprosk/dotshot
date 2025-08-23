"""Adjustment processing stage for brightness, contrast, and gamma correction."""

import logging
import numpy as np
from typing import Union
from PIL import Image, ImageEnhance

from config.settings import SETTINGS


class AdjustmentStage:
    """Handles brightness, contrast, and gamma adjustments."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Apply brightness, contrast, and gamma adjustments.
        
        Args:
            image: Input PIL Image
            **kwargs: Optional override parameters
            
        Returns:
            Adjusted PIL Image
        """
        brightness = kwargs.get("brightness", self.settings.BRIGHTNESS_ADJUST)
        contrast = kwargs.get("contrast", self.settings.CONTRAST_ADJUST) 
        gamma = kwargs.get("gamma", self.settings.GAMMA_ADJUST)
        
        adjusted_image = image.copy()
        
        # Apply brightness adjustment
        if brightness != 1.0:
            brightness_enhancer = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = brightness_enhancer.enhance(brightness)
            self.logger.debug(f"Applied brightness adjustment: {brightness}")
        
        # Apply contrast adjustment
        if contrast != 1.0:
            contrast_enhancer = ImageEnhance.Contrast(adjusted_image)
            adjusted_image = contrast_enhancer.enhance(contrast)
            self.logger.debug(f"Applied contrast adjustment: {contrast}")
        
        # Apply gamma correction
        if gamma != 1.0:
            adjusted_image = self._apply_gamma_correction(adjusted_image, gamma)
            self.logger.debug(f"Applied gamma correction: {gamma}")
        
        return adjusted_image
    
    def _apply_gamma_correction(self, image: Image.Image, gamma: float) -> Image.Image:
        """Apply gamma correction to an image.
        
        Args:
            image: Input PIL Image
            gamma: Gamma value (1.0 = no change, <1.0 = darker, >1.0 = brighter)
            
        Returns:
            Gamma-corrected PIL Image
        """
        # Convert to numpy array for gamma correction
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to 0-1 range
        img_array /= 255.0
        
        # Apply gamma correction
        img_array = np.power(img_array, gamma)
        
        # Convert back to 0-255 range
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def apply_histogram_equalization(self, image: Image.Image, method: str = "global") -> Image.Image:
        """Apply histogram equalization to improve contrast.
        
        Args:
            image: Input PIL Image
            method: Equalization method ("global" or "adaptive")
            
        Returns:
            Equalized PIL Image
        """
        if method == "global":
            return self._global_histogram_equalization(image)
        elif method == "adaptive":
            return self._adaptive_histogram_equalization(image)
        else:
            raise ValueError(f"Unknown equalization method: {method}")
    
    def _global_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply global histogram equalization."""
        import cv2
        
        # Convert PIL to OpenCV format
        img_cv = np.array(image)
        
        if len(img_cv.shape) == 3:
            # Color image - convert to YUV, equalize Y channel
            img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_cv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            # Grayscale image
            img_cv = cv2.equalizeHist(img_cv)
        
        return Image.fromarray(img_cv)
    
    def _adaptive_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply adaptive histogram equalization (CLAHE)."""
        import cv2
        
        # Convert PIL to OpenCV format
        img_cv = np.array(image)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if len(img_cv.shape) == 3:
            # Color image - convert to LAB, apply to L channel
            img_lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
            img_cv = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            img_cv = clahe.apply(img_cv)
        
        return Image.fromarray(img_cv)
    
    def apply_white_balance(self, image: Image.Image, method: str = "gray_world") -> Image.Image:
        """Apply white balance correction.
        
        Args:
            image: Input PIL Image
            method: White balance method ("gray_world" or "white_patch")
            
        Returns:
            White balanced PIL Image
        """
        img_array = np.array(image).astype(np.float32)
        
        if method == "gray_world":
            # Gray world assumption: average color should be gray
            avg_rgb = np.mean(img_array.reshape(-1, 3), axis=0)
            avg_gray = np.mean(avg_rgb)
            
            # Calculate scaling factors
            scale_factors = avg_gray / avg_rgb
            
        elif method == "white_patch":
            # White patch assumption: brightest point should be white
            max_rgb = np.max(img_array.reshape(-1, 3), axis=0)
            scale_factors = 255.0 / max_rgb
            
        else:
            raise ValueError(f"Unknown white balance method: {method}")
        
        # Apply scaling factors
        img_array = img_array * scale_factors
        
        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        self.logger.debug(f"Applied white balance ({method}): {scale_factors}")
        return Image.fromarray(img_array)
    
    def get_stage_name(self) -> str:
        """Get the name of this processing stage."""
        return "adjust"
    
    def configure(self, **kwargs) -> None:
        """Configure this stage with new parameters."""
        if "brightness" in kwargs:
            self.settings.BRIGHTNESS_ADJUST = kwargs["brightness"]
        if "contrast" in kwargs:
            self.settings.CONTRAST_ADJUST = kwargs["contrast"]
        if "gamma" in kwargs:
            self.settings.GAMMA_ADJUST = kwargs["gamma"]
