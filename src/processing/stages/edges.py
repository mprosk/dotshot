"""Edge enhancement processing stage."""

import logging
import numpy as np
from typing import Tuple
from PIL import Image, ImageFilter, ImageEnhance

from config.settings import SETTINGS


class EdgeEnhancementStage:
    """Handles edge enhancement and sharpening for better dot matrix printing."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Apply edge enhancement to improve dot matrix printing.
        
        Args:
            image: Input PIL Image
            **kwargs: Optional override parameters
            
        Returns:
            Edge-enhanced PIL Image
        """
        strength = kwargs.get("strength", self.settings.EDGE_ENHANCE_STRENGTH)
        method = kwargs.get("method", "unsharp_mask")
        
        if method == "unsharp_mask":
            enhanced_image = self._apply_unsharp_mask(image, **kwargs)
        elif method == "edge_enhance":
            enhanced_image = self._apply_edge_enhance(image, strength)
        elif method == "high_pass":
            enhanced_image = self._apply_high_pass_filter(image, strength)
        elif method == "laplacian":
            enhanced_image = self._apply_laplacian_sharpening(image, strength)
        else:
            self.logger.warning(f"Unknown enhancement method: {method}, using unsharp_mask")
            enhanced_image = self._apply_unsharp_mask(image, **kwargs)
        
        self.logger.debug(f"Applied edge enhancement using {method} with strength {strength}")
        return enhanced_image
    
    def _apply_unsharp_mask(self, image: Image.Image, **kwargs) -> Image.Image:
        """Apply unsharp mask sharpening.
        
        Args:
            image: Input PIL Image
            **kwargs: Parameters for unsharp mask
            
        Returns:
            Sharpened PIL Image
        """
        radius = kwargs.get("radius", self.settings.UNSHARP_MASK_RADIUS)
        percent = kwargs.get("percent", self.settings.UNSHARP_MASK_PERCENT)
        threshold = kwargs.get("threshold", self.settings.UNSHARP_MASK_THRESHOLD)
        
        # PIL's UnsharpMask filter
        unsharp_filter = ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold
        )
        
        return image.filter(unsharp_filter)
    
    def _apply_edge_enhance(self, image: Image.Image, strength: float) -> Image.Image:
        """Apply edge enhancement using PIL's built-in filter.
        
        Args:
            image: Input PIL Image
            strength: Enhancement strength
            
        Returns:
            Edge-enhanced PIL Image
        """
        # Apply edge enhance filter
        edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
        
        # Blend with original based on strength
        return Image.blend(image, edge_enhanced, strength / 2.0)
    
    def _apply_high_pass_filter(self, image: Image.Image, strength: float) -> Image.Image:
        """Apply high-pass filter for edge enhancement.
        
        Args:
            image: Input PIL Image
            strength: Filter strength
            
        Returns:
            High-pass filtered PIL Image
        """
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Apply Gaussian blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        blurred_array = np.array(blurred).astype(np.float32)
        
        # High-pass = original - blurred
        high_pass = img_array - blurred_array
        
        # Add high-pass to original with strength factor
        enhanced = img_array + (high_pass * strength)
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def _apply_laplacian_sharpening(self, image: Image.Image, strength: float) -> Image.Image:
        """Apply Laplacian sharpening for edge enhancement.
        
        Args:
            image: Input PIL Image
            strength: Sharpening strength
            
        Returns:
            Laplacian-sharpened PIL Image
        """
        import cv2
        
        # Convert to OpenCV format
        img_cv = np.array(image)
        
        if len(img_cv.shape) == 3:
            # Color image - work in grayscale for Laplacian
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Convert back to 3-channel for blending
            laplacian_3ch = np.stack([laplacian] * 3, axis=-1)
            
            # Apply sharpening
            sharpened = img_cv.astype(np.float64) - (strength * laplacian_3ch)
            
        else:
            # Grayscale image
            laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
            sharpened = img_cv.astype(np.float64) - (strength * laplacian)
        
        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sharpened)
    
    def detect_edges(self, image: Image.Image, method: str = "canny") -> Image.Image:
        """Detect edges in image for analysis.
        
        Args:
            image: Input PIL Image
            method: Edge detection method ("canny", "sobel", "prewitt")
            
        Returns:
            Edge map as PIL Image
        """
        import cv2
        
        # Convert to grayscale for edge detection
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        img_cv = np.array(gray_image)
        
        if method == "canny":
            edges = cv2.Canny(img_cv, 50, 150)
        elif method == "sobel":
            sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        elif method == "prewitt":
            kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
            kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
            prewittx = cv2.filter2D(img_cv, -1, kernelx)
            prewitty = cv2.filter2D(img_cv, -1, kernely)
            edges = np.sqrt(prewittx**2 + prewitty**2).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        return Image.fromarray(edges, 'L')
    
    def calculate_edge_density(self, image: Image.Image) -> float:
        """Calculate edge density metric for the image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Edge density as a float (0.0 to 1.0)
        """
        edge_map = self.detect_edges(image, "canny")
        edge_array = np.array(edge_map)
        
        # Calculate ratio of edge pixels to total pixels
        edge_pixels = np.sum(edge_array > 0)
        total_pixels = edge_array.size
        
        density = edge_pixels / total_pixels
        
        self.logger.debug(f"Edge density: {density:.3f}")
        return density
    
    def get_stage_name(self) -> str:
        """Get the name of this processing stage."""
        return "edges"
    
    def configure(self, **kwargs) -> None:
        """Configure this stage with new parameters."""
        if "strength" in kwargs:
            self.settings.EDGE_ENHANCE_STRENGTH = kwargs["strength"]
        if "unsharp_radius" in kwargs:
            self.settings.UNSHARP_MASK_RADIUS = kwargs["unsharp_radius"]
        if "unsharp_percent" in kwargs:
            self.settings.UNSHARP_MASK_PERCENT = kwargs["unsharp_percent"]
        if "unsharp_threshold" in kwargs:
            self.settings.UNSHARP_MASK_THRESHOLD = kwargs["unsharp_threshold"]
