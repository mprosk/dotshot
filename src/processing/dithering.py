"""Dithering algorithms for dot matrix printer simulation."""

import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from config.settings import SETTINGS


class DitheringProcessor:
    """Handles various dithering algorithms for print preview and optimization."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.logger = logging.getLogger(__name__)
    
    def apply_dithering(self, 
                       image: Image.Image, 
                       algorithm: str = None,
                       threshold: int = None) -> Image.Image:
        """Apply dithering algorithm to convert image to 1-bit output.
        
        Args:
            image: Input PIL Image
            algorithm: Dithering algorithm ("floyd_steinberg", "ordered", "threshold", "atkinson")
            threshold: Threshold value for threshold dithering
            
        Returns:
            Dithered 1-bit PIL Image
        """
        if algorithm is None:
            algorithm = self.settings.DITHER_ALGORITHM
        if threshold is None:
            threshold = self.settings.DITHER_THRESHOLD
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image.copy()
        
        self.logger.debug(f"Applying {algorithm} dithering with threshold {threshold}")
        
        if algorithm == "floyd_steinberg":
            dithered = self._floyd_steinberg_dithering(gray_image, threshold)
        elif algorithm == "ordered":
            dithered = self._ordered_dithering(gray_image, threshold)
        elif algorithm == "threshold":
            dithered = self._threshold_dithering(gray_image, threshold)
        elif algorithm == "atkinson":
            dithered = self._atkinson_dithering(gray_image, threshold)
        elif algorithm == "sierra":
            dithered = self._sierra_dithering(gray_image, threshold)
        else:
            self.logger.warning(f"Unknown dithering algorithm: {algorithm}, using floyd_steinberg")
            dithered = self._floyd_steinberg_dithering(gray_image, threshold)
        
        return dithered
    
    def _floyd_steinberg_dithering(self, image: Image.Image, threshold: int) -> Image.Image:
        """Apply Floyd-Steinberg error diffusion dithering.
        
        Args:
            image: Input grayscale PIL Image
            threshold: Threshold for black/white decision
            
        Returns:
            Dithered 1-bit PIL Image
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape
        
        # Floyd-Steinberg error distribution matrix
        # Current pixel gets error, distribute to:
        #     X  7/16
        # 3/16 5/16 1/16
        
        for y in range(height):
            for x in range(width):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x < width - 1:
                    img_array[y, x + 1] += error * 7 / 16
                if y < height - 1:
                    if x > 0:
                        img_array[y + 1, x - 1] += error * 3 / 16
                    img_array[y + 1, x] += error * 5 / 16
                    if x < width - 1:
                        img_array[y + 1, x + 1] += error * 1 / 16
        
        # Clip to valid range and convert to uint8
        result_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'L')
    
    def _ordered_dithering(self, image: Image.Image, threshold: int) -> Image.Image:
        """Apply ordered (Bayer) dithering.
        
        Args:
            image: Input grayscale PIL Image
            threshold: Base threshold (modified by dither matrix)
            
        Returns:
            Dithered 1-bit PIL Image
        """
        # 8x8 Bayer dithering matrix
        bayer_matrix = np.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ]) * 4  # Scale for better distribution
        
        img_array = np.array(image)
        height, width = img_array.shape
        result_array = np.zeros_like(img_array)
        
        for y in range(height):
            for x in range(width):
                matrix_val = bayer_matrix[y % 8, x % 8]
                adjusted_threshold = threshold + matrix_val - 128
                result_array[y, x] = 255 if img_array[y, x] > adjusted_threshold else 0
        
        return Image.fromarray(result_array, 'L')
    
    def _threshold_dithering(self, image: Image.Image, threshold: int) -> Image.Image:
        """Apply simple threshold dithering.
        
        Args:
            image: Input grayscale PIL Image
            threshold: Threshold value for black/white decision
            
        Returns:
            Dithered 1-bit PIL Image
        """
        img_array = np.array(image)
        result_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(result_array, 'L')
    
    def _atkinson_dithering(self, image: Image.Image, threshold: int) -> Image.Image:
        """Apply Atkinson error diffusion dithering.
        
        Args:
            image: Input grayscale PIL Image
            threshold: Threshold for black/white decision
            
        Returns:
            Dithered 1-bit PIL Image
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape
        
        # Atkinson error distribution pattern:
        #     X  1/8  1/8
        # 1/8 1/8  1/8
        #     1/8
        
        for y in range(height):
            for x in range(width):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error (Atkinson uses 1/8 instead of full error)
                error_fraction = error / 8
                
                if x < width - 1:
                    img_array[y, x + 1] += error_fraction
                if x < width - 2:
                    img_array[y, x + 2] += error_fraction
                if y < height - 1:
                    if x > 0:
                        img_array[y + 1, x - 1] += error_fraction
                    img_array[y + 1, x] += error_fraction
                    if x < width - 1:
                        img_array[y + 1, x + 1] += error_fraction
                if y < height - 2:
                    img_array[y + 2, x] += error_fraction
        
        result_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'L')
    
    def _sierra_dithering(self, image: Image.Image, threshold: int) -> Image.Image:
        """Apply Sierra error diffusion dithering.
        
        Args:
            image: Input grayscale PIL Image
            threshold: Threshold for black/white decision
            
        Returns:
            Dithered 1-bit PIL Image
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape
        
        # Sierra error distribution pattern:
        #         X  5/32 3/32
        # 2/32 4/32 5/32 4/32 2/32
        #      2/32 3/32 2/32
        
        for y in range(height):
            for x in range(width):
                old_pixel = img_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                img_array[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error according to Sierra pattern
                if x < width - 1:
                    img_array[y, x + 1] += error * 5 / 32
                if x < width - 2:
                    img_array[y, x + 2] += error * 3 / 32
                    
                if y < height - 1:
                    if x > 1:
                        img_array[y + 1, x - 2] += error * 2 / 32
                    if x > 0:
                        img_array[y + 1, x - 1] += error * 4 / 32
                    img_array[y + 1, x] += error * 5 / 32
                    if x < width - 1:
                        img_array[y + 1, x + 1] += error * 4 / 32
                    if x < width - 2:
                        img_array[y + 1, x + 2] += error * 2 / 32
                        
                if y < height - 2:
                    if x > 0:
                        img_array[y + 2, x - 1] += error * 2 / 32
                    img_array[y + 2, x] += error * 3 / 32
                    if x < width - 1:
                        img_array[y + 2, x + 1] += error * 2 / 32
        
        result_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'L')
    
    def create_dot_matrix_simulation(self, dithered_image: Image.Image, 
                                   dot_size: int = 2,
                                   dot_spacing: int = 1) -> Image.Image:
        """Create a visual simulation of dot matrix printing.
        
        Args:
            dithered_image: Input 1-bit dithered image
            dot_size: Size of each dot in pixels
            dot_spacing: Spacing between dots in pixels
            
        Returns:
            Dot matrix simulation as PIL Image
        """
        img_array = np.array(dithered_image)
        height, width = img_array.shape
        
        # Calculate output dimensions
        dot_pitch = dot_size + dot_spacing
        out_height = height * dot_pitch
        out_width = width * dot_pitch
        
        # Create output array (white background)
        output_array = np.full((out_height, out_width), 255, dtype=np.uint8)
        
        # Draw dots for black pixels
        for y in range(height):
            for x in range(width):
                if img_array[y, x] < 128:  # Black pixel in dithered image
                    # Calculate dot position
                    dot_y_start = y * dot_pitch
                    dot_x_start = x * dot_pitch
                    
                    # Draw circular dot
                    self._draw_circular_dot(output_array, 
                                          dot_x_start, dot_y_start, 
                                          dot_size)
        
        return Image.fromarray(output_array, 'L')
    
    def _draw_circular_dot(self, array: np.ndarray, 
                          center_x: int, center_y: int, 
                          size: int) -> None:
        """Draw a circular dot in the array.
        
        Args:
            array: Output array to draw on
            center_x: X coordinate of dot center
            center_y: Y coordinate of dot center
            size: Diameter of the dot
        """
        radius = size / 2
        height, width = array.shape
        
        for dy in range(size):
            for dx in range(size):
                y = center_y + dy
                x = center_x + dx
                
                if 0 <= y < height and 0 <= x < width:
                    # Check if point is inside circle
                    dist = np.sqrt((dx - radius) ** 2 + (dy - radius) ** 2)
                    if dist <= radius:
                        array[y, x] = 0  # Black dot
    
    def compare_algorithms(self, image: Image.Image) -> dict:
        """Compare different dithering algorithms on the same image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with algorithm names as keys and dithered images as values
        """
        algorithms = ["floyd_steinberg", "ordered", "threshold", "atkinson", "sierra"]
        results = {}
        
        for algorithm in algorithms:
            try:
                dithered = self.apply_dithering(image, algorithm=algorithm)
                results[algorithm] = dithered
                self.logger.debug(f"Successfully applied {algorithm} dithering")
            except Exception as e:
                self.logger.error(f"Failed to apply {algorithm} dithering: {e}")
        
        return results
    
    def optimize_for_printer(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Find the best dithering settings for the printer.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (optimized image, optimization info)
        """
        # Test different algorithms and thresholds
        best_image = None
        best_score = -1
        best_settings = {}
        
        algorithms = ["floyd_steinberg", "atkinson", "sierra"]
        thresholds = [100, 128, 156, 180]
        
        for algorithm in algorithms:
            for threshold in thresholds:
                try:
                    dithered = self.apply_dithering(image, 
                                                  algorithm=algorithm, 
                                                  threshold=threshold)
                    
                    # Calculate quality score (balance of detail preservation and printability)
                    score = self._calculate_print_quality_score(dithered)
                    
                    if score > best_score:
                        best_score = score
                        best_image = dithered
                        best_settings = {
                            "algorithm": algorithm,
                            "threshold": threshold,
                            "score": score
                        }
                        
                except Exception as e:
                    self.logger.error(f"Failed optimization with {algorithm}, {threshold}: {e}")
        
        self.logger.info(f"Best dithering settings: {best_settings}")
        return best_image, best_settings
    
    def _calculate_print_quality_score(self, dithered_image: Image.Image) -> float:
        """Calculate a quality score for print optimization.
        
        Args:
            dithered_image: Dithered 1-bit image
            
        Returns:
            Quality score (higher is better)
        """
        img_array = np.array(dithered_image)
        
        # Calculate metrics
        black_pixel_ratio = np.sum(img_array == 0) / img_array.size
        
        # Edge preservation (count transitions)
        horizontal_transitions = np.sum(np.diff(img_array, axis=1) != 0)
        vertical_transitions = np.sum(np.diff(img_array, axis=0) != 0)
        total_transitions = horizontal_transitions + vertical_transitions
        transition_density = total_transitions / img_array.size
        
        # Avoid extremes (too much black ink or too sparse)
        ink_penalty = abs(black_pixel_ratio - 0.3) * 2  # Penalize deviation from ~30% black
        
        # Score: favor detail preservation but penalize extreme ink usage
        score = transition_density - ink_penalty
        
        return max(0, score)  # Ensure non-negative score
