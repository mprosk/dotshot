"""Testing and validation utilities for DotShot."""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import random

from config.settings import SETTINGS


class TestImageLoader:
    """Utility for loading and managing test images."""
    
    def __init__(self, test_dir: str = None):
        if test_dir is None:
            test_dir = SETTINGS["system"].TEST_IMAGES_DIR
        
        self.test_dir = test_dir
        self.logger = logging.getLogger(__name__)
        self._ensure_test_dir()
    
    def _ensure_test_dir(self) -> None:
        """Ensure test images directory exists."""
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir, exist_ok=True)
            self.logger.info(f"Created test images directory: {self.test_dir}")
            
            # Generate some default test images
            self.generate_test_images()
    
    def load_image(self, filename: str) -> Optional[Image.Image]:
        """Load a test image by filename.
        
        Args:
            filename: Name of the image file in the test directory
            
        Returns:
            PIL Image if successful, None otherwise
        """
        filepath = os.path.join(self.test_dir, filename)
        
        try:
            image = Image.open(filepath)
            self.logger.debug(f"Loaded test image: {filepath}")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load test image {filepath}: {e}")
            return None
    
    def list_test_images(self) -> List[str]:
        """List all available test images.
        
        Returns:
            List of test image filenames
        """
        if not os.path.exists(self.test_dir):
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        images = []
        
        for filename in os.listdir(self.test_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                images.append(filename)
        
        return sorted(images)
    
    def generate_test_images(self) -> None:
        """Generate a set of test images for development and testing."""
        self.logger.info("Generating test images...")
        
        # Test image specifications
        test_specs = [
            ("gradient_horizontal.png", self._create_horizontal_gradient),
            ("gradient_vertical.png", self._create_vertical_gradient),
            ("checkerboard.png", self._create_checkerboard),
            ("text_sample.png", self._create_text_sample),
            ("edge_test.png", self._create_edge_test),
            ("noise_test.png", self._create_noise_test),
            ("resolution_test.png", self._create_resolution_test)
        ]
        
        for filename, generator_func in test_specs:
            filepath = os.path.join(self.test_dir, filename)
            
            if not os.path.exists(filepath):
                try:
                    image = generator_func()
                    image.save(filepath)
                    self.logger.debug(f"Generated test image: {filepath}")
                except Exception as e:
                    self.logger.error(f"Failed to generate {filename}: {e}")
    
    def _create_horizontal_gradient(self) -> Image.Image:
        """Create a horizontal gradient test image."""
        width, height = 800, 600
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                gray_value = int(255 * x / width)
                pixels.append((gray_value, gray_value, gray_value))
        
        image.putdata(pixels)
        return image
    
    def _create_vertical_gradient(self) -> Image.Image:
        """Create a vertical gradient test image."""
        width, height = 600, 800
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                gray_value = int(255 * y / height)
                pixels.append((gray_value, gray_value, gray_value))
        
        image.putdata(pixels)
        return image
    
    def _create_checkerboard(self) -> Image.Image:
        """Create a checkerboard test pattern."""
        width, height = 800, 800
        square_size = 50
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                square_x = x // square_size
                square_y = y // square_size
                if (square_x + square_y) % 2 == 0:
                    pixels.append((255, 255, 255))  # White
                else:
                    pixels.append((0, 0, 0))        # Black
        
        image.putdata(pixels)
        return image
    
    def _create_text_sample(self) -> Image.Image:
        """Create a text sample image."""
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font, fall back to built-in if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw various text samples
        draw.text((50, 50), "DotShot Test", fill='black', font=font_large)
        draw.text((50, 120), "Large text for readability testing", fill='black', font=font_medium)
        draw.text((50, 160), "Medium text with more detail", fill='black', font=font_small)
        draw.text((50, 190), "Small text for fine detail testing", fill='black', font=font_small)
        
        # Add some geometric shapes
        draw.rectangle([50, 250, 200, 350], outline='black', width=2)
        draw.ellipse([250, 250, 400, 350], outline='black', width=2)
        draw.line([450, 250, 600, 350], fill='black', width=3)
        
        return image
    
    def _create_edge_test(self) -> Image.Image:
        """Create an image for testing edge detection."""
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw various edges and shapes
        # Vertical edges
        draw.rectangle([100, 100, 150, 500], fill='black')
        draw.rectangle([200, 100, 250, 500], fill='gray')
        
        # Horizontal edges
        draw.rectangle([300, 150, 700, 200], fill='black')
        draw.rectangle([300, 250, 700, 300], fill='gray')
        
        # Diagonal edges
        points = [(400, 350), (500, 450), (600, 350), (500, 400)]
        draw.polygon(points, fill='black')
        
        # Circles for curved edges
        draw.ellipse([150, 350, 250, 450], fill='black')
        draw.ellipse([160, 360, 240, 440], fill='white')
        
        return image
    
    def _create_noise_test(self) -> Image.Image:
        """Create an image with various noise patterns."""
        width, height = 800, 600
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                if x < width // 3:
                    # Clean gradient
                    gray_value = int(255 * y / height)
                    pixels.append((gray_value, gray_value, gray_value))
                elif x < 2 * width // 3:
                    # Noisy gradient
                    base_value = int(255 * y / height)
                    noise = random.randint(-30, 30)
                    gray_value = max(0, min(255, base_value + noise))
                    pixels.append((gray_value, gray_value, gray_value))
                else:
                    # Random noise
                    gray_value = random.randint(0, 255)
                    pixels.append((gray_value, gray_value, gray_value))
        
        image.putdata(pixels)
        return image
    
    def _create_resolution_test(self) -> Image.Image:
        """Create a resolution test pattern."""
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw increasingly fine patterns
        for i in range(10):
            line_width = 1 + i // 2
            spacing = 5 + i * 2
            
            y_start = 50 + i * 50
            for x in range(0, width, spacing * 2):
                draw.line([(x, y_start), (x + spacing, y_start)], 
                         fill='black', width=line_width)
        
        return image


class ImageProcessor:
    """Utility for processing and analyzing images during testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Union[int, float, str]]:
        """Analyze image properties and statistics.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with image analysis results
        """
        img_array = np.array(image)
        
        analysis = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'size_bytes': len(image.tobytes()),
            'aspect_ratio': image.width / image.height
        }
        
        if image.mode in ['L', 'RGB']:
            if image.mode == 'RGB':
                # Convert to grayscale for analysis
                gray_array = np.array(image.convert('L'))
            else:
                gray_array = img_array
            
            analysis.update({
                'mean_brightness': float(np.mean(gray_array)),
                'std_brightness': float(np.std(gray_array)),
                'min_value': int(np.min(gray_array)),
                'max_value': int(np.max(gray_array)),
                'dynamic_range': int(np.max(gray_array) - np.min(gray_array))
            })
            
            # Calculate edge density
            edges = self._detect_simple_edges(gray_array)
            analysis['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        return analysis
    
    def _detect_simple_edges(self, img_array: np.ndarray) -> np.ndarray:
        """Simple edge detection for analysis."""
        # Simple Sobel-like edge detection
        height, width = img_array.shape
        edges = np.zeros_like(img_array)
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gx = (-1 * img_array[y-1, x-1] + 1 * img_array[y-1, x+1] +
                      -2 * img_array[y, x-1] + 2 * img_array[y, x+1] +
                      -1 * img_array[y+1, x-1] + 1 * img_array[y+1, x+1])
                
                gy = (-1 * img_array[y-1, x-1] + -2 * img_array[y-1, x] + -1 * img_array[y-1, x+1] +
                       1 * img_array[y+1, x-1] +  2 * img_array[y+1, x] +  1 * img_array[y+1, x+1])
                
                edges[y, x] = min(255, abs(gx) + abs(gy))
        
        return edges
    
    def compare_images(self, image1: Image.Image, image2: Image.Image) -> Dict[str, float]:
        """Compare two images and return similarity metrics.
        
        Args:
            image1: First PIL Image
            image2: Second PIL Image
            
        Returns:
            Dictionary with comparison metrics
        """
        # Ensure images are the same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale arrays
        array1 = np.array(image1.convert('L'))
        array2 = np.array(image2.convert('L'))
        
        # Calculate metrics
        mse = float(np.mean((array1 - array2) ** 2))
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Structural similarity approximation
        mean1, mean2 = np.mean(array1), np.mean(array2)
        var1, var2 = np.var(array1), np.var(array2)
        covar = np.mean((array1 - mean1) * (array2 - mean2))
        
        ssim = ((2 * mean1 * mean2 + 1) * (2 * covar + 1)) / \
               ((mean1**2 + mean2**2 + 1) * (var1 + var2 + 1))
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': float(ssim),
            'correlation': float(np.corrcoef(array1.flat, array2.flat)[0, 1])
        }


def validate_pipeline(pipeline, test_images: List[Image.Image]) -> Dict[str, Dict[str, float]]:
    """Validate processing pipeline with test images.
    
    Args:
        pipeline: ProcessingPipeline instance
        test_images: List of test PIL Images
        
    Returns:
        Dictionary with validation results for each test image
    """
    logger = logging.getLogger(__name__)
    processor = ImageProcessor()
    results = {}
    
    for i, test_image in enumerate(test_images):
        logger.info(f"Validating with test image {i+1}/{len(test_images)}")
        
        try:
            # Process the image
            processed = pipeline.process_image(test_image)
            
            # Analyze results
            original_analysis = processor.analyze_image(test_image)
            processed_analysis = processor.analyze_image(processed)
            comparison = processor.compare_images(test_image, processed)
            
            results[f"test_image_{i+1}"] = {
                'original': original_analysis,
                'processed': processed_analysis,
                'comparison': comparison,
                'processing_times': pipeline.get_stage_timings()
            }
            
        except Exception as e:
            logger.error(f"Validation failed for test image {i+1}: {e}")
            results[f"test_image_{i+1}"] = {'error': str(e)}
    
    return results


def benchmark_processing_pipeline(pipeline, test_image: Image.Image, 
                                iterations: int = 5) -> Dict[str, float]:
    """Benchmark processing pipeline performance.
    
    Args:
        pipeline: ProcessingPipeline instance
        test_image: Test PIL Image
        iterations: Number of iterations to average
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    logger = logging.getLogger(__name__)
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        processed = pipeline.process_image(test_image)
        end_time = time.time()
        
        iteration_time = end_time - start_time
        times.append(iteration_time)
        logger.debug(f"Iteration {i+1}: {iteration_time:.3f}s")
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'iterations': iterations
    }
