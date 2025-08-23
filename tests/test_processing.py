"""Tests for DotShot image processing pipeline."""

import unittest
import numpy as np
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processing import ProcessingPipeline, DitheringProcessor
from processing.stages import ResizeStage, CropStage, AdjustmentStage, EdgeEnhancementStage


class TestProcessingStages(unittest.TestCase):
    """Test individual processing stages."""
    
    def setUp(self):
        """Set up test images."""
        # Create a test image
        self.test_image = Image.new('RGB', (800, 600), color='white')
        
        # Add some content to make it more realistic
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.test_image)
        draw.rectangle([100, 100, 300, 300], fill='black')
        draw.ellipse([400, 200, 600, 400], fill='gray')
    
    def test_resize_stage(self):
        """Test resize stage functionality."""
        stage = ResizeStage()
        
        # Test basic resize
        resized = stage.process(self.test_image, target_width=400, target_height=300)
        self.assertEqual(resized.size, (400, 300))
        
        # Test aspect ratio preservation
        resized_aspect = stage.process(self.test_image, 
                                     target_width=400, 
                                     target_height=300,
                                     maintain_aspect_ratio=True)
        self.assertEqual(resized_aspect.size, (400, 300))
        
    def test_crop_stage(self):
        """Test crop stage functionality."""
        stage = CropStage()
        
        # Test center crop
        cropped = stage.process(self.test_image, crop_margin=0.1)
        self.assertLess(cropped.width, self.test_image.width)
        self.assertLess(cropped.height, self.test_image.height)
        
        # Test explicit crop box
        crop_box = (100, 100, 400, 300)
        cropped_box = stage.process(self.test_image, crop_box=crop_box)
        self.assertEqual(cropped_box.size, (300, 200))  # width=400-100, height=300-100
    
    def test_adjustment_stage(self):
        """Test adjustment stage functionality."""
        stage = AdjustmentStage()
        
        # Test brightness adjustment
        brighter = stage.process(self.test_image, brightness=1.5)
        self.assertEqual(brighter.size, self.test_image.size)
        
        # Test contrast adjustment
        higher_contrast = stage.process(self.test_image, contrast=1.5)
        self.assertEqual(higher_contrast.size, self.test_image.size)
        
        # Test gamma correction
        gamma_corrected = stage.process(self.test_image, gamma=0.8)
        self.assertEqual(gamma_corrected.size, self.test_image.size)
    
    def test_edge_enhancement_stage(self):
        """Test edge enhancement stage functionality."""
        stage = EdgeEnhancementStage()
        
        # Test edge enhancement
        enhanced = stage.process(self.test_image, strength=2.0)
        self.assertEqual(enhanced.size, self.test_image.size)
        
        # Test different methods
        methods = ["unsharp_mask", "edge_enhance", "high_pass", "laplacian"]
        for method in methods:
            enhanced_method = stage.process(self.test_image, method=method)
            self.assertEqual(enhanced_method.size, self.test_image.size)


class TestProcessingPipeline(unittest.TestCase):
    """Test the complete processing pipeline."""
    
    def setUp(self):
        """Set up test pipeline and image."""
        self.pipeline = ProcessingPipeline()
        self.test_image = Image.new('RGB', (1200, 900), color='white')
        
        # Add some content
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.test_image)
        draw.rectangle([200, 200, 600, 600], fill='black')
        draw.ellipse([700, 300, 1000, 600], fill='gray')
    
    def test_pipeline_processing(self):
        """Test complete pipeline processing."""
        processed = self.pipeline.process_image(self.test_image)
        
        # Should produce an image
        self.assertIsInstance(processed, Image.Image)
        
        # Should have stage results
        stage_results = self.pipeline.get_all_stage_results()
        self.assertGreater(len(stage_results), 0)
        self.assertIn('original', stage_results)
        
        # Should have timing information
        timings = self.pipeline.get_stage_timings()
        self.assertGreater(len(timings), 0)
    
    def test_pipeline_stage_configuration(self):
        """Test pipeline stage configuration."""
        # Configure a stage
        success = self.pipeline.configure_stage('resize', target_width=300, target_height=400)
        self.assertTrue(success)
        
        # Try to configure non-existent stage
        failure = self.pipeline.configure_stage('nonexistent', some_param=123)
        self.assertFalse(failure)


class TestDitheringProcessor(unittest.TestCase):
    """Test dithering algorithms."""
    
    def setUp(self):
        """Set up test image for dithering."""
        # Create a grayscale gradient image
        width, height = 400, 300
        self.test_image = Image.new('L', (width, height))
        
        pixels = []
        for y in range(height):
            for x in range(width):
                # Create gradient + some patterns
                gradient_value = int(255 * x / width)
                pattern_value = 50 if (x // 20 + y // 20) % 2 == 0 else 0
                pixel_value = min(255, gradient_value + pattern_value)
                pixels.append(pixel_value)
        
        self.test_image.putdata(pixels)
    
    def test_dithering_algorithms(self):
        """Test different dithering algorithms."""
        processor = DitheringProcessor()
        
        algorithms = ["floyd_steinberg", "ordered", "threshold", "atkinson", "sierra"]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                dithered = processor.apply_dithering(self.test_image, algorithm=algorithm)
                
                # Should produce a grayscale image
                self.assertIn(dithered.mode, ['L', '1'])
                self.assertEqual(dithered.size, self.test_image.size)
                
                # Should contain only black and white pixels (after conversion)
                if dithered.mode == 'L':
                    pixel_values = set(dithered.getdata())
                    self.assertTrue(pixel_values.issubset({0, 255}))
    
    def test_dithering_comparison(self):
        """Test dithering algorithm comparison."""
        processor = DitheringProcessor()
        
        results = processor.compare_algorithms(self.test_image)
        
        # Should return multiple results
        self.assertGreater(len(results), 1)
        
        # Each result should be a valid image
        for algorithm, dithered in results.items():
            self.assertIsInstance(dithered, Image.Image)
            self.assertEqual(dithered.size, self.test_image.size)
    
    def test_dot_matrix_simulation(self):
        """Test dot matrix printer simulation."""
        processor = DitheringProcessor()
        
        # First dither the image
        dithered = processor.apply_dithering(self.test_image)
        
        # Then create dot matrix simulation
        simulated = processor.create_dot_matrix_simulation(dithered, dot_size=3, dot_spacing=1)
        
        # Should be larger than original due to dot spacing
        self.assertGreater(simulated.width, dithered.width)
        self.assertGreater(simulated.height, dithered.height)
    
    def test_printer_optimization(self):
        """Test printer optimization."""
        processor = DitheringProcessor()
        
        optimized, settings = processor.optimize_for_printer(self.test_image)
        
        # Should return an image and settings
        self.assertIsInstance(optimized, Image.Image)
        self.assertIsInstance(settings, dict)
        
        # Settings should contain expected keys
        expected_keys = ['algorithm', 'threshold', 'score']
        for key in expected_keys:
            self.assertIn(key, settings)


class TestImageAnalysis(unittest.TestCase):
    """Test image analysis utilities."""
    
    def setUp(self):
        """Create test images."""
        # Solid color image
        self.solid_image = Image.new('RGB', (100, 100), color='gray')
        
        # High contrast image
        self.contrast_image = Image.new('RGB', (100, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.contrast_image)
        draw.rectangle([0, 0, 50, 100], fill='black')
    
    def test_image_properties(self):
        """Test basic image property extraction."""
        from utils.testing import ImageProcessor
        
        processor = ImageProcessor()
        
        # Test solid image
        analysis = processor.analyze_image(self.solid_image)
        
        expected_keys = ['width', 'height', 'mode', 'aspect_ratio', 'mean_brightness']
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        self.assertEqual(analysis['width'], 100)
        self.assertEqual(analysis['height'], 100)
        self.assertEqual(analysis['mode'], 'RGB')
        self.assertEqual(analysis['aspect_ratio'], 1.0)
    
    def test_image_comparison(self):
        """Test image comparison metrics."""
        from utils.testing import ImageProcessor
        
        processor = ImageProcessor()
        
        # Compare identical images
        same_comparison = processor.compare_images(self.solid_image, self.solid_image)
        
        self.assertIn('mse', same_comparison)
        self.assertIn('psnr', same_comparison)
        self.assertEqual(same_comparison['mse'], 0.0)
        
        # Compare different images
        diff_comparison = processor.compare_images(self.solid_image, self.contrast_image)
        
        self.assertGreater(diff_comparison['mse'], 0.0)
        self.assertLess(diff_comparison['psnr'], float('inf'))


if __name__ == '__main__':
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestProcessingStages))
    suite.addTest(loader.loadTestsFromTestCase(TestProcessingPipeline))
    suite.addTest(loader.loadTestsFromTestCase(TestDitheringProcessor))
    suite.addTest(loader.loadTestsFromTestCase(TestImageAnalysis))
    
    # Run tests
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
