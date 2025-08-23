"""Image processing pipeline coordinator."""

import time
import logging
from typing import List, Dict, Any, Optional, Protocol
from PIL import Image
from abc import ABC, abstractmethod

from config.settings import SETTINGS
from .stages import ResizeStage, CropStage, AdjustmentStage, EdgeEnhancementStage


class ProcessingStage(Protocol):
    """Protocol for processing stage implementations."""
    
    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process an image through this stage."""
        ...
    
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this processing stage."""
        ...


class ProcessingPipeline:
    """Coordinates the image processing pipeline."""
    
    def __init__(self):
        self.settings = SETTINGS["processing"]
        self.system_settings = SETTINGS["system"]
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing stages
        self.stages: List[ProcessingStage] = [
            ResizeStage(),
            CropStage(), 
            AdjustmentStage(),
            EdgeEnhancementStage()
        ]
        
        self.stage_results: Dict[str, Image.Image] = {}
        self.stage_timings: Dict[str, float] = {}
    
    def process_image(self, 
                     image: Image.Image, 
                     save_intermediates: bool = None,
                     display_stages: bool = False) -> Image.Image:
        """Process an image through the complete pipeline.
        
        Args:
            image: Input PIL Image
            save_intermediates: Whether to save intermediate results
            display_stages: Whether to display each stage result
            
        Returns:
            Processed PIL Image
        """
        if save_intermediates is None:
            save_intermediates = self.system_settings.SAVE_INTERMEDIATE_IMAGES
        
        current_image = image.copy()
        self.stage_results.clear()
        self.stage_timings.clear()
        
        self.logger.info(f"Starting pipeline processing for image {current_image.size}")
        pipeline_start = time.time()
        
        # Store original image
        self.stage_results["original"] = image.copy()
        
        # Process through each stage
        for stage in self.stages:
            stage_name = stage.get_stage_name()
            self.logger.debug(f"Processing stage: {stage_name}")
            
            stage_start = time.time()
            current_image = stage.process(current_image)
            stage_time = time.time() - stage_start
            
            self.stage_timings[stage_name] = stage_time
            self.stage_results[stage_name] = current_image.copy()
            
            if self.system_settings.DISPLAY_PROCESSING_TIME:
                self.logger.info(f"{stage_name} completed in {stage_time:.3f}s")
                
            if save_intermediates:
                self._save_intermediate(current_image, stage_name)
            
            if display_stages:
                self._display_stage_result(current_image, stage_name)
        
        total_time = time.time() - pipeline_start
        self.logger.info(f"Pipeline processing completed in {total_time:.3f}s")
        
        return current_image
    
    def get_stage_result(self, stage_name: str) -> Optional[Image.Image]:
        """Get the result of a specific processing stage.
        
        Args:
            stage_name: Name of the stage to retrieve
            
        Returns:
            PIL Image from that stage, or None if not found
        """
        return self.stage_results.get(stage_name)
    
    def get_all_stage_results(self) -> Dict[str, Image.Image]:
        """Get results from all processing stages.
        
        Returns:
            Dictionary mapping stage names to PIL Images
        """
        return self.stage_results.copy()
    
    def get_stage_timings(self) -> Dict[str, float]:
        """Get timing information for all stages.
        
        Returns:
            Dictionary mapping stage names to execution times in seconds
        """
        return self.stage_timings.copy()
    
    def _save_intermediate(self, image: Image.Image, stage_name: str) -> None:
        """Save intermediate processing result to file."""
        import os
        
        output_dir = self.system_settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"intermediate_{stage_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        try:
            image.save(filepath)
            self.logger.debug(f"Saved intermediate result: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate result: {e}")
    
    def _display_stage_result(self, image: Image.Image, stage_name: str) -> None:
        """Display processing stage result (for debugging)."""
        try:
            # This will be implemented in the display utils
            from ..utils.display import show_image
            show_image(image, title=f"Stage: {stage_name}")
        except ImportError:
            self.logger.warning("Display utilities not available")
        except Exception as e:
            self.logger.error(f"Failed to display stage result: {e}")
    
    def configure_stage(self, stage_name: str, **kwargs) -> bool:
        """Configure a specific processing stage.
        
        Args:
            stage_name: Name of the stage to configure
            **kwargs: Stage-specific configuration parameters
            
        Returns:
            True if stage was configured successfully
        """
        for stage in self.stages:
            if stage.get_stage_name() == stage_name:
                if hasattr(stage, 'configure'):
                    stage.configure(**kwargs)
                    self.logger.info(f"Configured stage {stage_name}")
                    return True
                else:
                    self.logger.warning(f"Stage {stage_name} does not support configuration")
                    return False
        
        self.logger.error(f"Stage {stage_name} not found")
        return False
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the processing pipeline.
        
        Returns:
            Dictionary with pipeline configuration and stage information
        """
        return {
            "stages": [stage.get_stage_name() for stage in self.stages],
            "settings": {
                "target_size": (self.settings.PRINTER_WIDTH_PIXELS, 
                              self.settings.PRINTER_HEIGHT_PIXELS),
                "resize_algorithm": self.settings.RESIZE_ALGORITHM,
                "maintain_aspect_ratio": self.settings.MAINTAIN_ASPECT_RATIO,
                "brightness_adjust": self.settings.BRIGHTNESS_ADJUST,
                "contrast_adjust": self.settings.CONTRAST_ADJUST,
                "edge_enhance_strength": self.settings.EDGE_ENHANCE_STRENGTH
            },
            "last_processing_times": self.stage_timings
        }
