"""Camera capture module using picamera2."""

import time
import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    from picamera2 import Picamera2
    from libcamera import controls
    CAMERA_AVAILABLE = True
except ImportError:
    logging.warning("picamera2 not available. Camera functionality will be limited.")
    CAMERA_AVAILABLE = False

from config.settings import SETTINGS


class CameraController:
    """Controls the Raspberry Pi camera for image capture."""
    
    def __init__(self):
        self.camera: Optional[Picamera2] = None
        self.camera_settings = SETTINGS["camera"]
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize the camera.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise.
        """
        if not CAMERA_AVAILABLE:
            self.logger.error("Camera not available - picamera2 not installed")
            return False
            
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={"size": self.camera_settings.RESOLUTION}
            )
            self.camera.configure(config)
            
            # Set camera controls
            controls_dict = {
                controls.AwbMode: getattr(controls.AwbModeEnum, 
                                        self.camera_settings.AWB_MODE.capitalize(), 
                                        controls.AwbModeEnum.Auto),
                controls.AeExposureMode: getattr(controls.AeExposureModeEnum,
                                               self.camera_settings.EXPOSURE_MODE.capitalize(),
                                               controls.AeExposureModeEnum.Normal),
            }
            
            if self.camera_settings.ISO > 0:
                controls_dict[controls.AnalogueGain] = self.camera_settings.ISO / 100.0
                
            self.camera.set_controls(controls_dict)
            
            # Start camera
            self.camera.start()
            
            # Warm-up time
            time.sleep(self.camera_settings.WARMUP_TIME)
            
            self.logger.info(f"Camera initialized with resolution {self.camera_settings.RESOLUTION}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_image(self, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """Capture an image from the camera.
        
        Args:
            output_path: Optional path to save the captured image.
            
        Returns:
            PIL Image object if successful, None otherwise.
        """
        if not self.camera:
            self.logger.error("Camera not initialized")
            return None
            
        try:
            # Capture image to numpy array
            array = self.camera.capture_array()
            
            # Convert to PIL Image
            if array.ndim == 3:
                image = Image.fromarray(array, 'RGB')
            else:
                image = Image.fromarray(array)
            
            if output_path:
                image.save(output_path)
                self.logger.info(f"Image saved to {output_path}")
            
            self.logger.info(f"Image captured: {image.size[0]}x{image.size[1]}")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            return None
    
    def preview_image(self, duration: float = 5.0) -> None:
        """Show camera preview for specified duration.
        
        Args:
            duration: Preview duration in seconds.
        """
        if not self.camera:
            self.logger.error("Camera not initialized")
            return
            
        try:
            self.logger.info(f"Starting preview for {duration} seconds...")
            self.camera.start_preview()
            time.sleep(duration)
            self.camera.stop_preview()
            
        except Exception as e:
            self.logger.error(f"Preview failed: {e}")
    
    def get_camera_info(self) -> dict:
        """Get camera information and current settings.
        
        Returns:
            Dictionary with camera information.
        """
        if not self.camera:
            return {"status": "not_initialized"}
            
        try:
            sensor_modes = self.camera.sensor_modes
            current_config = self.camera.camera_configuration()
            
            return {
                "status": "initialized",
                "sensor_modes": len(sensor_modes),
                "current_resolution": self.camera_settings.RESOLUTION,
                "configuration": str(current_config)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get camera info: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up camera resources."""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.logger.info("Camera cleaned up successfully")
            except Exception as e:
                self.logger.error(f"Error during camera cleanup: {e}")
            finally:
                self.camera = None


# Mock camera for testing when picamera2 is not available
class MockCameraController(CameraController):
    """Mock camera controller for testing purposes."""
    
    def initialize(self) -> bool:
        """Mock initialization."""
        self.logger.info("Mock camera initialized")
        return True
    
    def capture_image(self, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """Generate a mock test image."""
        import numpy as np
        
        # Create a simple test pattern
        width, height = self.camera_settings.RESOLUTION
        array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add some pattern
        for i in range(0, height, 100):
            array[i:i+10, :] = [255, 0, 0]  # Red lines
        
        image = Image.fromarray(array, 'RGB')
        
        if output_path:
            image.save(output_path)
        
        self.logger.info(f"Mock image generated: {image.size[0]}x{image.size[1]}")
        return image
    
    def preview_image(self, duration: float = 5.0) -> None:
        """Mock preview."""
        self.logger.info(f"Mock preview for {duration} seconds")
        time.sleep(min(duration, 1.0))  # Don't actually wait full duration
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        self.logger.info("Mock camera cleaned up")
