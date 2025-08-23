"""Photobooth controller for automated photo capture and printing."""

import time
import logging
import threading
from datetime import datetime
from typing import Optional, Callable
from PIL import Image, ImageDraw, ImageFont

from config.settings import SETTINGS
from camera import CameraController
from processing import ProcessingPipeline, DitheringProcessor
from printer import PrinterController

# GPIO imports with fallback for non-Pi systems
try:
    import RPi.GPIO as GPIO
    from gpiozero import Button, LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    # Mock classes for development
    class Button:
        def __init__(self, pin, bounce_time=0.1):
            self.when_pressed = None
    
    class LED:
        def __init__(self, pin):
            pass
        def on(self): pass
        def off(self): pass
        def blink(self, on_time=1, off_time=1): pass


class PhotoboothController:
    """Controls photobooth operations including GPIO, camera, processing, and printing."""
    
    def __init__(self):
        self.settings = SETTINGS["photobooth"]
        self.camera_settings = SETTINGS["camera"] 
        self.logger = logging.getLogger(__name__)
        
        # Hardware components
        self.button: Optional[Button] = None
        self.status_led: Optional[LED] = None
        self.camera: Optional[CameraController] = None
        self.printer: Optional[PrinterController] = None
        
        # Processing components
        self.pipeline: Optional[ProcessingPipeline] = None
        self.dithering: Optional[DitheringProcessor] = None
        
        # State management
        self.running = False
        self.session_active = False
        self.last_error = None
        self.total_photos_taken = 0
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
    def initialize(self) -> bool:
        """Initialize all photobooth components."""
        self.logger.info("Initializing photobooth controller...")
        
        # Initialize GPIO components
        if not self._setup_gpio():
            return False
        
        # Initialize camera
        self.camera = CameraController()
        if not self.camera.initialize():
            self.logger.error("Failed to initialize camera")
            return False
        
        # Initialize printer
        if self.settings.AUTO_PRINT:
            self.printer = PrinterController()
            if not self.printer.connect():
                self.logger.warning("Failed to connect to printer (will retry later)")
        
        # Initialize processing components
        self.pipeline = ProcessingPipeline()
        self.dithering = DitheringProcessor()
        
        self.logger.info("Photobooth controller initialized successfully")
        return True
    
    def _setup_gpio(self) -> bool:
        """Setup GPIO pins for button and LED."""
        if not GPIO_AVAILABLE:
            self.logger.warning("GPIO not available - running in mock mode")
            self.button = Button(self.settings.BUTTON_PIN)
            self.status_led = LED(self.settings.LED_PIN)
            return True
        
        try:
            # Setup button with debounce
            self.button = Button(
                self.settings.BUTTON_PIN,
                bounce_time=self.settings.BUTTON_BOUNCE_TIME
            )
            self.button.when_pressed = self._on_button_press
            
            # Setup status LED
            self.status_led = LED(self.settings.LED_PIN)
            
            self.logger.info(f"GPIO setup complete - Button: GPIO{self.settings.BUTTON_PIN}, LED: GPIO{self.settings.LED_PIN}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup GPIO: {e}")
            return False
    
    def start(self) -> None:
        """Start the photobooth in continuous operation mode."""
        if self.running:
            self.logger.warning("Photobooth is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start status LED blinking to show ready state
        if self.status_led:
            self.status_led.blink(on_time=0.5, off_time=2.0)
        
        if self.settings.CONTINUOUS_MODE:
            self.main_thread = threading.Thread(target=self._continuous_operation, daemon=True)
            self.main_thread.start()
            self.logger.info("Photobooth started in continuous mode")
        else:
            self.logger.info("Photobooth started in manual mode")
        
        # Display instructions
        if self.settings.DISPLAY_INSTRUCTIONS:
            self._display_instructions()
    
    def stop(self) -> None:
        """Stop the photobooth operation."""
        self.logger.info("Stopping photobooth...")
        self.running = False
        self.shutdown_event.set()
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        if self.status_led:
            self.status_led.off()
        
        self.logger.info("Photobooth stopped")
    
    def _continuous_operation(self) -> None:
        """Main continuous operation loop."""
        self.logger.info("Starting continuous operation loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Wait for shutdown signal with timeout
                if self.shutdown_event.wait(timeout=1.0):
                    break
                
                # Perform periodic maintenance
                self._periodic_maintenance()
                
            except Exception as e:
                self.logger.error(f"Error in continuous operation: {e}")
                self.last_error = e
                
                if self.settings.RESTART_ON_CRITICAL_ERROR:
                    self.logger.info(f"Restarting in {self.settings.ERROR_RECOVERY_DELAY} seconds...")
                    time.sleep(self.settings.ERROR_RECOVERY_DELAY)
                    self._recover_from_error()
    
    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        # Check printer connection if auto-print is enabled
        if self.settings.AUTO_PRINT and self.printer and not self.printer.is_connected():
            self.logger.debug("Attempting to reconnect printer...")
            self.printer.connect()
    
    def _recover_from_error(self) -> None:
        """Attempt to recover from errors."""
        try:
            # Try to reinitialize components
            if self.camera and not self.camera.initialize():
                self.camera = CameraController()
                self.camera.initialize()
            
            if self.settings.AUTO_PRINT and self.printer and not self.printer.is_connected():
                self.printer.connect()
                
            self.last_error = None
            self.logger.info("Error recovery completed")
            
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")
    
    def _on_button_press(self) -> None:
        """Handle button press event."""
        if not self.running or self.session_active:
            return
        
        self.logger.info("Button pressed - starting photo session")
        self.session_active = True
        
        # Start photo session in separate thread
        session_thread = threading.Thread(target=self._photo_session, daemon=True)
        session_thread.start()
    
    def _photo_session(self) -> None:
        """Execute a complete photo session."""
        try:
            # Turn on steady LED to indicate session active
            if self.status_led:
                self.status_led.on()
            
            # Show live preview
            if self.settings.SHOW_PREVIEW:
                self._show_live_preview()
            
            # Countdown
            self._countdown()
            
            # Take photos
            photos = []
            for i in range(self.settings.PHOTOS_PER_SESSION):
                photo = self._capture_photo()
                if photo:
                    photos.append(photo)
                    
                    # Add delay between multiple photos
                    if i < self.settings.PHOTOS_PER_SESSION - 1:
                        time.sleep(self.settings.PHOTO_DELAY)
                else:
                    self.logger.error(f"Failed to capture photo {i+1}")
            
            if photos:
                # Process and print photos
                for i, photo in enumerate(photos):
                    processed_photo = self._process_photo(photo, session_number=self.total_photos_taken + i + 1)
                    
                    if self.settings.AUTO_PRINT and processed_photo:
                        self._print_photo(processed_photo)
                
                self.total_photos_taken += len(photos)
                self.logger.info(f"Photo session completed - {len(photos)} photos taken")
            else:
                self.logger.error("No photos captured in session")
            
        except Exception as e:
            self.logger.error(f"Error in photo session: {e}")
            self.last_error = e
        
        finally:
            self.session_active = False
            # Return LED to ready state
            if self.status_led:
                self.status_led.blink(on_time=0.5, off_time=2.0)
    
    def _show_live_preview(self) -> None:
        """Show live camera preview."""
        self.logger.info(f"Showing live preview for {self.settings.PREVIEW_DURATION}s")
        if self.camera:
            self.camera.preview_image(self.settings.PREVIEW_DURATION)
    
    def _countdown(self) -> None:
        """Execute countdown before photo capture."""
        self.logger.info(f"Starting {self.settings.COUNTDOWN_SECONDS}s countdown")
        
        for i in range(self.settings.COUNTDOWN_SECONDS, 0, -1):
            print(f"\r{i}...", end="", flush=True)
            
            # Fast LED blink during countdown
            if self.status_led:
                self.status_led.on()
                time.sleep(0.1)
                self.status_led.off()
                time.sleep(0.9)
            else:
                time.sleep(1.0)
        
        print("\rSay cheese! 📸", flush=True)
    
    def _capture_photo(self) -> Optional[Image.Image]:
        """Capture a single photo."""
        self.logger.info("Capturing photo...")
        
        if not self.camera:
            self.logger.error("Camera not initialized")
            return None
        
        # Bright LED flash during capture
        if self.status_led:
            self.status_led.on()
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dotshot_{timestamp}_{self.total_photos_taken + 1}.jpg"
            
            photo = self.camera.capture_image(filename)
            
            if photo:
                self.logger.info(f"Photo captured successfully: {filename}")
            else:
                self.logger.error("Failed to capture photo")
            
            return photo
            
        finally:
            # Turn off LED after capture
            if self.status_led:
                time.sleep(0.1)  # Brief flash
                self.status_led.off()
    
    def _process_photo(self, photo: Image.Image, session_number: int) -> Optional[Image.Image]:
        """Process photo through the image pipeline."""
        self.logger.info("Processing photo...")
        
        try:
            # Add timestamp and branding if configured
            if self.settings.ADD_TIMESTAMP or self.settings.WATERMARK_TEXT:
                photo = self._add_branding(photo, session_number)
            
            # Process through pipeline
            if self.pipeline:
                processed = self.pipeline.process_image(photo)
            else:
                processed = photo
            
            # Apply dithering
            if self.dithering:
                dithered = self.dithering.apply_dithering(processed)
                return dithered
            else:
                return processed.convert('1')  # Convert to 1-bit
                
        except Exception as e:
            self.logger.error(f"Error processing photo: {e}")
            return None
    
    def _add_branding(self, photo: Image.Image, session_number: int) -> Image.Image:
        """Add timestamp, watermark, and other branding to photo."""
        # Create a copy to avoid modifying original
        branded_photo = photo.copy()
        draw = ImageDraw.Draw(branded_photo)
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add timestamp
        if self.settings.ADD_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"{self.settings.BOOTH_NAME} - {timestamp} - #{session_number:04d}"
            
            # Position at bottom of image
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (photo.width - text_width) // 2
            y = photo.height - text_height - 10
            
            # Add background rectangle for better readability
            draw.rectangle(
                [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
                fill='white',
                outline='black'
            )
            draw.text((x, y), text, fill='black', font=font)
        
        # Add custom watermark
        if self.settings.WATERMARK_TEXT:
            watermark_bbox = draw.textbbox((0, 0), self.settings.WATERMARK_TEXT, font=font)
            watermark_width = watermark_bbox[2] - watermark_bbox[0]
            
            x = (photo.width - watermark_width) // 2
            y = 20
            
            draw.text((x, y), self.settings.WATERMARK_TEXT, fill='white', font=font)
        
        return branded_photo
    
    def _print_photo(self, photo: Image.Image) -> bool:
        """Print the processed photo."""
        self.logger.info("Printing photo...")
        
        if not self.printer:
            self.logger.warning("Printer not configured")
            return False
        
        # Try to connect if not connected
        if not self.printer.is_connected():
            if not self.printer.connect():
                self.logger.error("Failed to connect to printer")
                return False
        
        # Attempt to print with retries
        for attempt in range(self.settings.MAX_RETRY_ATTEMPTS):
            try:
                if self.printer.print_image(photo):
                    self.logger.info("Photo printed successfully")
                    return True
                else:
                    self.logger.warning(f"Print attempt {attempt + 1} failed")
                    
            except Exception as e:
                self.logger.error(f"Print attempt {attempt + 1} error: {e}")
            
            if attempt < self.settings.MAX_RETRY_ATTEMPTS - 1:
                time.sleep(1.0)  # Brief delay between retries
        
        self.logger.error("All print attempts failed")
        return False
    
    def _display_instructions(self) -> None:
        """Display instructions for users."""
        instructions = f"""
📸 Welcome to {self.settings.BOOTH_NAME}! 📸

Instructions:
1. Press the button to start
2. Look at the camera during preview
3. Get ready during countdown
4. Smile for the photo!
5. Your photo will print automatically

Total photos taken today: {self.total_photos_taken}
        
Press Ctrl+C to exit
"""
        print(instructions)
    
    def get_status(self) -> dict:
        """Get current photobooth status."""
        return {
            "running": self.running,
            "session_active": self.session_active,
            "total_photos_taken": self.total_photos_taken,
            "camera_ready": self.camera is not None and hasattr(self.camera, 'camera'),
            "printer_connected": self.printer is not None and self.printer.is_connected(),
            "last_error": str(self.last_error) if self.last_error else None,
            "gpio_available": GPIO_AVAILABLE,
            "settings": {
                "auto_print": self.settings.AUTO_PRINT,
                "photos_per_session": self.settings.PHOTOS_PER_SESSION,
                "countdown_seconds": self.settings.COUNTDOWN_SECONDS
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up photobooth resources...")
        
        self.stop()
        
        if self.camera:
            self.camera.cleanup()
        
        if self.printer and self.printer.is_connected():
            self.printer.disconnect()
        
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except:
                pass
        
        self.logger.info("Photobooth cleanup completed")
