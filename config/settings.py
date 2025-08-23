"""Configuration settings for PrinterPhoto project."""

from typing import Dict, Any, Tuple
import os


class CameraSettings:
    """Camera configuration settings."""
    # Camera resolution for capture
    RESOLUTION: Tuple[int, int] = (2592, 1944)  # Pi Camera v2 max resolution
    
    # Camera parameters
    EXPOSURE_MODE: str = "auto"
    AWB_MODE: str = "auto"
    ISO: int = 100
    
    # Capture settings
    WARMUP_TIME: float = 2.0  # seconds
    CAPTURE_TIMEOUT: int = 10  # seconds


class ProcessingSettings:
    """Image processing pipeline settings."""
    
    # Target dimensions for printer (in pixels, at printer resolution)
    PRINTER_WIDTH_PIXELS: int = 576  # ~8 inches at 72 DPI
    PRINTER_HEIGHT_PIXELS: int = 720  # ~10 inches at 72 DPI
    
    # Resize settings
    RESIZE_ALGORITHM: str = "LANCZOS"  # PIL resize algorithm
    MAINTAIN_ASPECT_RATIO: bool = True
    
    # Crop settings
    CROP_TO_CENTER: bool = True
    CROP_MARGIN: float = 0.1  # 10% margin
    
    # Adjustment settings
    BRIGHTNESS_ADJUST: float = 1.2    # 1.0 = no change
    CONTRAST_ADJUST: float = 1.3      # 1.0 = no change
    GAMMA_ADJUST: float = 0.9         # 1.0 = no change
    
    # Edge enhancement settings
    EDGE_ENHANCE_STRENGTH: float = 2.0  # Edge enhancement factor
    UNSHARP_MASK_RADIUS: float = 1.0
    UNSHARP_MASK_PERCENT: int = 150
    UNSHARP_MASK_THRESHOLD: int = 3
    
    # Dithering settings
    DITHER_ALGORITHM: str = "floyd_steinberg"  # or "ordered", "threshold"
    DITHER_THRESHOLD: int = 128  # For threshold dithering


class PrinterSettings:
    """Printer configuration settings."""
    
    # Device settings
    DEVICE_PATH: str = "/dev/usb/lp0"  # USB parallel adapter path
    BACKUP_DEVICE_PATHS: list = ["/dev/lp0", "/dev/usb/lp1"]
    
    # Printer specifications (OKI Microline 320 Turbo)
    CHARS_PER_LINE: int = 80
    LINES_PER_INCH: int = 6
    DOTS_PER_INCH_HORIZONTAL: int = 72
    DOTS_PER_INCH_VERTICAL: int = 72
    
    # Print settings
    PRINT_DENSITY: str = "normal"  # "draft", "normal", "high"
    LINE_SPACING: float = 1.0
    
    # Margins (in inches)
    LEFT_MARGIN: float = 0.5
    TOP_MARGIN: float = 0.5
    RIGHT_MARGIN: float = 0.5
    BOTTOM_MARGIN: float = 0.5
    
    # Text settings
    FONT_SIZE: int = 12
    TEXT_LINE_HEIGHT: float = 1.2


class SystemSettings:
    """System and debugging settings."""
    
    # File paths
    OUTPUT_DIR: str = "output"
    TEMP_DIR: str = "temp"
    TEST_IMAGES_DIR: str = "tests/test_images"
    
    # Debugging
    DEBUG_MODE: bool = False
    SAVE_INTERMEDIATE_IMAGES: bool = False
    DISPLAY_PROCESSING_TIME: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "dotshot.log"


class PhotoboothSettings:
    """Photobooth-specific configuration settings."""
    
    # GPIO Settings (Raspberry Pi)
    BUTTON_PIN: int = 18  # GPIO pin for capture button
    LED_PIN: int = 24     # GPIO pin for status LED
    BUTTON_BOUNCE_TIME: float = 0.3  # Button debounce time in seconds
    
    # Photobooth Operation
    COUNTDOWN_SECONDS: int = 5        # Countdown before capture
    PREVIEW_DURATION: float = 3.0     # Live preview duration
    AUTO_PRINT: bool = True           # Automatically print after capture
    CONTINUOUS_MODE: bool = True      # Run continuously
    
    # Photo Session Settings  
    PHOTOS_PER_SESSION: int = 1       # Number of photos per button press
    PHOTO_DELAY: float = 2.0          # Delay between multiple photos
    SESSION_TIMEOUT: float = 30.0     # Max time for a photo session
    
    # Display and Feedback
    SHOW_PREVIEW: bool = True         # Show live preview
    PLAY_SOUNDS: bool = False         # Play sound effects (requires audio)
    DISPLAY_INSTRUCTIONS: bool = True # Show instructions on screen
    
    # Photo Booth Branding
    BOOTH_NAME: str = "DotShot Booth"
    WATERMARK_TEXT: str = ""          # Optional watermark text
    ADD_TIMESTAMP: bool = True        # Add timestamp to photos
    
    # Error Handling
    MAX_RETRY_ATTEMPTS: int = 3       # Max retries for failed operations
    ERROR_RECOVERY_DELAY: float = 5.0 # Delay before retrying after error
    RESTART_ON_CRITICAL_ERROR: bool = True  # Auto-restart on critical errors


# Environment-specific overrides
def load_environment_settings() -> Dict[str, Any]:
    """Load settings from environment variables."""
    env_settings = {}
    
    # Camera settings
    if os.getenv("CAMERA_RESOLUTION"):
        resolution = os.getenv("CAMERA_RESOLUTION").split("x")
        env_settings["CAMERA_RESOLUTION"] = (int(resolution[0]), int(resolution[1]))
    
    # Printer device path
    if os.getenv("PRINTER_DEVICE"):
        env_settings["PRINTER_DEVICE"] = os.getenv("PRINTER_DEVICE")
    
    # Debug mode
    if os.getenv("DEBUG_MODE"):
        env_settings["DEBUG_MODE"] = os.getenv("DEBUG_MODE").lower() == "true"
    
    return env_settings


# Global settings instance
SETTINGS = {
    "camera": CameraSettings(),
    "processing": ProcessingSettings(), 
    "printer": PrinterSettings(),
    "system": SystemSettings(),
    "photobooth": PhotoboothSettings(),
    "env": load_environment_settings()
}
