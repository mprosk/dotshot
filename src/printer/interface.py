"""Printer interface for OKI Microline 320 Turbo via USB-parallel adapter."""

import os
import time
import logging
from typing import Optional, List, Union
import numpy as np
from PIL import Image

from config.settings import SETTINGS
from .escp_commands import ESCPCommands


class PrinterController:
    """Controls the OKI Microline 320 Turbo dot matrix printer."""
    
    def __init__(self):
        self.settings = SETTINGS["printer"]
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.device_path = None
        
    def connect(self, device_path: str = None) -> bool:
        """Connect to the printer.
        
        Args:
            device_path: Optional specific device path to use
            
        Returns:
            True if connection successful, False otherwise
        """
        if device_path:
            paths_to_try = [device_path]
        else:
            paths_to_try = [self.settings.DEVICE_PATH] + self.settings.BACKUP_DEVICE_PATHS
        
        for path in paths_to_try:
            if self._try_connect_to_device(path):
                self.device_path = path
                self.logger.info(f"Connected to printer at {path}")
                return True
        
        self.logger.error("Failed to connect to printer on any device path")
        return False
    
    def _try_connect_to_device(self, device_path: str) -> bool:
        """Try to connect to a specific device path.
        
        Args:
            device_path: Device path to try
            
        Returns:
            True if successful
        """
        try:
            if not os.path.exists(device_path):
                return False
                
            # Try to open the device for writing
            self.device = open(device_path, 'wb', buffering=0)
            
            # Send a simple command to test communication
            test_cmd = ESCPCommands.printer_status()
            self.device.write(test_cmd)
            self.device.flush()
            
            return True
            
        except (OSError, IOError) as e:
            self.logger.debug(f"Failed to connect to {device_path}: {e}")
            if self.device:
                try:
                    self.device.close()
                except:
                    pass
                self.device = None
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the printer."""
        if self.device:
            try:
                self.device.close()
                self.logger.info("Disconnected from printer")
            except Exception as e:
                self.logger.error(f"Error disconnecting from printer: {e}")
            finally:
                self.device = None
                self.device_path = None
    
    def is_connected(self) -> bool:
        """Check if printer is connected."""
        return self.device is not None
    
    def send_command(self, command: bytes) -> bool:
        """Send a command to the printer.
        
        Args:
            command: Raw command bytes to send
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            self.logger.error("Printer not connected")
            return False
        
        try:
            self.device.write(command)
            self.device.flush()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            return False
    
    def initialize_printer(self) -> bool:
        """Initialize the printer with default settings."""
        if not self.is_connected():
            return False
        
        init_commands = ESCPCommands.setup_page_for_image(
            left_margin=int(self.settings.LEFT_MARGIN * 10),  # Convert inches to columns (approx)
            top_margin=int(self.settings.TOP_MARGIN * 6)      # Convert inches to lines at 6 LPI
        )
        
        return self.send_command(init_commands)
    
    def print_text(self, text: str, **formatting) -> bool:
        """Print text with optional formatting.
        
        Args:
            text: Text to print
            **formatting: Formatting options (bold, underline, etc.)
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            return False
        
        bold = formatting.get('bold', False)
        underline = formatting.get('underline', False)
        font_size = formatting.get('font_size', '12')
        
        commands = []
        
        # Set font size
        commands.append(ESCPCommands.set_font_size(font_size))
        
        # Print the text with formatting
        text_cmd = ESCPCommands.print_text_line(text, bold=bold, underline=underline)
        commands.append(text_cmd)
        
        full_command = b''.join(commands)
        return self.send_command(full_command)
    
    def print_image(self, image: Image.Image, 
                   caption: str = None,
                   optimize_for_printer: bool = True) -> bool:
        """Print a 1-bit dithered image.
        
        Args:
            image: 1-bit PIL Image (black and white)
            caption: Optional caption text to print below image
            optimize_for_printer: Whether to apply printer-specific optimizations
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            self.logger.error("Printer not connected")
            return False
        
        # Ensure image is 1-bit
        if image.mode != '1':
            if image.mode == 'L':
                # Convert grayscale to 1-bit using threshold
                image = image.point(lambda p: p > 128 and 255)
                image = image.convert('1')
            else:
                self.logger.error("Image must be 1-bit or grayscale")
                return False
        
        # Initialize printer
        if not self.initialize_printer():
            return False
        
        # Convert image to bitmap data
        bitmap_data = self._image_to_bitmap_data(image)
        
        if not bitmap_data:
            return False
        
        # Print the bitmap
        success = self._print_bitmap_data(bitmap_data, image.width, image.height)
        
        if success and caption:
            # Add some spacing before caption
            self.send_command(ESCPCommands.LF * 2)
            success = self.print_text(caption, bold=True)
        
        # End the print job
        if success:
            success = self.send_command(ESCPCommands.end_print_job())
        
        return success
    
    def _image_to_bitmap_data(self, image: Image.Image) -> Optional[List[bytes]]:
        """Convert 1-bit PIL Image to printer bitmap data.
        
        Args:
            image: 1-bit PIL Image
            
        Returns:
            List of bytes for each image row, or None if error
        """
        try:
            # Convert image to numpy array
            img_array = np.array(image)
            height, width = img_array.shape
            
            # Convert to printer format (1 bit per pixel, packed into bytes)
            bitmap_lines = []
            
            for y in range(height):
                row_data = []
                
                # Process pixels in groups of 8 (1 byte)
                for x in range(0, width, 8):
                    byte_value = 0
                    
                    # Pack 8 pixels into one byte (MSB first)
                    for bit in range(8):
                        if x + bit < width:
                            # 0 = black pixel (print), 1 = white pixel (no print)
                            # PIL stores True=white, False=black, so we need to invert
                            if not img_array[y, x + bit]:  # Black pixel
                                byte_value |= (1 << (7 - bit))
                    
                    row_data.append(byte_value)
                
                bitmap_lines.append(bytes(row_data))
            
            self.logger.debug(f"Converted image to {len(bitmap_lines)} lines of bitmap data")
            return bitmap_lines
            
        except Exception as e:
            self.logger.error(f"Failed to convert image to bitmap data: {e}")
            return None
    
    def _print_bitmap_data(self, bitmap_lines: List[bytes], 
                          width: int, height: int) -> bool:
        """Print bitmap data line by line.
        
        Args:
            bitmap_lines: List of bitmap data bytes for each line
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            True if successful
        """
        try:
            density = self.settings.PRINT_DENSITY
            
            for line_num, line_data in enumerate(bitmap_lines):
                # Print bitmap line
                cmd = ESCPCommands.print_bitmap_line(line_data, width, density)
                
                if not self.send_command(cmd):
                    self.logger.error(f"Failed to print bitmap line {line_num}")
                    return False
                
                # Small delay to avoid overwhelming the printer
                if line_num % 10 == 0:  # Every 10 lines
                    time.sleep(0.01)  # 10ms delay
            
            self.logger.info(f"Successfully printed {height} lines of bitmap data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to print bitmap data: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test the printer connection with a simple print.
        
        Returns:
            True if test successful
        """
        if not self.is_connected():
            if not self.connect():
                return False
        
        # Print a simple test pattern
        test_commands = []
        test_commands.append(ESCPCommands.initialize_printer())
        test_commands.append(ESCPCommands.print_text_line("PrinterPhoto Test Page", bold=True))
        test_commands.append(ESCPCommands.LF)
        test_commands.append(ESCPCommands.print_text_line(f"Device: {self.device_path}"))
        test_commands.append(ESCPCommands.print_text_line(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"))
        test_commands.append(ESCPCommands.LF)
        
        # Test some basic graphics
        test_bitmap = self._create_test_pattern()
        if test_bitmap:
            test_commands.extend(test_bitmap)
        
        test_commands.append(ESCPCommands.end_print_job())
        
        full_command = b''.join(test_commands)
        success = self.send_command(full_command)
        
        if success:
            self.logger.info("Printer test successful")
        else:
            self.logger.error("Printer test failed")
        
        return success
    
    def _create_test_pattern(self) -> List[bytes]:
        """Create a simple test pattern for printer testing.
        
        Returns:
            List of commands to print test pattern
        """
        commands = []
        
        # Create a simple checkerboard pattern
        pattern_width = 40  # pixels
        pattern_height = 8   # lines
        
        for line in range(pattern_height):
            # Create alternating pattern
            line_data = []
            for byte_pos in range(pattern_width // 8):
                if (line + byte_pos) % 2 == 0:
                    line_data.append(0xAA)  # 10101010
                else:
                    line_data.append(0x55)  # 01010101
            
            cmd = ESCPCommands.print_bitmap_line(bytes(line_data), pattern_width, "normal")
            commands.append(cmd)
        
        return commands
    
    def get_printer_info(self) -> dict:
        """Get printer information and status.
        
        Returns:
            Dictionary with printer information
        """
        return {
            "connected": self.is_connected(),
            "device_path": self.device_path,
            "settings": {
                "chars_per_line": self.settings.CHARS_PER_LINE,
                "lines_per_inch": self.settings.LINES_PER_INCH,
                "print_density": self.settings.PRINT_DENSITY,
                "margins": {
                    "left": self.settings.LEFT_MARGIN,
                    "top": self.settings.TOP_MARGIN,
                    "right": self.settings.RIGHT_MARGIN,
                    "bottom": self.settings.BOTTOM_MARGIN
                }
            }
        }
