"""ESC/P control codes and commands for dot matrix printers."""

from typing import List, Union


class ESCPCommands:
    """ESC/P control codes for OKI Microline 320 Turbo and compatible printers."""
    
    # Control characters
    ESC = b'\x1b'  # Escape character
    CR = b'\x0d'   # Carriage return
    LF = b'\x0a'   # Line feed
    FF = b'\x0c'   # Form feed (page break)
    CAN = b'\x18'  # Cancel
    
    # Basic printer control
    @staticmethod
    def initialize_printer() -> bytes:
        """Initialize/reset the printer."""
        return ESCPCommands.ESC + b'@'
    
    @staticmethod
    def line_feed() -> bytes:
        """Single line feed."""
        return ESCPCommands.LF
    
    @staticmethod
    def carriage_return() -> bytes:
        """Carriage return."""
        return ESCPCommands.CR
    
    @staticmethod
    def form_feed() -> bytes:
        """Form feed (eject page)."""
        return ESCPCommands.FF
    
    @staticmethod
    def set_line_spacing(n: int) -> bytes:
        """Set line spacing to n/216 inch.
        
        Args:
            n: Line spacing (0-255, default is 36 for 6 LPI)
        """
        return ESCPCommands.ESC + b'3' + bytes([n])
    
    @staticmethod
    def set_line_spacing_standard(lpi: int) -> bytes:
        """Set standard line spacing.
        
        Args:
            lpi: Lines per inch (6 or 8)
        """
        if lpi == 6:
            return ESCPCommands.ESC + b'2'  # 6 LPI
        elif lpi == 8:
            return ESCPCommands.ESC + b'0'  # 8 LPI
        else:
            raise ValueError("Only 6 or 8 LPI supported")
    
    # Font and character control
    @staticmethod
    def set_font_size(size: str) -> bytes:
        """Set font size.
        
        Args:
            size: "10", "12", "17" for 10 cpi, 12 cpi, or 17 cpi
        """
        if size == "10":
            return ESCPCommands.ESC + b'P'  # 10 CPI
        elif size == "12":
            return ESCPCommands.ESC + b'M'  # 12 CPI 
        elif size == "17":
            return ESCPCommands.ESC + b'g'  # 17 CPI
        else:
            raise ValueError("Supported sizes: 10, 12, 17")
    
    @staticmethod
    def set_bold(enable: bool = True) -> bytes:
        """Enable or disable bold text."""
        if enable:
            return ESCPCommands.ESC + b'E'  # Bold on
        else:
            return ESCPCommands.ESC + b'F'  # Bold off
    
    @staticmethod
    def set_underline(enable: bool = True) -> bytes:
        """Enable or disable underline."""
        if enable:
            return ESCPCommands.ESC + b'-1'  # Underline on
        else:
            return ESCPCommands.ESC + b'-0'  # Underline off
    
    @staticmethod
    def set_italic(enable: bool = True) -> bytes:
        """Enable or disable italic text."""
        if enable:
            return ESCPCommands.ESC + b'4'  # Italic on
        else:
            return ESCPCommands.ESC + b'5'  # Italic off
    
    # Positioning commands
    @staticmethod
    def set_left_margin(columns: int) -> bytes:
        """Set left margin.
        
        Args:
            columns: Number of columns from left edge
        """
        return ESCPCommands.ESC + b'l' + bytes([columns])
    
    @staticmethod
    def set_right_margin(columns: int) -> bytes:
        """Set right margin.
        
        Args:
            columns: Number of columns from left edge
        """
        return ESCPCommands.ESC + b'Q' + bytes([columns])
    
    @staticmethod
    def set_top_margin(lines: int) -> bytes:
        """Set top margin.
        
        Args:
            lines: Number of lines from top
        """
        return ESCPCommands.ESC + b'N' + bytes([lines])
    
    @staticmethod
    def set_bottom_margin(lines: int) -> bytes:
        """Set bottom margin.
        
        Args:
            lines: Number of lines from top for bottom margin
        """
        return ESCPCommands.ESC + b'O' + bytes([lines])
    
    @staticmethod
    def horizontal_tab(positions: List[int]) -> bytes:
        """Set horizontal tab positions.
        
        Args:
            positions: List of column positions for tabs
        """
        cmd = ESCPCommands.ESC + b'D'
        for pos in positions:
            cmd += bytes([pos])
        cmd += b'\x00'  # Null terminator
        return cmd
    
    @staticmethod
    def vertical_tab(positions: List[int]) -> bytes:
        """Set vertical tab positions.
        
        Args:
            positions: List of line positions for tabs
        """
        cmd = ESCPCommands.ESC + b'B'
        for pos in positions:
            cmd += bytes([pos])
        cmd += b'\x00'  # Null terminator
        return cmd
    
    @staticmethod
    def absolute_horizontal_position(column: int) -> bytes:
        """Move to absolute horizontal position.
        
        Args:
            column: Column position (0-based)
        """
        return ESCPCommands.ESC + b'$' + bytes([column % 256, column // 256])
    
    @staticmethod
    def relative_horizontal_position(columns: int) -> bytes:
        """Move relative horizontal position.
        
        Args:
            columns: Number of columns to move (can be negative)
        """
        if columns >= 0:
            return ESCPCommands.ESC + b'\\' + bytes([columns % 256, columns // 256])
        else:
            # Handle negative movement
            abs_columns = abs(columns)
            return ESCPCommands.ESC + b'\\' + bytes([(256 - abs_columns) % 256, 255])
    
    # Graphics and bitmap printing
    @staticmethod
    def start_bitmap_mode(density: str = "normal") -> bytes:
        """Start bitmap graphics mode.
        
        Args:
            density: Graphics density ("draft", "normal", "high")
        """
        if density == "draft":
            return ESCPCommands.ESC + b'K'  # 60 DPI
        elif density == "normal":
            return ESCPCommands.ESC + b'L'  # 120 DPI
        elif density == "high":
            return ESCPCommands.ESC + b'Y'  # 180 DPI
        else:
            return ESCPCommands.ESC + b'L'  # Default to normal
    
    @staticmethod
    def bitmap_data(width: int, data: bytes) -> bytes:
        """Send bitmap data.
        
        Args:
            width: Width in pixels
            data: Bitmap data bytes
        """
        # ESC/P bitmap format: ESC mode width_low width_high data
        width_low = width % 256
        width_high = width // 256
        return bytes([width_low, width_high]) + data
    
    @staticmethod
    def print_bitmap_line(data: bytes, width: int, density: str = "normal") -> bytes:
        """Print a single line of bitmap data.
        
        Args:
            data: Bitmap data for one line
            width: Width in pixels
            density: Print density
        """
        mode_cmd = ESCPCommands.start_bitmap_mode(density)
        bitmap_cmd = ESCPCommands.bitmap_data(width, data)
        return mode_cmd + bitmap_cmd + ESCPCommands.CR + ESCPCommands.LF
    
    # Advanced commands
    @staticmethod
    def set_print_density(density: str) -> bytes:
        """Set print density.
        
        Args:
            density: "draft", "normal", or "high"
        """
        if density == "draft":
            return ESCPCommands.ESC + b'x0'  # Draft quality
        elif density == "normal":
            return ESCPCommands.ESC + b'x1'  # Normal quality  
        elif density == "high":
            return ESCPCommands.ESC + b'x2'  # High quality
        else:
            return ESCPCommands.ESC + b'x1'  # Default to normal
    
    @staticmethod
    def set_character_width(cpi: int) -> bytes:
        """Set character width in characters per inch.
        
        Args:
            cpi: Characters per inch
        """
        if cpi == 10:
            return ESCPCommands.ESC + b'P'
        elif cpi == 12:
            return ESCPCommands.ESC + b'M'
        elif cpi == 17:
            return ESCPCommands.ESC + b'g'
        else:
            # Use proportional spacing command for other values
            spacing = int(1440 / cpi)  # Convert CPI to 1/1440 inch units
            return ESCPCommands.ESC + b' ' + bytes([spacing])
    
    @staticmethod
    def cancel_line() -> bytes:
        """Cancel current line buffer."""
        return ESCPCommands.CAN
    
    @staticmethod
    def printer_status() -> bytes:
        """Request printer status."""
        return ESCPCommands.ESC + b'?'
    
    # Utility methods for common sequences
    @staticmethod
    def setup_page_for_image(left_margin: int = 5, top_margin: int = 3) -> bytes:
        """Setup page formatting for image printing.
        
        Args:
            left_margin: Left margin in columns
            top_margin: Top margin in lines
        """
        commands = []
        commands.append(ESCPCommands.initialize_printer())
        commands.append(ESCPCommands.set_line_spacing_standard(6))  # 6 LPI
        commands.append(ESCPCommands.set_left_margin(left_margin))
        commands.append(ESCPCommands.set_top_margin(top_margin))
        commands.append(ESCPCommands.set_print_density("normal"))
        
        return b''.join(commands)
    
    @staticmethod
    def print_text_line(text: str, bold: bool = False, underline: bool = False) -> bytes:
        """Print a line of text with optional formatting.
        
        Args:
            text: Text to print
            bold: Whether to make text bold
            underline: Whether to underline text
        """
        commands = []
        
        if bold:
            commands.append(ESCPCommands.set_bold(True))
        if underline:
            commands.append(ESCPCommands.set_underline(True))
        
        commands.append(text.encode('ascii', errors='ignore'))
        
        if bold:
            commands.append(ESCPCommands.set_bold(False))
        if underline:
            commands.append(ESCPCommands.set_underline(False))
        
        commands.append(ESCPCommands.CR)
        commands.append(ESCPCommands.LF)
        
        return b''.join(commands)
    
    @staticmethod
    def end_print_job() -> bytes:
        """End print job and eject page."""
        return ESCPCommands.FF
