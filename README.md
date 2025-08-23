# DotShot 📸

```
██████╗  ██████╗ ████████╗███████╗██╗  ██╗ ██████╗ ████████╗
██╔══██╗██╔═══██╗╚══██╔══╝██╔════╝██║  ██║██╔═══██╗╚══██╔══╝
██║  ██║██║   ██║   ██║   ███████╗███████║██║   ██║   ██║   
██║  ██║██║   ██║   ██║   ╚════██║██╔══██║██║   ██║   ██║   
██████╔╝╚██████╔╝   ██║   ███████║██║  ██║╚██████╔╝   ██║   
╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   
```
**Automated Photo Booth System for Raspberry Pi**

A complete **photobooth system** that captures photos, processes them for optimal dot matrix printing, and automatically prints them. Perfect for events, parties, or permanent installations!

## 🎉 Photobooth Features

### **🤖 Automated Operation**
- **Push-button activation** - Hardware button trigger via GPIO
- **LED status indicators** - Shows ready/busy/error states
- **Countdown timer** with visual feedback
- **Live camera preview** before capture
- **Automatic printing** of photos
- **Continuous operation** for events

### **📸 Professional Photo Processing**
- **Smart image pipeline**: Resize, crop, adjust brightness/contrast
- **Edge enhancement** optimized for dot matrix printing  
- **Multiple dithering algorithms** (Floyd-Steinberg, Atkinson, etc.)
- **Print preview simulation** - see exactly how it will look
- **Custom branding** - Add timestamps, watermarks, booth name

### **🔧 Hardware Integration**
- **Raspberry Pi Camera** (CSI connector)
- **GPIO button and LED** control
- **OKI Microline 320 Turbo** dot matrix printer via USB-parallel adapter
- **Robust error handling** for reliable operation

### **⚙️ Flexible Configuration**
- **Multiple operation modes**: Photobooth, manual capture, file processing
- **Configurable settings**: Countdown time, photos per session, auto-print
- **Testing tools**: Mock hardware mode for development
- **Comprehensive logging** and status monitoring

## 🛠️ Hardware Requirements

### **Core Components**
- **Raspberry Pi 4** (recommended) or Pi 3B+
- **Raspberry Pi Camera Module** (v2 or HQ camera)
- **OKI Microline 320 Turbo** dot matrix printer
- **USB to Centronix parallel adapter cable**

### **Photobooth Hardware**
- **Push button** (momentary, normally open)
- **LED indicator** (any color, current-limiting resistor recommended)
- **Jumper wires** for GPIO connections
- **Enclosure/mounting** for permanent installation

### **Optional Additions**
- **External display** for preview (HDMI)
- **Speakers** for sound effects
- **Power supply** suitable for continuous operation
- **Camera tripod mount** or custom mounting hardware

## 🚀 Installation & Setup

### **1. Prepare Raspberry Pi**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install python3-pip python3-venv libcamera-apps python3-libcamera python3-opencv git

# Enable camera
sudo raspi-config  # Navigate to Interface Options > Camera > Enable
```

### **2. Clone and Install**
```bash
# Clone repository
git clone <your-repo-url>
cd dotshot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Hardware Wiring**

#### **GPIO Connections**
```
┌─────────────┬──────────────┬─────────────────┐
│ Component   │ GPIO Pin     │ Physical Pin    │
├─────────────┼──────────────┼─────────────────┤
│ Button      │ GPIO 18      │ Pin 12          │
│ LED         │ GPIO 24      │ Pin 18          │
│ Ground      │ Ground       │ Pin 6 or 14     │
└─────────────┴──────────────┴─────────────────┘
```

#### **Button Wiring**
```
Button ──┐
         ├── GPIO 18 (Pin 12)
         └── Ground (Pin 6)
```

#### **LED Wiring** 
```
GPIO 24 (Pin 18) ──[220Ω resistor]── LED(+) ──LED(-)── Ground (Pin 14)
```

### **4. Test Installation**
```bash
# Test system components
python run_dotshot.py info

# Test camera (mock mode on non-Pi systems)
python run_dotshot.py preview --mock-camera

# Test printer connection
python run_dotshot.py test-printer
```

## ⚙️ Configuration

### **Quick Setup**
Most settings work out of the box! Edit `config/settings.py` to customize:

```python
class PhotoboothSettings:
    # GPIO pins (change if needed)
    BUTTON_PIN: int = 18
    LED_PIN: int = 24
    
    # Photobooth behavior
    COUNTDOWN_SECONDS: int = 5        # Countdown before photo
    AUTO_PRINT: bool = True           # Print automatically
    PHOTOS_PER_SESSION: int = 1       # Photos per button press
    
    # Branding
    BOOTH_NAME: str = "DotShot Booth"
    ADD_TIMESTAMP: bool = True
    WATERMARK_TEXT: str = ""          # Custom watermark
```

### **Advanced Configuration**
- **Camera settings**: Resolution, exposure, ISO
- **Processing pipeline**: Brightness, contrast, edge enhancement
- **Printer settings**: Device path, margins, print density
- **Error handling**: Retry attempts, recovery delays

## 📱 Usage

### **🎪 Photobooth Mode (Main Usage)**
```bash
# Start the photobooth (runs continuously)
python run_dotshot.py photobooth

# Manual testing mode (press Enter instead of button)
python run_dotshot.py photobooth --manual-trigger
```

**How it works:**
1. **Press the button** to start a photo session
2. **Live preview** shows for 3 seconds  
3. **Countdown** from 5 to 1
4. **Photo captured** with LED flash
5. **Automatic processing** and printing
6. **Ready for next photo** (LED blinks)

### **🔧 Development & Testing**
```bash
# Check system status
python run_dotshot.py info

# Test individual components
python run_dotshot.py preview --mock-camera
python run_dotshot.py test-printer
python run_dotshot.py test-pipeline --generate-images

# Process images from files
python run_dotshot.py process-file photo.jpg --display-stages --compare-dithering

# Manual photo capture
python run_dotshot.py capture --text "Test Photo" --display-stages
```

## 🎨 Photo Processing Pipeline

The system automatically optimizes photos for dot matrix printing:

1. **📏 Resize**: Scale to printer dimensions (576x720 pixels)
2. **✂️ Crop**: Smart cropping with aspect ratio preservation  
3. **🎨 Adjust**: Auto-enhance brightness, contrast, gamma
4. **⚡ Edge Enhancement**: Sharpen details for dot matrix clarity
5. **🖼️ Dithering**: Convert to 1-bit with multiple algorithms
6. **🏷️ Branding**: Add timestamps, watermarks, booth name

### **Dithering Algorithms Available**
- **Floyd-Steinberg** (default) - Best overall quality
- **Atkinson** - Apple-style dithering with lighter appearance  
- **Sierra** - Good detail preservation
- **Ordered (Bayer)** - Classic checkerboard pattern
- **Threshold** - Simple black/white conversion

## 🏗️ Project Structure

```
dotshot/
├── run_dotshot.py              # 🚀 Main entry point
├── config/
│   └── settings.py            # ⚙️ All configuration settings
├── src/
│   ├── main.py                # 🎛️ CLI commands
│   ├── photobooth/            # 🎪 Photobooth controller
│   │   └── controller.py      #    Automated operation & GPIO
│   ├── camera/                # 📷 Camera interface
│   │   └── capture.py         #    Pi Camera + mock support
│   ├── processing/            # 🖼️ Image processing pipeline
│   │   ├── pipeline.py        #    Pipeline coordinator
│   │   ├── dithering.py       #    Multiple dithering algorithms
│   │   └── stages/            #    Individual processing stages
│   ├── printer/               # 🖨️ Printer communication
│   │   ├── interface.py       #    OKI Microline 320 Turbo
│   │   └── escp_commands.py   #    Complete ESC/P command set
│   └── utils/                 # 🛠️ Utilities
│       ├── display.py         #    Image visualization
│       └── testing.py         #    Testing & validation
└── tests/                     # 🧪 Test suite
    └── test_images/           #    Generated test images
```

## 🎬 Running as a Service (Optional)

For permanent installations, run the photobooth as a system service:

### **1. Create Service File**
```bash
sudo nano /etc/systemd/system/dotshot.service
```

```ini
[Unit]
Description=DotShot Photo Booth
After=multi-user.target

[Service]
Type=simple
Restart=always
ExecStart=/home/pi/dotshot/venv/bin/python /home/pi/dotshot/run_dotshot.py photobooth
WorkingDirectory=/home/pi/dotshot
User=pi
Environment=PYTHONPATH=/home/pi/dotshot/src

[Install]  
WantedBy=multi-user.target
```

### **2. Enable Service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable dotshot.service
sudo systemctl start dotshot.service

# Check status
sudo systemctl status dotshot.service
```

## 🔧 Troubleshooting

### **📷 Camera Issues**
```bash
# Enable camera
sudo raspi-config  # Interface Options > Camera

# Test camera
libcamera-hello --preview

# Check connections
vcgencmd get_camera
```

### **🖨️ Printer Issues**
```bash
# Check USB connection
lsusb
dmesg | grep usb

# Test printer manually
echo "Test" > /dev/usb/lp0

# Check permissions
sudo chmod 666 /dev/usb/lp0
```

### **🔘 GPIO Issues**
```bash
# Check GPIO state
gpio readall

# Test LED manually
echo "24" > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio24/direction  
echo "1" > /sys/class/gpio/gpio24/value
```

### **🐛 Software Issues**
```bash
# Check status
python run_dotshot.py info

# View logs
tail -f dotshot.log

# Test with mock hardware
python run_dotshot.py photobooth --manual-trigger
```

## 🛠️ Development

### **Running Tests**
```bash
# Run test suite
pytest tests/

# Test with generated images
python run_dotshot.py test-pipeline --generate-images

# Test individual components
python run_dotshot.py test-printer
```

### **Adding Custom Features**

#### **New Processing Stages**
1. Create stage in `src/processing/stages/`
2. Implement the stage interface  
3. Add to pipeline in `pipeline.py`
4. Update `settings.py` for configuration

#### **Custom Dithering Algorithms**
1. Add method to `DitheringProcessor` class
2. Update `compare_algorithms()` method
3. Test with various image types

#### **Hardware Extensions**
- Additional GPIO pins for more buttons/LEDs
- LCD display integration
- Sound effects via GPIO or USB audio
- Multiple printer support

### **Code Architecture**
- **Modular design** - Easy to extend and modify
- **Hardware abstraction** - Mock mode for development
- **Error resilience** - Handles hardware failures gracefully
- **Configurable** - All settings in one place
- **Well documented** - Type hints and comprehensive docstrings

## 🎉 Usage Ideas

- **Birthday parties** - Instant photo memories
- **Weddings** - Guest photo station
- **Corporate events** - Branded photo booth
- **Art installations** - Interactive photography
- **School events** - Fun photo activity
- **Maker fairs** - Demonstrate Pi capabilities
- **Retro photography** - Vintage dot matrix aesthetic

## 📄 License

MIT License - see LICENSE file for details.

---

```
   📸 DotShot - Where memories meet pixels! 📸
    Perfect for events, parties, and pure fun!
```

**🎪 Built with ❤️ for makers, event organizers, and photography enthusiasts!**

*Questions? Issues? Contributions welcome! Open an issue or pull request.*
