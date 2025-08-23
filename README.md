# DotShot

```
██████╗  ██████╗ ████████╗███████╗██╗  ██╗ ██████╗ ████████╗
██╔══██╗██╔═══██╗╚══██╔══╝██╔════╝██║  ██║██╔═══██╗╚══██╔══╝
██║  ██║██║   ██║   ██║   ███████╗███████║██║   ██║   ██║   
██║  ██║██║   ██║   ██║   ╚════██║██╔══██║██║   ██║   ██║   
██████╔╝╚██████╔╝   ██║   ███████║██║  ██║╚██████╔╝   ██║   
╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   
```

DotShot is a Raspberry Pi–based photobooth that captures, processes, and prints photos on a dot‑matrix printer. It provides a modular image processing pipeline, GPIO‑driven operation, and a command‑line interface for capture, processing, and testing.

## Features

- Automated capture with GPIO button and LED status indicators
- Live preview and countdown
- Modular processing pipeline: resize, crop, tone/contrast adjustment, edge enhancement
- Multiple dithering algorithms (Floyd–Steinberg, Atkinson, Sierra, Ordered, Threshold)
- Printer integration with ESC/P commands and caption support
- Mock hardware mode for development
- Configurable via `config/settings.py`

## Hardware

- Raspberry Pi 4 (recommended) or 3B+
- Raspberry Pi Camera Module (v2 or HQ)
- Dot matrix printer (Epson ESC/P compatible)
- USB‑to‑Centronics parallel adapter
- Momentary push button (GPIO) and LED indicator

## Installation

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libcamera-apps python3-libcamera python3-opencv git

git clone <repo-url>
cd dotshot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Enable the camera using `raspi-config` (Interface Options → Camera).

## Quick start

All commands are executed from the project root. The CLI entry point is `src/main.py`.

```bash
# Show available commands
python src/main.py -h

# System information and configuration
python src/main.py info

# Live camera preview (mock camera optional)
python src/main.py preview --preview-time 5 --mock-camera

# Photobooth mode (continuous)
python src/main.py photobooth

# Manual trigger mode for development
python src/main.py photobooth --manual-trigger

# Capture, process, display stages, and print (omit --preview-only to print)
python src/main.py capture --output capture.jpg --display-stages --preview-only

# Process an existing image and compare dithering
python src/main.py process-file input.jpg --display-stages --compare-dithering -o processed.png

# Print a prepared image with optional caption
python src/main.py print-image processed.png --text "Hello"

# Test printer connectivity
python src/main.py test-printer
```

## GPIO wiring (reference)

| Component | GPIO Pin | Physical Pin |
|-----------|----------|--------------|
| Button    | GPIO 18  | 12           |
| LED       | GPIO 24  | 18           |
| Ground    | GND      | 6 or 14      |

LED example: GPIO 24 → 220Ω resistor → LED(+) → LED(−) → GND.

## Configuration

Adjust settings in `config/settings.py`. Typical options include:

- Photobooth: button pin, LED pin, countdown seconds, photos per session, auto‑print
- Camera: resolution, exposure, ISO, AWB
- Processing: printer target size, resize algorithm, brightness/contrast, edge enhance strength, dither algorithm
- Printer: device path, print density, margins

## Processing pipeline

1. Resize to printer target resolution
2. Smart crop with aspect‑ratio preservation
3. Brightness/contrast/gamma adjustments
4. Edge enhancement for dot‑matrix clarity
5. Dithering to 1‑bit using configurable algorithm

## Running as a service (optional)

Create `/etc/systemd/system/dotshot.service`:

```ini
[Unit]
Description=DotShot Photobooth
After=multi-user.target

[Service]
Type=simple
Restart=always
ExecStart=/home/pi/dotshot/venv/bin/python /home/pi/dotshot/src/main.py photobooth
WorkingDirectory=/home/pi/dotshot
User=pi
Environment=PYTHONPATH=/home/pi/dotshot/src

[Install]
WantedBy=multi-user.target
```

Then enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable dotshot.service
sudo systemctl start dotshot.service
```

## Troubleshooting

- Camera: enable via `raspi-config`; validate with `libcamera-hello --preview`
- Printer device path: verify with `lsusb` and `dmesg`; try `/dev/usb/lp0`
- Permissions: `sudo chmod 666 /dev/usb/lp0` (adjust to your environment)
- GPIO not available on non‑Pi systems: use mock modes and `--manual-trigger`

## Tests

```bash
pytest tests/
python src/main.py test-pipeline --generate-images
```

## Project structure

```
dotshot/
├── config/
├── src/
│   ├── main.py               # CLI entry point
│   ├── photobooth/
│   ├── camera/
│   ├── processing/
│   ├── printer/
│   └── utils/
└── tests/
```

## License

MIT License. See `LICENSE`.


