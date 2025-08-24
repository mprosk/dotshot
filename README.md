# DotShot

```
██████╗  ██████╗ ████████╗███████╗██╗  ██╗ ██████╗ ████████╗
██╔══██╗██╔═══██╗╚══██╔══╝██╔════╝██║  ██║██╔═══██╗╚══██╔══╝
██║  ██║██║   ██║   ██║   ███████╗███████║██║   ██║   ██║   
██║  ██║██║   ██║   ██║   ╚════██║██╔══██║██║   ██║   ██║   
██████╔╝╚██████╔╝   ██║   ███████║██║  ██║╚██████╔╝   ██║   
╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   
```

DotShot is a Raspberry Pi–based photobooth that captures and prints photos on a dot‑matrix printer.

## Hardware

- Raspberry Pi 4 (recommended) or 3B+
- Raspberry Pi Camera Module (v2 or HQ)
- Epson ESC/P compatible dot matrix printer (I am using an OKI Microline 320 Turbo)
- USB‑to‑Centronics parallel adapter

## Printer

```bash
# Install CUPS and useful drivers
sudo apt update && sudo apt install -y cups

# Let your user administer printers (log out/in after this once)
sudo usermod -aG lpadmin "$USER"

# Enable file devices in CUPS
sudo nano /etc/cups/cups-files.conf
# Add or set exactly this line (save and exit):
FileDevice Yes

# Start and enable CUPS
sudo systemctl enable --now cups

# Add textmode drivers
sudo cp ./textonly.pdd /usr/share/ppd/
sudo cp ./textonly /usr/lib/cups/filter/
sudo chmod +x /usr/lib/cups/filter/textonly

# Add textonly printer queue pointing to the lp0 port
sudo lpadmin -p ml320_text -E -v file:/dev/usb/lp0 -P /usr/share/ppd/textonly.ppd

# Add graphics mode printer queue, also on the lp0 port
sudo lpadmin -p ml320_gfx -E -v file:/dev/usb/lp0 -m drv:///sample.drv/epson9.ppd

# add raw mode printer queue
sudo lpadmin -p ml320_raw -E -v file:/dev/usb/lp0 -m raw

lp -d ml320_text test/hello.txt
lp -d ml320_gfx test/tux.png

lpstat -t
```

## Installation

Use Raspberry Pi OS Lite

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv python3-picamera2

git clone <repo-url>
cd dotshot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
