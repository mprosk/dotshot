#!/bin/bash
# Set up a CUPS print server that shares the ML320 (USB->Centronics on /dev/usb/lp0)
# as two network queues: ml320_gfx (ESC/P 9-pin raster) and ml320_text (plain text).
#
# Run from this directory as root: sudo ./setup-print-server.sh
set -euo pipefail

cd "$(dirname "$0")"
DEVICE=file:/dev/usb/lp0

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi

# CUPS + filters for PDF/image -> raster. Skip recommends to avoid cups-browsed
# auto-adding every other printer on the LAN.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cups cups-client cups-filters ghostscript poppler-utils avahi-daemon

# Allow the file: backend to write to /dev/usb/lp0
if ! grep -q '^FileDevice Yes' /etc/cups/cups-files.conf; then
  sed -i 's/^#\?FileDevice.*/FileDevice Yes/' /etc/cups/cups-files.conf
  grep -q '^FileDevice Yes' /etc/cups/cups-files.conf || echo 'FileDevice Yes' >> /etc/cups/cups-files.conf
fi

systemctl enable --now cups
systemctl restart cups

# Text-only driver
install -m 644 textonly.ppd /usr/share/ppd/textonly.ppd
install -m 755 textonly /usr/lib/cups/filter/textonly

lpadmin -p ml320_text -E -v "$DEVICE" -P /usr/share/ppd/textonly.ppd \
  -D "OKI Microline 320 (text only)" -L "$(hostname)" -o printer-is-shared=true
lpadmin -p ml320_gfx -E -v "$DEVICE" -m drv:///sample.drv/epson9.ppd \
  -D "OKI Microline 320 (graphics)" -L "$(hostname)" -o printer-is-shared=true
lpadmin -d ml320_gfx

# Listen on the network, allow LAN clients, advertise via DNS-SD
cupsctl --share-printers

# Open IPP + mDNS to the local subnet if ufw is on
if ufw status | grep -q '^Status: active'; then
  SUBNET=$(ip -4 route show scope link | awk '!/docker|br-|virbr/ {print $1; exit}')
  ufw allow from "$SUBNET" to any port 631 proto tcp comment 'CUPS IPP'
  ufw allow from "$SUBNET" to any port 5353 proto udp comment 'mDNS'
fi

lpstat -t
