#!/bin/bash

# Install the printer drivers
sudo cp ./textonly.pdd /usr/share/ppd/
sudo cp ./textonly /usr/lib/cups/filter/
sudo chmod +x /usr/lib/cups/filter/textonly

# Add textonly printer queue pointing to the lp0 port
sudo lpadmin -p ml320_text -E -v file:/dev/usb/lp0 -P /usr/share/ppd/textonly.ppd

# Add graphics mode printer queue, also on the lp0 port
sudo lpadmin -p ml320_gfx -E -v file:/dev/usb/lp0 -m drv:///sample.drv/epson9.ppd
