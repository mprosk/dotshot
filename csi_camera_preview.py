#!/usr/bin/env python3
"""
CSI camera live preview at native resolution using Picamera2.

Behavior:
- Attempts to display frames using OpenCV if available.
- Falls back to Picamera2 built-in preview (QTGL or DRM) if OpenCV is not available.

Quit:
- OpenCV window: press 'q' or 'Esc'.
- Built-in preview: press Ctrl+C in the terminal.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

try:
    import cv2  # type: ignore
    _HAS_OPENCV = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    _HAS_OPENCV = False

try:
    from picamera2 import Picamera2, Preview
except Exception as exc:
    raise SystemExit(
        (
            "Picamera2 is required.\n"
            "On Raspberry Pi OS: sudo apt update && sudo apt install -y python3-picamera2\n"
            "On Ubuntu for Raspberry Pi: install libcamera + Picamera2 from Ubuntu repo if available,\n"
            "or follow Raspberry Pi Foundation instructions to build Picamera2 from source:\n"
            "https://github.com/raspberrypi/picamera2#installing\n"
            "Alternatively, use apt to install libcamera-apps and run the script with the built-in preview fallback."
        )
    ) from exc


def get_native_resolution(picam2: Picamera2) -> Tuple[int, int]:
    """Return the camera's native sensor resolution as (width, height).

    Tries `camera_properties['PixelArraySize']`, then falls back to the maximum
    size in `sensor_modes`.
    """
    # Primary: PixelArraySize from camera properties
    try:
        props = picam2.camera_properties
        pixel_array_size = props.get("PixelArraySize") if isinstance(props, dict) else None
        if pixel_array_size and len(pixel_array_size) == 2:
            width, height = int(pixel_array_size[0]), int(pixel_array_size[1])
            if width > 0 and height > 0:
                return width, height
    except Exception:
        pass

    # Fallback: choose the largest advertised sensor mode by area
    try:
        modes = getattr(picam2, "sensor_modes", None)
        if modes:
            def area(mode: dict) -> int:
                size = mode.get("size", (0, 0))
                return int(size[0]) * int(size[1])

            best = max(modes, key=area)
            size = best.get("size", (0, 0))
            width, height = int(size[0]), int(size[1])
            if width > 0 and height > 0:
                return width, height
    except Exception:
        pass

    # Last resort: a reasonable default
    return 1920, 1080


def configure_camera_for_size(picam2: Picamera2, size: Tuple[int, int]) -> None:
    """Configure the camera to output RGB frames at the requested size.

    Prefer a still configuration for full-resolution capture. Fall back to a
    preview configuration if still configuration at the desired size is not
    supported.
    """
    width, height = size

    # Prefer full-res still configuration
    try:
        config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        return
    except Exception:
        pass

    # Try a preview configuration at the requested size
    try:
        config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        return
    except Exception:
        pass

    # Fall back to default preview configuration
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))


def run_with_opencv(picam2: Picamera2) -> None:
    """Display frames using OpenCV in a window until 'q' or 'Esc' is pressed."""
    assert _HAS_OPENCV and cv2 is not None

    window_name = "CSI Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    picam2.start()
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue
            cv2.imshow(window_name, frame)
            key = int(cv2.waitKey(1)) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        picam2.stop()


def run_with_builtin_preview(picam2: Picamera2) -> None:
    """Display frames using Picamera2's built-in preview until Ctrl+C."""
    preview_backend = Preview.QTGL
    try:
        # Will raise if QT is not available, then we fall back to DRM
        picam2.start_preview(preview_backend)
    except Exception:
        preview_backend = Preview.DRM
        picam2.start_preview(preview_backend)

    picam2.start()
    try:
        print("Running built-in preview ({}). Press Ctrl+C to quit.".format(
            "QTGL" if preview_backend == Preview.QTGL else "DRM"
        ))
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            picam2.stop_preview()
        except Exception:
            pass
        picam2.stop()


def main() -> None:
    """Entry point: open CSI camera and show live preview at native resolution."""
    picam2 = Picamera2()
    native_size = get_native_resolution(picam2)
    print(f"Configuring camera for native resolution: {native_size[0]}x{native_size[1]}")
    configure_camera_for_size(picam2, native_size)

    if _HAS_OPENCV:
        print("Using OpenCV display. Press 'q' or 'Esc' to quit.")
        run_with_opencv(picam2)
    else:
        print("OpenCV not available; falling back to Picamera2 built-in preview.")
        run_with_builtin_preview(picam2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass


