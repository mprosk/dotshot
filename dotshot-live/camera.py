#!/usr/bin/env python3
"""
Typed USB webcam capture utility using OpenCV (V4L2 on Linux).

Usage example:
    from dotshot.camera import USBCamera

    with USBCamera(device="/dev/video0", width=1280, height=720, fps=30) as cam:
        frame = cam.capture_frame()
        # frame is a NumPy array: grayscale uint8 (H, W)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "OpenCV is required. Install with: sudo apt install -y python3-opencv"
    ) from exc

import numpy as np

# Camera config
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960
CAMERA_FPS = 30
WARMUP_FRAMES = 6


DeviceArg = Union[int, str]


def _device_to_index(device: DeviceArg) -> int:
    """Convert a device argument to an OpenCV index.

    Accepts numeric indices (e.g. 0) or strings like "/dev/video2".
    """
    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    if device.startswith("/dev/video"):
        try:
            return int(device.replace("/dev/video", ""))
        except ValueError:
            return 0
    return 0


class USBCamera:
    """Minimal, typed USB camera wrapper for single-frame capture.

    Frames are returned as grayscale (uint8) arrays of shape (H, W).
    """

    def __init__(
        self,
        device: DeviceArg = 0,
        *,
        width: Optional[int] = CAMERA_WIDTH,
        height: Optional[int] = CAMERA_HEIGHT,
        fps: Optional[int] = CAMERA_FPS,
        fourcc: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._requested_device: DeviceArg = device
        self._requested_width: Optional[int] = width
        self._requested_height: Optional[int] = height
        self._requested_fps: Optional[int] = fps
        self._requested_fourcc: Optional[str] = fourcc
        self._logger: logging.Logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )

        self._capture: Optional[cv2.VideoCapture] = None
        self._index: Optional[int] = None

    def open(self) -> None:
        """Open the video device and apply requested properties if provided."""
        if self._capture is not None:
            return

        index = _device_to_index(self._requested_device)
        self._logger.info(
            "Opening video device %s (index %d)", self._requested_device, index
        )
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            self._logger.debug("CAP_V4L2 failed, falling back to default backend")
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Failed to open video device {self._requested_device} (index {index})."
            )

        if self._requested_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._requested_width))
        if self._requested_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._requested_height))
        if self._requested_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self._requested_fps))
        if self._requested_fourcc and len(self._requested_fourcc) == 4:
            fourcc_code = cv2.VideoWriter_fourcc(*self._requested_fourcc)
            capture.set(cv2.CAP_PROP_FOURCC, float(fourcc_code))

        self._capture = capture
        self._index = index

        capture.read()
        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._logger.info(
            "Opened %s at %dx%d", self._requested_device, actual_w, actual_h
        )

    def close(self) -> None:
        """Release the device if open."""
        if self._capture is not None:
            try:
                self._logger.info("Releasing video device %s", self._requested_device)
                self._capture.release()
            finally:
                self._capture = None

    def is_open(self) -> bool:
        """Return whether the device is currently open."""
        return self._capture is not None and bool(self._capture.isOpened())

    def get_actual_size(self) -> Tuple[int, int]:
        """Return the current device output size (width, height)."""
        if not self.is_open():
            raise RuntimeError("Camera is not open.")
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._logger.debug("Current device size is %dx%d", width, height)
        return width, height

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame and return it as uint8 array.

        Returns:
            np.ndarray: grayscale uint8 array (H, W)
        """
        if not self.is_open():
            self.open()
        assert self._capture is not None

        # Read frame from camera
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            self._logger.error("Failed to read frame from camera")
            raise RuntimeError("Failed to read frame from camera.")
        self._logger.debug("Captured frame with shape %s", tuple(frame_bgr.shape))

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Log min and max values for debugging
        min_val = np.min(frame_gray)
        max_val = np.max(frame_gray)
        self._logger.info("Frame grayscale range: min=%d, max=%d", min_val, max_val)

        # Normalize to 0-255 range
        frame_gray = cv2.normalize(
            frame_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        # If the frame is 16:9, center-crop to 4:3
        h, w = int(frame_gray.shape[0]), int(frame_gray.shape[1])
        if w * 9 == h * 16:
            # 16:9 -> crop to 4:3
            if w * 3 > h * 4:
                # Too wide: crop width
                target_w = max(1, (h * 4) // 3)
                x0 = (w - target_w) // 2
                x1 = x0 + target_w
                y0, y1 = 0, h
            else:
                # Too tall: crop height (unlikely for 16:9, but safe guard)
                target_h = max(1, (w * 3) // 4)
                y0 = (h - target_h) // 2
                y1 = y0 + target_h
                x0, x1 = 0, w
            self._logger.debug(
                "Cropping 16:9 to 4:3: (%dx%d) -> (%dx%d)", w, h, x1 - x0, y1 - y0
            )
            frame_gray = frame_gray[y0:y1, x0:x1]

        # Ensure contiguous array
        if not frame_gray.flags["C_CONTIGUOUS"]:
            frame_gray = np.ascontiguousarray(frame_gray)
        return frame_gray
