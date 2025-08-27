import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np

from dotshot.camera import USBCamera
from dotshot.printer import Printer

# Display/processing config
RESOLUTIONS = [
    (160, 120),
    (320, 240),
    (640, 480),
    (1024, 768),
    (1280, 960),
]

QUANT_LEVELS = [2, 4, 8, 16, 32, 64, 128, 256]


class ImagePipeline:
    """Encapsulates capture, resize, quantization, and display state."""

    def __init__(
        self,
        *,
        cam: Optional[USBCamera],
        printer: Printer,
    ) -> None:
        """Initialize the pipeline and preallocate image buffers.

        Parameters:
            cam: Optional camera for live capture; set to None for file mode.
            printer: Printer used to send image files.
        """
        # Devices
        self.cam: Optional[USBCamera] = cam
        self.printer: Printer = printer

        # State
        self.res_index: int = 0
        self.level_offset: int = 0
        self.quant_index: int = 0

        # Image buffers
        self.raw: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.orig: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.quant: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.frame: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

    def capture(self) -> None:
        """Capture a fresh frame from the camera and rebuild the processed state."""
        if self.cam is None:
            logging.debug("capture() called without a camera; ignoring")
            return
        self.raw = self.cam.capture_frame()
        self._rebuild_from_raw()

    def load_file(self, path: str) -> None:
        """Load a grayscale image from file as the raw frame and rebuild state."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image from {path}")
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        self.raw = img
        logging.info("Loaded file %s with shape %s", path, tuple(img.shape))
        self._rebuild_from_raw()

    def cycle_resolution(self) -> None:
        """Cycle to the next target resolution and rebuild derived images."""
        prev = self.res_index
        self.res_index = (self.res_index + 1) % len(RESOLUTIONS)
        logging.debug("Resolution index changed: %d -> %d", prev, self.res_index)
        self._rebuild_from_raw()

    def adjust_offset(self, delta_levels: int) -> None:
        """Shift brightness by integer quantization steps and update the frame."""
        prev = self.level_offset
        self.level_offset += int(delta_levels)
        logging.debug("Offset adjusted: %d -> %d (delta %+d)", prev, self.level_offset, delta_levels)
        self.frame = self._shift_quant_levels(self.quant, self.level_offset)

    def set_offset(self, new_offset: int) -> None:
        """Set absolute brightness offset in quantization steps and update the frame."""
        prev = self.level_offset
        self.level_offset = int(new_offset)
        if self.level_offset != prev:
            logging.debug("Offset set: %d -> %d", prev, self.level_offset)
        self.frame = self._shift_quant_levels(self.quant, self.level_offset)

    def set_resolution_index(self, index: int) -> None:
        """Set target resolution by index and rebuild derived images."""
        if not RESOLUTIONS:
            return
        clamped = max(0, min(int(index), len(RESOLUTIONS) - 1))
        if clamped == self.res_index:
            return
        prev = self.res_index
        self.res_index = clamped
        logging.debug("Resolution index set: %d -> %d", prev, self.res_index)
        self._rebuild_from_raw()

    def set_quant_index(self, index: int) -> None:
        """Set quantization level index and rebuild derived images."""
        if not QUANT_LEVELS:
            return
        clamped = max(0, min(int(index), len(QUANT_LEVELS) - 1))
        if clamped == self.quant_index:
            return
        prev = self.quant_index
        self.quant_index = clamped
        logging.debug("Quantization index set: %d -> %d (levels %d)", prev, self.quant_index, QUANT_LEVELS[self.quant_index])
        self._rebuild_from_raw()

    def adjust_quant(self, delta: int) -> None:
        """Adjust quantization level index by delta (wraps across available levels)."""
        if not QUANT_LEVELS:
            return
        prev = self.quant_index
        self.quant_index = (self.quant_index + int(delta)) % len(QUANT_LEVELS)
        if self.quant_index != prev:
            logging.debug(
                "Quantization index adjusted: %d -> %d (levels %d)",
                prev,
                self.quant_index,
                QUANT_LEVELS[self.quant_index],
            )
            self._rebuild_from_raw()

    def current_levels(self) -> int:
        """Return the current number of quantization levels."""
        return int(QUANT_LEVELS[self.quant_index]) if QUANT_LEVELS else 256

    def get_display(self) -> np.ndarray:
        """Return a grayscale display image without any overlay (clean preview)."""
        if self.frame.ndim == 2:
            return self.frame
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def get_print_frame(self) -> np.ndarray:
        """Return the current printable grayscale frame."""
        return self.frame

    def get_raw_frame(self) -> np.ndarray:
        """Return the latest captured raw grayscale frame from the camera."""
        return self.raw

    def print_current(self) -> None:
        """Write the current frame to a temp PNG and send it to the printer."""
        # Write current printable frame to a temp file and submit to printer
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            logging.info(f"Writing {self.frame.shape} frame to {tmp_path}")
            cv2.imwrite(tmp_path, self.frame)
        try:
            logging.info(f"Printing {self.frame.shape} frame from {tmp_path}")
            self.printer.print_image_file(tmp_path)
            logging.info(f"File {tmp_path} sent to print queue")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _current_target(self) -> tuple[int, int]:
        """Return current target (height, width) derived from the resolution index."""
        w, h = RESOLUTIONS[self.res_index]
        return h, w

    def _rebuild_from_raw(self) -> None:
        """Recompute `orig`, `quant`, and `frame` arrays from the latest `raw`."""
        tgt_h, tgt_w = self._current_target()
        self.orig = self._resize_fit(self.raw, tgt_h, tgt_w)
        self.quant = self._quantize_gray(self.orig)
        self.frame = self._shift_quant_levels(self.quant, self.level_offset)

    @staticmethod
    def _resize_fit(image: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
        """Resize a grayscale image to fit within (max_h, max_w) keeping aspect ratio."""
        h, w = int(image.shape[0]), int(image.shape[1])
        if max_h <= 0 or max_w <= 0:
            return image
        scale_h = float(max_h) / float(h)
        scale_w = float(max_w) / float(w)
        scale = min(scale_h, scale_w)
        if scale <= 0:
            return image
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    def _quantize_gray(self, image: np.ndarray) -> np.ndarray:
        """Quantize a grayscale image to `self.levels` evenly spaced values in [0, 255]."""
        levels = QUANT_LEVELS[self.quant_index] if QUANT_LEVELS else 256
        if levels >= 256:
            return image if image.flags["C_CONTIGUOUS"] else np.ascontiguousarray(image)
        f32 = image.astype(np.float32)
        indices = np.rint(f32 * (levels - 1) / 255.0)
        quantized = np.rint(indices * (255.0 / (levels - 1))).astype(np.uint8)
        if not quantized.flags["C_CONTIGUOUS"]:
            quantized = np.ascontiguousarray(quantized)
        return quantized

    def _shift_quant_levels(self, image: np.ndarray, delta_levels: int) -> np.ndarray:
        """Shift brightness by `delta_levels` quant steps; saturate to [0, 255]."""
        levels = QUANT_LEVELS[self.quant_index] if QUANT_LEVELS else 256
        step = 255.0 / float(max(1, levels - 1))
        indices = np.rint(image.astype(np.float32) / step) + float(delta_levels)
        indices = np.clip(indices, 0.0, float(levels - 1))
        shifted = np.rint(indices * step).astype(np.uint8)
        if not shifted.flags["C_CONTIGUOUS"]:
            shifted = np.ascontiguousarray(shifted)
        return shifted
