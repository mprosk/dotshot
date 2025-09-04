import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np

from dotshot.camera import USBCamera
from dotshot.printer import Printer


class ImagePipeline:
    """Minimal pipeline for live grayscale preview and printing."""

    def __init__(
        self,
        *,
        cam: Optional[USBCamera],
        printer: Printer,
    ) -> None:
        """Initialize devices and image buffers.

        Parameters:
            cam: Optional camera for live capture; set to None for file mode.
            printer: Printer used to send image files.
        """
        self.cam: Optional[USBCamera] = cam
        self.printer: Printer = printer

        # Image buffers
        self.raw: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.orig: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.frame: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

    def capture(self) -> None:
        """Capture a fresh frame from the camera for live preview."""
        if self.cam is None:
            logging.debug("capture() called without a camera; ignoring")
            return
        gray = self.cam.capture_frame()
        self.raw = gray
        self.orig = gray
        self.frame = gray

    def load_file(self, path: str) -> None:
        """Load a grayscale image from file as the raw frame."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image from {path}")
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        self.raw = img
        self.orig = img
        self.frame = img
        logging.info("Loaded file %s with shape %s", path, tuple(img.shape))

    def get_display(self) -> np.ndarray:
        """Return the current preview image (grayscale)."""
        return self.frame

    def get_print_frame(self) -> np.ndarray:
        """Return the current printable grayscale frame."""
        return self.frame

    def get_raw_frame(self) -> np.ndarray:
        """Return the latest captured raw grayscale frame from the camera."""
        return self.raw

    def print_current(self) -> None:
        """Write the current frame to a temp PNG and send it to the printer."""
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
