import logging
import os
import tempfile

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
DEFAULT_LEVELS = 4

UP_KEYS = {82, 2490368}  # Up arrow (Linux/Windows)
DOWN_KEYS = {84, 2621440}  # Down arrow (Linux/Windows)


class ImagePipeline:
    """Encapsulates capture, resize, quantization, and display state."""

    @staticmethod
    def _resize_fit(image: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
        """Resize a grayscale image to fit within (max_h, max_w) maintaining aspect."""
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
        """Quantize normalized grayscale image to evenly spaced levels across 0..255."""
        levels = self.levels
        if levels >= 256:
            return image if image.flags["C_CONTIGUOUS"] else np.ascontiguousarray(image)
        f32 = image.astype(np.float32)
        indices = np.rint(f32 * (levels - 1) / 255.0)
        quantized = np.rint(indices * (255.0 / (levels - 1))).astype(np.uint8)
        if not quantized.flags["C_CONTIGUOUS"]:
            quantized = np.ascontiguousarray(quantized)
        return quantized

    def _shift_quant_levels(self, image: np.ndarray, delta_levels: int) -> np.ndarray:
        """Shift image up/down by integer quantization steps (saturating to 0..255)."""
        levels = self.levels
        step = 255.0 / float(levels - 1)
        indices = np.rint(image.astype(np.float32) / step) + float(delta_levels)
        indices = np.clip(indices, 0.0, float(levels - 1))
        shifted = np.rint(indices * step).astype(np.uint8)
        if not shifted.flags["C_CONTIGUOUS"]:
            shifted = np.ascontiguousarray(shifted)
        return shifted

    def __init__(
        self,
        *,
        cam: USBCamera,
        printer: Printer,
        resolutions: list[tuple[int, int]],
        initial_index: int,
        levels: int,
    ) -> None:
        self.cam: USBCamera = cam
        self.printer: Printer = printer
        self.resolutions: list[tuple[int, int]] = resolutions
        self.res_index: int = max(0, min(initial_index, len(resolutions) - 1))
        self.levels: int = max(2, int(levels))
        self.level_offset: int = 0

        self.raw: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.orig: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.quant: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.frame: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

    def _current_target(self) -> tuple[int, int]:
        w, h = self.resolutions[self.res_index]
        return h, w

    def _rebuild_from_raw(self) -> None:
        tgt_h, tgt_w = self._current_target()
        self.orig = self._resize_fit(self.raw, tgt_h, tgt_w)
        self.quant = self._quantize_gray(self.orig)
        self.frame = self._shift_quant_levels(self.quant, self.level_offset)

    def capture(self) -> None:
        self.raw = self.cam.capture_frame()
        self.level_offset = 0
        self._rebuild_from_raw()

    def recapture(self) -> None:
        self.capture()

    def cycle_resolution(self) -> None:
        self.res_index = (self.res_index + 1) % len(self.resolutions)
        self._rebuild_from_raw()

    def adjust_offset(self, delta_levels: int) -> None:
        self.level_offset += int(delta_levels)
        self.frame = self._shift_quant_levels(self.quant, self.level_offset)

    def get_display(self) -> np.ndarray:
        # Build a display-only copy with HUD overlay; do not affect printable content
        if self.frame.ndim == 2:
            display = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        else:
            display = self.frame.copy()

        # Scaled HUD overlay (consistent apparent size across resolutions)
        ref_w, ref_h = 640.0, 480.0
        cur_h, cur_w = float(self.orig.shape[0]), float(self.orig.shape[1])
        scale_factor = max(0.25, min(4.0, min(cur_w / ref_w, cur_h / ref_h)))
        font_scale = 0.6 * scale_factor
        thickness = max(1, int(round(2 * scale_factor)))
        margin = int(round(10 * scale_factor))

        lines = [
            f"Levels: {self.levels}  Offset: {self.level_offset:+d}",
            f"Res: {self.orig.shape[1]}x{self.orig.shape[0]}",
        ]
        y = margin + int(round(14 * scale_factor))
        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.putText(
                display,
                line,
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            y += text_h + baseline + int(round(6 * scale_factor))
        return display

    def get_print_frame(self) -> np.ndarray:
        return self.frame

    def print_current(self) -> None:
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


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cam = USBCamera()
    printer = Printer()

    # Choose starting resolution index (use second option if present)
    start_index = 1 if len(RESOLUTIONS) > 1 else 0
    pipeline = ImagePipeline(
        cam=cam,
        printer=printer,
        resolutions=RESOLUTIONS,
        initial_index=start_index,
        levels=DEFAULT_LEVELS,
    )

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        with cam:
            pipeline.capture()
            while True:
                display = pipeline.get_display()
                cv2.imshow("DotShot", display)
                key_full = int(cv2.waitKey(0))
                key = key_full & 0xFFFFFFFF

                if key in (27, ord("q")):
                    break

                if (key & 0xFF) == ord("c"):
                    pipeline.recapture()
                    continue

                if key in UP_KEYS:
                    pipeline.adjust_offset(+1)
                    continue

                if key in DOWN_KEYS:
                    pipeline.adjust_offset(-1)
                    continue

                if (key & 0xFF) == ord("r"):
                    pipeline.cycle_resolution()
                    continue

                if (key & 0xFF) == ord("p"):
                    pipeline.print_current()
                    continue
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
