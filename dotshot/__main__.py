import logging
import cv2
import tempfile
import os
import numpy as np

from dotshot.camera import USBCamera
from dotshot.printer import Printer

# Display/processing config
RESIZE_HEIGHT = 576
RESIZE_WIDTH = 756
LEVELS = 8  # number of gray levels used after normalization

UP_KEYS = {82, 2490368}    # Up arrow (Linux/Windows)
DOWN_KEYS = {84, 2621440}  # Down arrow (Linux/Windows)


def resize_fit(image: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Resize a grayscale image to fit within (max_h, max_w) maintaining aspect.

    Args:
        image: Grayscale uint8 image (H, W)
        max_h: Max output height
        max_w: Max output width

    Returns:
        Resized grayscale uint8 image
    """
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


def quantize_gray(image: np.ndarray, levels: int) -> np.ndarray:
    """Quantize normalized grayscale image to evenly spaced levels across 0..255.

    Args:
        image: Grayscale uint8 image (H, W), expected normalized 0..255
        levels: Number of distinct output levels (>=2)

    Returns:
        Quantized grayscale uint8 image (H, W)
    """
    levels = max(2, int(levels))
    if levels == 256:
        return image
    f32 = image.astype(np.float32)
    indices = np.rint(f32 * (levels - 1) / 255.0)
    quantized = np.rint(indices * (255.0 / (levels - 1))).astype(np.uint8)
    if not quantized.flags["C_CONTIGUOUS"]:
        quantized = np.ascontiguousarray(quantized)
    return quantized


def shift_quant_levels(image: np.ndarray, levels: int, delta_levels: int) -> np.ndarray:
    """Shift image up/down by integer quantization steps (saturating to 0..255).

    Args:
        image: Grayscale uint8 image (H, W), already normalized 0..255
        levels: Number of quantization levels
        delta_levels: Positive to brighten, negative to darken

    Returns:
        Grayscale uint8 image shifted by delta_levels in quantized space
    """
    levels = max(2, int(levels))
    step = 255.0 / float(levels - 1)
    # Work in index space to avoid accumulating rounding errors
    indices = np.rint(image.astype(np.float32) / step) + float(delta_levels)
    indices = np.clip(indices, 0.0, float(levels - 1))
    shifted = np.rint(indices * step).astype(np.uint8)
    if not shifted.flags["C_CONTIGUOUS"]:
        shifted = np.ascontiguousarray(shifted)
    return shifted


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cam = USBCamera()
    printer = Printer()

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        with cam:
            # Capture, resize (keep aspect), then quantize
            orig = cam.capture_frame()
            orig = resize_fit(orig, RESIZE_HEIGHT, RESIZE_WIDTH)
            quant = quantize_gray(orig, LEVELS)
            level_offset = 0

            frame = quant  # displayed frame (quant + offset)
            while True:
                # Build display-only copy with overlay text (not used for printing)
                if frame.ndim == 2:
                    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display = frame.copy()
                overlay_text = f"Levels: {LEVELS}  Offset: {level_offset:+d}"
                cv2.putText(
                    display,
                    overlay_text,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("DotShot", display)
                key = int(cv2.waitKey(0)) & 0xFFFFFFFF

                if key in (27, ord("q")):
                    break

                if key == ord("c"):
                    # New capture resets offset
                    orig = cam.capture_frame()
                    orig = resize_fit(orig, RESIZE_HEIGHT, RESIZE_WIDTH)
                    quant = quantize_gray(orig, LEVELS)
                    level_offset = 0
                    frame = quant
                    continue

                if key in UP_KEYS:
                    level_offset += 1
                    frame = shift_quant_levels(quant, LEVELS, level_offset)
                    continue

                if key in DOWN_KEYS:
                    level_offset -= 1
                    frame = shift_quant_levels(quant, LEVELS, level_offset)
                    continue

                if key == ord("p"):
                    # Write current display frame to temporary file and print
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        logging.info(f"Writing {frame.shape} frame to {tmp_path}")
                        cv2.imwrite(tmp_path, frame)
                    try:
                        logging.info(f"Printing {frame.shape} frame from {tmp_path}")
                        printer.print_image_file(tmp_path)
                        logging.info(f"File {tmp_path} sent to print queue")
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

