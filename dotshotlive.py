#!/usr/bin/env python3
"""DotShot Live photobooth application.

Provides a live camera preview with a photobooth flow:
- Space: start 3-2-1 countdown overlay, white flash, capture a frame
- Space again: return to live view
- 'p' and 's': print or save only when a frame is captured
- 'f': toggle fullscreen; 'q'/Esc: quit

This module uses OpenCV for camera I/O and window management.
"""
import argparse
import logging
import os
import tempfile
import time
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from dotshot.camera import USBCamera
from dotshot.printer import Printer
from dotshot.utils import CROP_MODES, crop_center_to_aspect


class FPSCounter:
    """Lightweight frames-per-second calculator with periodic updates."""

    def __init__(self, *, enabled: bool, update_period_s: float = 1.0) -> None:
        self._enabled: bool = enabled
        self._update_period_s: float = update_period_s
        self._last_time: float = time.perf_counter()
        self._frame_count: int = 0
        self._fps: float = 0.0

    def update(self) -> None:
        """Record a frame and update FPS if the update period elapsed."""
        if not self._enabled:
            return
        self._frame_count += 1
        now: float = time.perf_counter()
        elapsed: float = now - self._last_time
        if elapsed >= self._update_period_s:
            self._fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            self._frame_count = 0
            self._last_time = now

    @property
    def fps(self) -> float:
        """Last measured frames-per-second value."""
        return self._fps

    def text(self) -> str:
        """Short text representation of the current FPS."""
        return "N/A" if not self._enabled else f"{self._fps:.1f}"


class PhotoboothState(Enum):
    """Enumeration of photobooth UI states."""

    LIVE = auto()
    COUNTDOWN = auto()
    FLASH = auto()
    CAPTURED = auto()


class DotShotLiveApp:
    """Main application class encapsulating the photobooth UI loop and actions."""

    def __init__(self, *, cam: USBCamera, printer: Printer) -> None:
        """Initialize the app with a camera and printer.

        Args:
            cam: OpenCV-backed camera wrapper.
            printer: Printer interface used for printing captured frames.
        """
        self.cam: USBCamera = cam
        self.printer: Printer = printer
        self.window_name: str = "DotShot"
        self.fps_counter: FPSCounter = FPSCounter(enabled=True, update_period_s=1.0)
        self.state: PhotoboothState = PhotoboothState.LIVE
        self.countdown_start: float = 0.0
        self.countdown_total: int = 3
        self.flash_start: float = 0.0
        self.flash_duration_s: float = 0.15
        self.captured_frame: Optional[np.ndarray] = None
        self.captured_frame_original: Optional[np.ndarray] = None
        self.fullscreen: bool = False
        self.edge_enabled: bool = False
        self.sobel_threshold: int = 24
        self.crop_index: int = 0  # 0: 4:3, 1:1

    def run(self) -> None:
        """Run the main event loop until the user quits."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            self.cam.open()
            current_frame: np.ndarray = self.cam.capture_frame()
            while True:
                display_frame: np.ndarray = self._update_state_and_get_display(
                    current_frame
                )
                cv2.imshow(self.window_name, display_frame)
                self._update_window_title(display_frame)

                key_full: int = int(cv2.waitKey(1))
                if key_full != -1:
                    logging.debug(f"Key: {key_full}")
                key: int = key_full & 0xFFFFFFFF
                if not self._handle_key(key):
                    break

                if self.state in (PhotoboothState.LIVE, PhotoboothState.COUNTDOWN):
                    current_frame = display_frame
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                self.cam.close()
            except Exception as e:
                logging.error("Error closing camera: %s", e)

    def _update_state_and_get_display(self, last_frame: np.ndarray) -> np.ndarray:
        """Advance state as needed and return the frame to display.

        Args:
            last_frame: The last frame displayed; used during flash to size the white screen.

        Returns:
            The image to present in the window for the current iteration.
        """
        if self.state == PhotoboothState.LIVE:
            frame: np.ndarray = self.cam.capture_frame()
            # Apply center crop to selected aspect before further processing
            aspect_w, aspect_h = CROP_MODES[self.crop_index]
            frame = crop_center_to_aspect(frame, aspect_w, aspect_h)
            if self.edge_enabled:
                frame = self._apply_edge_filter(frame)
            self.fps_counter.update()
            return frame
        if self.state == PhotoboothState.COUNTDOWN:
            frame = self.cam.capture_frame()
            aspect_w, aspect_h = CROP_MODES[self.crop_index]
            frame = crop_center_to_aspect(frame, aspect_w, aspect_h)
            if self.edge_enabled:
                frame = self._apply_edge_filter(frame)
            elapsed: float = time.perf_counter() - self.countdown_start
            remaining: int = self.countdown_total - int(elapsed)
            if remaining <= 0:
                self._start_flash()
                return np.full_like(frame, 255)
            return self._draw_center_text(frame, str(remaining))
        if self.state == PhotoboothState.FLASH:
            if (time.perf_counter() - self.flash_start) >= self.flash_duration_s:
                shot: np.ndarray = self.cam.capture_frame()
                aspect_w, aspect_h = CROP_MODES[self.crop_index]
                shot = crop_center_to_aspect(shot, aspect_w, aspect_h)
                self.captured_frame_original = shot
                rendered: np.ndarray = self._render_captured(shot)
                self.captured_frame = rendered
                self.state = PhotoboothState.CAPTURED
                return rendered
            return np.full_like(last_frame, 255)
        assert self.captured_frame_original is not None
        rendered: np.ndarray = self._render_captured(self.captured_frame_original)
        self.captured_frame = rendered
        return rendered

    def _update_window_title(self, display_frame: np.ndarray) -> None:
        """Update the window title with resolution and FPS."""
        res_txt = f"{display_frame.shape[1]}x{display_frame.shape[0]}"
        fps_txt = (
            f"{self.fps_counter.text()}"
            if self.state == PhotoboothState.LIVE
            else "N/A"
        )
        title = (
            f"DotShot Live | {res_txt} | {fps_txt} fps | Thresh: {self.sobel_threshold}"
        )
        try:
            cv2.setWindowTitle(self.window_name, title)
        except Exception:
            pass

    def _handle_key(self, key: int) -> bool:
        """Handle a single key event.

        Args:
            key: Integer key code from OpenCV's waitKey.

        Returns:
            False to request exit; True to continue running.
        """
        if key in (27, ord("q")):
            return False
        if (key & 0xFF) == ord("f"):
            self.fullscreen = not self.fullscreen
            logging.debug("Fullscreen %s", "ON" if self.fullscreen else "OFF")
            try:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
                )
            except Exception as e:
                logging.error("Failed to set fullscreen mode: %s", e)
            return True
        if (key & 0xFF) == ord("c"):
            self.crop_index = (self.crop_index + 1) % len(CROP_MODES)
            logging.info(
                "Crop mode -> %d:%d",
                CROP_MODES[self.crop_index][0],
                CROP_MODES[self.crop_index][1],
            )
            return True
        if (key & 0xFF) == ord(" "):
            if self.state == PhotoboothState.LIVE:
                self._start_countdown()
            elif self.state == PhotoboothState.CAPTURED:
                self.state = PhotoboothState.LIVE
                self.captured_frame = None
                self.captured_frame_original = None
            return True
        if (key & 0xFF) == ord("e"):
            self.edge_enabled = not self.edge_enabled
            logging.info(
                "Edge detection %s", "ENABLED" if self.edge_enabled else "DISABLED"
            )
            return True
        # Arrow keys for threshold adjustment (Up/Down)
        if key in {82, 2490368}:  # Up
            self.sobel_threshold = min(255, self.sobel_threshold - 4)
            logging.debug("Threshold -> %d", self.sobel_threshold)
            return True
        if key in {84, 2621440}:  # Down
            self.sobel_threshold = max(0, self.sobel_threshold + 4)
            logging.debug("Threshold -> %d", self.sobel_threshold)
            return True
        if self.state != PhotoboothState.CAPTURED:
            return True
        if (key & 0xFF) == ord("p") and self.captured_frame is not None:
            self._print_captured()
            return True
        if (key & 0xFF) == ord("s") and self.captured_frame is not None:
            self._save_captured()
            return True
        return True

    def _start_countdown(self) -> None:
        """Enter countdown state and clear any previous capture."""
        self.state = PhotoboothState.COUNTDOWN
        self.countdown_start = time.perf_counter()
        self.captured_frame = None
        self.captured_frame_original = None

    def _start_flash(self) -> None:
        """Enter flash state and timestamp its start."""
        self.state = PhotoboothState.FLASH
        self.flash_start = time.perf_counter()

    def _print_captured(self) -> None:
        """Print the captured frame via the configured printer."""
        assert self.captured_frame is not None

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, self.captured_frame)
            logging.info(f"Printing {self.captured_frame.shape} frame from {tmp_path}")
            self.printer.print_image_file(tmp_path)
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _save_captured(self) -> None:
        """Save the captured frame to the repository's images directory."""
        assert self.captured_frame is not None
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        images_dir = os.path.join(repo_root, "images")
        os.makedirs(images_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(images_dir, f"dotshot_{ts}.png")
        cv2.imwrite(out_path, self.captured_frame)
        logging.info(f"Saved image to {out_path}")

    def _draw_center_text(
        self,
        image: np.ndarray,
        text: str,
        *,
        scale: float = 5.0,
        thickness: int = 6,
    ) -> np.ndarray:
        """Return a copy of image with centered white text overlay.

        Args:
            image: Grayscale or color image to draw on.
            text: Text to render at the center.
            scale: Font scale passed to OpenCV.
            thickness: Line thickness for the text strokes.

        Returns:
            A new image containing the centered text overlay.
        """
        output: np.ndarray = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, (output.shape[1] - text_w) // 2)
        y = max(text_h + baseline, (output.shape[0] + text_h) // 2)
        value = 0 if self.edge_enabled else 255
        color = value if output.ndim == 2 else (value, value, value)
        cv2.putText(output, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return output

    def _apply_edge_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel + thresholded Sobel union to a grayscale image.

        Args:
            image: Grayscale uint8 image.

        Returns:
            Grayscale uint8 image where edges are emphasized.
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            img_u8 = image.astype(np.uint8)
        else:
            img_u8 = image

        # Light blur to reduce noise
        blurred = cv2.GaussianBlur(img_u8, (3, 3), 0)

        # Sobel gradients and magnitude
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag_u8 = mag.astype(np.uint8)

        # Thresholded Sobel
        _, sobel_bin = cv2.threshold(
            mag_u8, int(self.sobel_threshold), 255, cv2.THRESH_BINARY
        )

        # Union of non-thresholded magnitude and thresholded map
        union = cv2.max(mag_u8, sobel_bin)

        # Invert so background is white and edges are black
        inv = cv2.bitwise_not(union)
        return inv

    def _render_captured(self, original: np.ndarray) -> np.ndarray:
        """Render the captured preview based on current edge settings.

        Args:
            original: The unmodified frame captured from the camera.

        Returns:
            The image to display for the captured state, re-rendered using the
            current edge detection settings and threshold.
        """
        if self.edge_enabled:
            return self._apply_edge_filter(original)
        return original


def main() -> None:
    """Entry point: parse args, build the app, and run it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="DotShot Live UI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Controls:\n"
            "  Space: Capture a frame or return to live view\n"
            "  p: Print captured frame\n"
            "  s: Save captured frame\n"
            "  f: Toggle fullscreen\n"
            "  q/Esc: Quit"
        ),
    )
    parser.add_argument(
        "--camera",
        dest="camera",
        type=str,
        default=None,
        help="Camera device index or path (e.g., 0 or /dev/video0)",
    )
    args = parser.parse_args()
    cam = USBCamera(device=(args.camera if args.camera is not None else 0))
    printer = Printer()
    app = DotShotLiveApp(cam=cam, printer=printer)
    app.run()


if __name__ == "__main__":
    main()
