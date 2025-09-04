import argparse
import logging
import os
from datetime import datetime
import time

import cv2
import numpy as np

from dotshot.livecamera import LiveCamera
from dotshot.printer import Printer


class FPSCounter:
    """Lightweight FPS calculator with periodic updates."""

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
        return self._fps

    def text(self) -> str:
        return "N/A" if not self._enabled else f"{self._fps:.1f}"


def _draw_center_text(
    image: np.ndarray,
    text: str,
    *,
    scale: float = 5.0,
    thickness: int = 6,
) -> np.ndarray:
    """Return a copy of image with centered white text overlay."""
    output: np.ndarray = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (output.shape[1] - text_w) // 2)
    y = max(text_h + baseline, (output.shape[0] + text_h) // 2)
    color = 255 if output.ndim == 2 else (255, 255, 255)
    cv2.putText(output, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return output

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="DotShot Live UI")
    parser.add_argument(
        "--camera",
        dest="camera",
        type=str,
        default=None,
        help="Camera device index or path (e.g., 0 or /dev/video0)",
    )
    args = parser.parse_args()

    cam = LiveCamera(device=(args.camera if args.camera is not None else 0))
    printer = Printer()

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        current_frame: np.ndarray
        cam.open()
        current_frame = cam.capture_frame()

        # FPS tracking for live mode only
        fps_counter = FPSCounter(enabled=True, update_period_s=1.0)

        # Photobooth state machine
        state: str = "live"  # live | countdown | flash | captured
        countdown_start: float = 0.0
        countdown_total: int = 3
        flash_start: float = 0.0
        flash_duration_s: float = 0.15
        captured_frame: np.ndarray | None = None

        fullscreen = False
        while True:
            # State update and frame selection
            if state == "live":
                current_frame = cam.capture_frame()
                fps_counter.update()
                display_frame: np.ndarray = current_frame
            elif state == "countdown":
                current_frame = cam.capture_frame()
                elapsed: float = time.perf_counter() - countdown_start
                remaining: int = countdown_total - int(elapsed)
                if remaining <= 0:
                    state = "flash"
                    flash_start = time.perf_counter()
                    display_frame = np.full_like(current_frame, 255)
                else:
                    display_frame = _draw_center_text(current_frame, str(remaining))
            elif state == "flash":
                if (time.perf_counter() - flash_start) >= flash_duration_s:
                    shot: np.ndarray = cam.capture_frame()
                    captured_frame = shot
                    state = "captured"
                    display_frame = captured_frame
                else:
                    display_frame = np.full_like(current_frame, 255)
            else:  # captured
                assert captured_frame is not None
                display_frame = captured_frame

            cv2.imshow("DotShot", display_frame)
            # Window presentation managed via 'f' toggle

            # Update window title (Qt builds only; safe to ignore if unsupported)
            res_txt = f"{display_frame.shape[1]}x{display_frame.shape[0]}"
            fps_txt = fps_counter.text() if state == "live" else "—"
            title = f"DotShot Live - {res_txt} @ {fps_txt}"
            try:
                cv2.setWindowTitle("DotShot", title)
            except Exception:
                pass

            key_full = int(cv2.waitKey(1))
            if key_full != -1:
                logging.debug(f"Key: {key_full}")
            key = key_full & 0xFFFFFFFF

            if key in (27, ord("q")):
                break

            if (key & 0xFF) == ord("f"):
                fullscreen = not fullscreen
                logging.debug("Fullscreen %s", "ON" if fullscreen else "OFF")
                try:
                    cv2.setWindowProperty(
                        "DotShot",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                except Exception as e:
                    logging.error("Failed to set fullscreen mode: %s", e)
                continue

            # Spacebar behavior
            if (key & 0xFF) == ord(" "):
                if state == "live":
                    state = "countdown"
                    countdown_start = time.perf_counter()
                    captured_frame = None
                elif state == "captured":
                    state = "live"
                    captured_frame = None
                # Ignore in countdown/flash
                continue

            # Only allow actions below when a photo is captured
            if state != "captured":
                continue

            if (key & 0xFF) == ord("p") and captured_frame is not None:
                import tempfile

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        cv2.imwrite(tmp_path, captured_frame)
                    logging.info(f"Printing {captured_frame.shape} frame from {tmp_path}")
                    printer.print_image_file(tmp_path)
                finally:
                    if tmp_path is not None:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                continue

            if (key & 0xFF) == ord("s") and captured_frame is not None:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                images_dir = os.path.join(repo_root, "images")
                os.makedirs(images_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(images_dir, f"dotshot_{ts}.png")
                cv2.imwrite(out_path, captured_frame)
                logging.info(f"Saved image to {out_path}")
                continue
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        finally:
            if cam is not None:
                try:
                    cam.close()
                except Exception as e:
                    logging.error("Error closing camera: %s", e)


if __name__ == "__main__":
    main()
