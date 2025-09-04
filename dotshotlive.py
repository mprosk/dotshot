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

        # FPS tracking (camera only)
        fps_counter = FPSCounter(enabled=True, update_period_s=1.0)

        fullscreen = False
        while True:
            current_frame = cam.capture_frame()
            fps_counter.update()
            cv2.imshow("DotShot", current_frame)
            # Window presentation managed via 'f' toggle

            # Update window title with current status (Qt builds only; safe to ignore if unsupported)
            res_txt = f"{current_frame.shape[1]}x{current_frame.shape[0]}"
            title = f"DotShot Live - {res_txt} @ {fps_counter.text()}"
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

            if (key & 0xFF) == ord("p"):
                # Write current frame to a temp file and send to printer
                import tempfile

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        cv2.imwrite(tmp_path, current_frame)
                    logging.info(f"Printing {current_frame.shape} frame from {tmp_path}")
                    printer.print_image_file(tmp_path)
                finally:
                    if tmp_path is not None:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                continue

            if (key & 0xFF) == ord("s"):
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                images_dir = os.path.join(repo_root, "images")
                os.makedirs(images_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(images_dir, f"dotshot_{ts}.png")
                cv2.imwrite(out_path, current_frame)
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
