import argparse
import logging
import os
from datetime import datetime

import cv2
import numpy as np

from dotshotlive.livecamera import LiveCamera
from dotshotlive.printer import Printer


def _load_and_process_file(path: str) -> np.ndarray:
    """Load image file as grayscale, normalize, and crop 16:9 to 4:3 if needed."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image from {path}")
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    h, w = int(img.shape[0]), int(img.shape[1])
    if w * 9 == h * 16:
        if w * 3 > h * 4:
            target_w = max(1, (h * 4) // 3)
            x0 = (w - target_w) // 2
            x1 = x0 + target_w
            y0, y1 = 0, h
        else:
            target_h = max(1, (w * 3) // 4)
            y0 = (h - target_h) // 2
            y1 = y0 + target_h
            x0, x1 = 0, w
        img = img[y0:y1, x0:x1]
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="DotShot UI")
    parser.add_argument(
        "--file",
        dest="file",
        type=str,
        default=None,
        help="Run without camera: load this image file",
    )
    parser.add_argument(
        "--camera",
        dest="camera",
        type=str,
        default=None,
        help="Camera device index or path (e.g., 0 or /dev/video0)",
    )
    args = parser.parse_args()

    use_file = args.file is not None

    cam = None if use_file else LiveCamera(
        device=(args.camera if args.camera is not None else 0)
    )
    printer = Printer()

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        current_frame: np.ndarray
        if use_file:
            current_frame = _load_and_process_file(args.file)
        else:
            assert cam is not None
            cam.open()
            current_frame = cam.capture_frame()

        fullscreen = False
        while True:
            if not use_file:
                assert cam is not None
                current_frame = cam.capture_frame()
            cv2.imshow("DotShot", current_frame)
            # Window presentation managed via 'f' toggle

            # Update window title with current status (Qt builds only; safe to ignore if unsupported)
            mode = "File" if use_file else "Camera"
            res_txt = f"{current_frame.shape[1]}x{current_frame.shape[0]}"
            title = f"DotShot Live - Mode: {mode} | Res: {res_txt}"
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
            if not use_file and cam is not None:
                try:
                    cam.close()
                except Exception as e:
                    logging.error("Error closing camera: %s", e)


if __name__ == "__main__":
    main()
