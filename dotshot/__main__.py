import argparse
import logging
import os
from datetime import datetime

import cv2

from dotshot.camera import USBCamera
from dotshot.pipeline import QUANT_LEVELS, ImagePipeline
from dotshot.printer import Printer

UP_KEYS = {82, 2490368, 0}  # Up arrow (Linux/Windows/macOS)
DOWN_KEYS = {84, 2621440, 1}  # Down arrow (Linux/Windows/macOS)
LEFT_KEYS = {81, 2424832, 2}  # Left arrow (Linux/Windows/macOS)
RIGHT_KEYS = {83, 2555904, 3}  # Right arrow (Linux/Windows/macOS)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
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
    args = parser.parse_args()

    use_file = args.file is not None

    cam = None if use_file else USBCamera()
    printer = Printer()

    pipeline = ImagePipeline(
        cam=cam,
        printer=printer,
    )

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("DotShot", 1280, 960)
    try:
        if use_file:
            pipeline.load_file(args.file)
        else:
            assert cam is not None
            with cam:
                pipeline.capture()

        while True:
            display = pipeline.get_display()
            cv2.imshow("DotShot", display)
            cv2.resizeWindow("DotShot", 1280, 960)

            # Update window title with current status (Qt builds only; safe to ignore if unsupported)
            mode = "File" if use_file else "Camera"
            levels_val = QUANT_LEVELS[pipeline.quant_index] if QUANT_LEVELS else 256
            res_txt = f"{pipeline.orig.shape[1]}x{pipeline.orig.shape[0]}"
            title = (
                f"DotShot - Mode: {mode} | Res: {res_txt} | Levels: {levels_val} | "
                f"Offset: {pipeline.level_offset:+d}"
            )
            try:
                cv2.setWindowTitle("DotShot", title)
            except Exception:
                pass

            key_full = int(cv2.waitKey(30))
            if key_full != -1:
                logging.debug(f"Key: {key_full}")
            key = key_full & 0xFFFFFFFF

            if key in (27, ord("q")):
                break

            if (key & 0xFF) == ord("c") and not use_file:
                pipeline.recapture()
                continue

            if (key & 0xFF) == ord("r"):
                pipeline.cycle_resolution()
                continue

            if key in UP_KEYS:
                pipeline.adjust_offset(+1)
                continue

            if key in DOWN_KEYS:
                pipeline.adjust_offset(-1)
                continue

            if key in LEFT_KEYS and len(QUANT_LEVELS) > 0:
                new_idx = (pipeline.quant_index - 1) % len(QUANT_LEVELS)
                pipeline.set_quant_index(new_idx)
                continue

            if key in RIGHT_KEYS and len(QUANT_LEVELS) > 0:
                new_idx = (pipeline.quant_index + 1) % len(QUANT_LEVELS)
                pipeline.set_quant_index(new_idx)
                continue

            if (key & 0xFF) == ord("p"):
                pipeline.print_current()
                continue

            if (key & 0xFF) == ord("s"):
                # Save current printable and raw frames to repo-root images folder
                frame = pipeline.get_print_frame()
                raw = pipeline.get_raw_frame()
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                images_dir = os.path.join(repo_root, "images")
                os.makedirs(images_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_print = os.path.join(images_dir, f"dotshot_{ts}_print.png")
                out_raw = os.path.join(images_dir, f"dotshot_{ts}_raw.png")
                cv2.imwrite(out_print, frame)
                cv2.imwrite(out_raw, raw)
                logging.info(f"Saved images to {out_print} and {out_raw}")
                continue
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
