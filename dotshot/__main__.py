import logging
import os
from datetime import datetime

import cv2

from dotshot.camera import USBCamera
from dotshot.pipeline import ImagePipeline
from dotshot.printer import Printer

UP_KEYS = {82, 2490368}  # Up arrow (Linux/Windows)
DOWN_KEYS = {84, 2621440}  # Down arrow (Linux/Windows)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cam = USBCamera()
    printer = Printer()

    pipeline = ImagePipeline(
        cam=cam,
        printer=printer,
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
