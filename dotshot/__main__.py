import argparse
import logging
import os
from datetime import datetime

import cv2

from dotshot.camera import USBCamera
from dotshot.pipeline import ImagePipeline
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

    parser = argparse.ArgumentParser(
        description="DotShot UI",
        epilog=(
            "Keys:\n"
            "  Space: capture (camera mode)\n"
            "  f: toggle fullscreen\n"
            "  e: cycle edge mode\n"
            "  r: cycle resolution\n"
            "  c: cycle crop mode (4:3, 1:1)\n"
            "  Up/Down: adjust Sobel threshold\n"
            "  p: print current\n"
            "  s: save print/raw images\n"
            "  q/Esc: quit"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
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

    cam = (
        None
        if use_file
        else USBCamera(device=(args.camera if args.camera is not None else 0))
    )
    printer = Printer()

    pipeline = ImagePipeline(
        cam=cam,
        printer=printer,
    )

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        if use_file:
            pipeline.load_file(args.file)
        else:
            assert cam is not None
            cam.open()
            pipeline.capture()

        fullscreen = False
        while True:
            display = pipeline.get_display()
            cv2.imshow("DotShot", display)
            # Window presentation managed via 'f' toggle

            # Update window title with current status (Qt builds only; safe to ignore if unsupported)
            mode = "File" if use_file else "Camera"
            res_txt = f"{pipeline.orig.shape[1]}x{pipeline.orig.shape[0]}"
            title = (
                f"DotShot - Mode: {mode} | Res: {res_txt} | Thresh: {pipeline.sobel_threshold} | "
                f"Edge: {pipeline.edge_mode_name()} | Crop: {pipeline.crop_mode_name()}"
            )
            try:
                cv2.setWindowTitle("DotShot", title)
            except Exception:
                pass

            key_full = int(cv2.waitKey())
            if key_full != -1:
                logging.debug(f"Key: {key_full}")
            key = key_full & 0xFFFFFFFF

            if key in (27, ord("q")):
                break

            if (key & 0xFF) == ord(" ") and not use_file:
                pipeline.capture()
                continue

            if (key & 0xFF) == ord("c"):
                pipeline.cycle_crop_mode()
                continue

            if (key & 0xFF) == ord("r"):
                pipeline.cycle_resolution()
                continue

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

            if (key & 0xFF) == ord("e"):
                pipeline.toggle_edge()
                continue

            if key in UP_KEYS:
                pipeline.adjust_sobel_threshold(-4)
                continue

            if key in DOWN_KEYS:
                pipeline.adjust_sobel_threshold(+4)
                continue

            # Quantization key mappings removed; quantization remains in pipeline

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
        finally:
            if not use_file and cam is not None:
                try:
                    cam.close()
                except Exception as e:
                    logging.error("Error closing camera: %s", e)


if __name__ == "__main__":
    main()
