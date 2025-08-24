import logging
import cv2
import tempfile
import os

from dotshot.camera import USBCamera
from dotshot.printer import Printer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cam = USBCamera()
    printer = Printer()

    cv2.namedWindow("DotShot", cv2.WINDOW_NORMAL)
    try:
        with cam:
            frame = cam.capture_frame()
            while True:
                cv2.imshow("DotShot", frame)
                key = int(cv2.waitKey(0)) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    frame = cam.capture_frame()

                if key == ord("p"):
                    # Write frame to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        logging.info(f"Writing frame to temporary file: {tmp_path}")
                        cv2.imwrite(tmp_path, frame)
                    
                    try:
                        logging.info(f"Printing frame from temporary file: {tmp_path}")
                        printer.print_image_file(tmp_path)
                        logging.info(f"File sent to print queue: {tmp_path}")
                    finally:
                        # Clean up temporary file
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

