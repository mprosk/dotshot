"""
Windows printing utilities using the native Win32 spooler (pywin32) and GDI.

This module provides a Windows-specific `PrinterWindows` class with a similar
shape to the Linux/CUPS version but does not depend on CUPS.

Text printing is sent as RAW bytes to the spooler so the printer's own
builtin/system font is used (important for dot-matrix printers). Image printing
uses GDI to render bitmaps.

Notes:
- Requires pywin32 (win32print, win32ui, win32con) and Pillow (PIL).
- Returned job id is not guaranteed; methods return Optional[int] and may
  return None when a job id cannot be retrieved from the GDI path.
- This file is standalone and not wired into the rest of the project yet.
"""

from __future__ import annotations

from typing import Optional, Tuple
import os

try:
    import win32print  # type: ignore
    import win32ui  # type: ignore
    import win32con  # type: ignore
except Exception as exc:  # pragma: no cover - Windows-only dependency
    raise SystemExit(
        "pywin32 is required on Windows. Install with: pip install pywin32"
    ) from exc

try:
    from PIL import Image, ImageWin  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Pillow is required for image printing. Install with: pip install Pillow"
    ) from exc


class PrinterWindows:
    """
    Simple Windows printer interface using GDI.

    Args:
        printer_name: Target printer name. If None, uses the default printer.
    """

    def __init__(self, printer_name: Optional[str] = None) -> None:
        self.printer_name: str = (
            printer_name if printer_name is not None else win32print.GetDefaultPrinter()
        )

    def print_text(
        self,
        text: str,
        *,
        title: str = "DotShot Text",
        append_form_feed: bool = False,
        encoding: str = "ascii",
        normalize_crlf: bool = True,
    ) -> Optional[int]:
        """Print plain text by sending RAW bytes to the spooler (printer font).

        This avoids GDI text rendering and lets the printer use its builtin
        font (common on dot-matrix printers).

        Args:
            text: The content to print.
            title: Document title shown in the spooler UI.
            append_form_feed: If true, append \x0c to advance paper at end.
            encoding: Text encoding for bytes conversion (e.g., "ascii", "cp437").
            normalize_crlf: Convert all newlines to Windows CRLF (\r\n).

        Returns:
            Optional[int]: Spool job id if available, else None.
        """
        data_text: str = text
        if normalize_crlf:
            # Normalize newlines to CRLF for many printer interpreters
            data_text = data_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
        if append_form_feed:
            data_text += "\x0c"
        data: bytes = data_text.encode(encoding, errors="replace")

        hprinter = win32print.OpenPrinter(self.printer_name)
        try:
            # Level 1 DOCINFO: (pDocName, pOutputFile, pDatatype)
            job_id = win32print.StartDocPrinter(hprinter, 1, (title, None, "RAW"))
            try:
                win32print.StartPagePrinter(hprinter)
                win32print.WritePrinter(hprinter, data)
                win32print.EndPagePrinter(hprinter)
            finally:
                win32print.EndDocPrinter(hprinter)
        finally:
            win32print.ClosePrinter(hprinter)

        try:
            return int(job_id)
        except Exception:
            return None

    def print_image_file(
        self,
        image_path: str,
        *,
        title: str = "DotShot Image",
        fit_mode: str = "fit",  # "fit" or "fill"
        margin_px: int = 50,
    ) -> Optional[int]:
        """Print an image file using GDI. Maintains aspect ratio.

        Args:
            image_path: Path to the image file on disk.
            title: Document title for the spooler.
            fit_mode: "fit" to fit within printable area, "fill" to cover it.
            margin_px: Margins in pixels on each side within the printable area.

        Returns:
            Optional[int]: Spool job id if known, otherwise None.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")

        hdc = win32ui.CreateDC()
        hdc.CreatePrinterDC(self.printer_name)

        # Printable area in pixels
        horz_res = hdc.GetDeviceCaps(win32con.HORZRES)
        vert_res = hdc.GetDeviceCaps(win32con.VERTRES)

        # Compute destination rectangle while keeping aspect
        max_w = max(1, int(horz_res) - 2 * margin_px)
        max_h = max(1, int(vert_res) - 2 * margin_px)
        dst_w, dst_h = _compute_fit(img.size, (max_w, max_h), fit_mode)

        left = (horz_res - dst_w) // 2
        top = (vert_res - dst_h) // 2
        right = left + dst_w
        bottom = top + dst_h

        dib = ImageWin.Dib(img)

        # Begin document
        hdc.StartDoc(title)
        try:
            hdc.StartPage()
            dib.draw(hdc.GetHandleOutput(), (left, top, right, bottom))
            hdc.EndPage()
            hdc.EndDoc()
        except Exception:
            try:
                hdc.AbortDoc()
            except Exception:
                pass
            raise
        finally:
            hdc.DeleteDC()

        # GDI path does not expose job id reliably
        return None


def _compute_fit(
    src_size: Tuple[int, int],
    dst_max: Tuple[int, int],
    fit_mode: str,
) -> Tuple[int, int]:
    """Compute (w, h) that fits or fills dst_max while preserving aspect ratio.

    Args:
        src_size: Source (width, height)
        dst_max: Max (width, height)
        fit_mode: "fit" or "fill"

    Returns:
        Tuple[int, int]: Destination (width, height)
    """
    src_w, src_h = src_size
    max_w, max_h = dst_max
    if src_w <= 0 or src_h <= 0 or max_w <= 0 or max_h <= 0:
        return max(1, max_w), max(1, max_h)

    scale_w = max_w / float(src_w)
    scale_h = max_h / float(src_h)
    if fit_mode.lower() == "fill":
        scale = max(scale_w, scale_h)
    else:
        scale = min(scale_w, scale_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    return dst_w, dst_h


__all__ = ["PrinterWindows"]


