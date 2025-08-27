"""
Utilities for printing to CUPS-managed dot matrix printers.

Defaults target the text queue "ml320_text" and a graphics queue
"ml320_gfx".

This module shells out to the system `lp` command rather than requiring
pycups, which keeps deployment simple on systems with CUPS already
configured.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional
import logging


class Printer:
    """
    A simple interface for printing text and images via CUPS.

    Args:
        text_queue_name: Name of the printer queue to use for text (defaults to "ml320_text").
        graphics_queue_name: Name of the printer queue to use for graphics (defaults to "ml320_gfx").
    """

    _REQUEST_ID_PATTERN = re.compile(r"request id is .*?-(\d+)")

    def __init__(
        self,
        text_queue_name: str = "ml320_text",
        graphics_queue_name: str = "ml320_gfx",
    ) -> None:
        self.text_queue_name: str = text_queue_name
        self.graphics_queue_name: str = graphics_queue_name

    def print_text(
        self,
        text: str,
        *,
        append_form_feed: bool = False,
    ) -> int:
        """
        Print the provided text and return the CUPS job id.

        Args:
            text: The content to print.
            append_form_feed: If true, append form feed (\x0c) at end to advance paper.

        Returns:
            int: Numeric CUPS job id.

        Raises:
            RuntimeError: If `lp` fails or job id cannot be parsed.
        """

        if append_form_feed:
            text = f"{text}\x0c"

        tmp_path: Optional[str] = None
        try:
            tmp_path = self._write_temp_text_file(text)
            job_id = self._lp_submit(self.text_queue_name, [tmp_path])
            logging.info("Submitted text print job %d to queue %s", job_id, self.text_queue_name)
            return job_id
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    # Best effort cleanup; ignore failures.
                    pass

    def print_image_file(self, image_path: str) -> int:
        """
        Print an image file via the graphics queue and return the CUPS job id.

        CUPS should rasterize the file according to the printer's PPD/filter.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            int: Numeric CUPS job id.

        Raises:
            FileNotFoundError: If the image path does not exist.
            RuntimeError: If `lp` fails or job id cannot be parsed.
        """

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        job_id = self._lp_submit(
            self.graphics_queue_name, [image_path], ["-o", "Resolution=60x72dpi"]
        )
        logging.info("Submitted image print job %d to queue %s", job_id, self.graphics_queue_name)
        return job_id

    def print_text_file(self, file_path: str) -> int:
        """
        Print an existing text file via the text queue.

        Args:
            file_path: Path to an existing text file on disk.

        Returns:
            int: Numeric CUPS job id.

        Raises:
            FileNotFoundError: If the file path does not exist.
            RuntimeError: If `lp` fails or job id cannot be parsed.
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        job_id = self._lp_submit(self.text_queue_name, [file_path])
        logging.info("Submitted text-file print job %d to queue %s", job_id, self.text_queue_name)
        return job_id

    def _lp_submit(
        self, queue_name: str, file_paths: list[str], options: list[str] = []
    ) -> int:
        """
        Submit one or more files to a CUPS queue using `lp` and return the job id.

        Args:
            queue_name: Target CUPS queue name.
            file_paths: One or more filesystem paths to submit to the queue.

        Returns:
            int: Numeric CUPS job id.

        Raises:
            RuntimeError: If `lp` fails or job id cannot be parsed.
        """

        cmd: list[str] = ["lp", "-d", queue_name]
        cmd += options
        cmd += file_paths

        logging.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "").strip()
            raise RuntimeError(
                f"lp failed (exit {result.returncode}): {stderr_snippet}"
            )

        job_id: Optional[int] = self._parse_lp_request_id(result.stdout)
        if job_id is None:
            stdout_snippet = (result.stdout or "").strip()
            raise RuntimeError(
                f"Could not parse job id from lp output: {stdout_snippet}"
            )
        return job_id

    @staticmethod
    def _write_temp_text_file(text: str, encoding: str = "utf-8") -> str:
        """
        Write text to a temporary file and return its path.

        Args:
            text: The text content to write.
            encoding: Encoding used for the file.

        Returns:
            str: Path to the created temporary file.
        """

        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as tmp:
            data: bytes = text.encode(encoding, errors="replace")
            tmp.write(data)
            tmp.flush()
            return tmp.name

    @classmethod
    def _parse_lp_request_id(cls, output: str) -> Optional[int]:
        """
        Extract a numeric job id from `lp` output.

        Args:
            output: The stdout from an `lp` invocation.

        Returns:
            Optional[int]: Parsed job id or None if not found.
        """

        match = cls._REQUEST_ID_PATTERN.search(output or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None


__all__ = [
    "Printer",
]
