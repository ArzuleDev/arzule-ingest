"""JSONL file sink for local trace event storage."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Optional

from .base import TelemetrySink


class JsonlFileSink(TelemetrySink):
    """
    Write trace events to a JSONL file (optionally gzipped).

    This is the MVP sink for local development and testing.
    """

    def __init__(
        self,
        path: str | Path,
        compress: bool = False,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize the JSONL file sink.

        Args:
            path: Output file path (will create parent directories)
            compress: If True, write gzipped output (.jsonl.gz)
            buffer_size: Number of events to buffer before flushing
        """
        self.path = Path(path)
        self.compress = compress
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, Any]] = []
        self._file: Optional[Any] = None

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file handle
        if self.compress:
            if not str(self.path).endswith(".gz"):
                self.path = Path(str(self.path) + ".gz")
            self._file = gzip.open(self.path, "wt", encoding="utf-8")
        else:
            self._file = open(self.path, "w", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        """Buffer and write a trace event."""
        self._buffer.append(event)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if not self._buffer or not self._file:
            return

        for event in self._buffer:
            line = json.dumps(event, separators=(",", ":"), default=str)
            self._file.write(line + "\n")

        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Close the file handle."""
        self.flush()
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonlFileSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

