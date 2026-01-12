"""JSONL file sink for local trace event storage."""

from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
from typing import Any, Optional

from .base import TelemetrySink


class JsonlFileSink(TelemetrySink):
    """
    Write trace events to a JSONL file (optionally gzipped and/or encrypted).

    SOC2: Supports encryption at rest via AES-256 (Fernet).
    """

    def __init__(
        self,
        path: str | Path,
        compress: bool = False,
        buffer_size: int = 100,
        encrypt: bool = False,
        encryption_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the JSONL file sink.

        Args:
            path: Output file path (will create parent directories)
            compress: If True, write gzipped output (.jsonl.gz)
            buffer_size: Number of events to buffer before flushing
            encrypt: SOC2 - If True, encrypt file contents (AES-256)
            encryption_key: Base64-encoded 32-byte key. If None and encrypt=True,
                           a key will be generated and saved to {path}.key
        """
        self.path = Path(path)
        self.compress = compress
        self.buffer_size = buffer_size
        self.encrypt = encrypt
        self._buffer: list[dict[str, Any]] = []
        self._file: Optional[Any] = None
        self._fernet: Optional[Any] = None

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # SOC2: Setup encryption if enabled
        if self.encrypt:
            self._setup_encryption(encryption_key)

        # Open file handle
        if self.compress:
            if not str(self.path).endswith(".gz"):
                self.path = Path(str(self.path) + ".gz")
            self._file = gzip.open(self.path, "wt", encoding="utf-8")
        else:
            self._file = open(self.path, "w", encoding="utf-8")

    def _setup_encryption(self, encryption_key: Optional[str]) -> None:
        """Setup Fernet encryption for data at rest."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError(
                "SOC2 encryption requires 'cryptography' package. "
                "Install with: pip install cryptography"
            )

        if encryption_key:
            # Use provided key
            key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
        else:
            # Generate new key and save it
            key = Fernet.generate_key()
            key_path = Path(str(self.path) + ".key")
            key_path.write_bytes(key)
            # Set restrictive permissions (owner read/write only)
            os.chmod(key_path, 0o600)

            # SOC2: Audit log key generation
            from ..audit import audit_log
            audit_log().log_encryption_key_generated(str(key_path))

        self._fernet = Fernet(key)

    def write(self, event: dict[str, Any]) -> None:
        """Buffer and write a trace event."""
        self._buffer.append(event)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if not self._buffer or not self._file:
            return

        # Capture count before clearing
        event_count = len(self._buffer)

        for event in self._buffer:
            line = json.dumps(event, separators=(",", ":"), default=str)

            # SOC2: Encrypt each line if encryption is enabled
            if self._fernet:
                encrypted = self._fernet.encrypt(line.encode("utf-8"))
                line = base64.b64encode(encrypted).decode("ascii")

            self._file.write(line + "\n")

        self._file.flush()
        self._buffer.clear()

        # SOC2: Audit log the write
        from ..audit import audit_log
        audit_log().log_data_write(
            sink_type="jsonl_file",
            destination=str(self.path),
            event_count=event_count,
            encrypted=self.encrypt,
        )

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

