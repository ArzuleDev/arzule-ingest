"""Inline blob storage - embeds data directly in the event."""

from __future__ import annotations

import base64
from typing import Any

from .base import BlobStorage


class InlineBlob(BlobStorage):
    """
    Inline blob storage that embeds data directly in the raw_ref.

    This is the default for small payloads. Data is base64-encoded
    and stored in the event itself.
    """

    def __init__(self, max_size_bytes: int = 64 * 1024) -> None:
        """
        Initialize inline blob storage.

        Args:
            max_size_bytes: Maximum size for inline storage (default 64KB)
        """
        self.max_size_bytes = max_size_bytes

    def upload(self, data: bytes, content_type: str = "application/octet-stream") -> dict[str, Any]:
        """Store data inline as base64."""
        if len(data) > self.max_size_bytes:
            raise ValueError(
                f"Data too large for inline storage: {len(data)} > {self.max_size_bytes}"
            )

        return {
            "storage": "inline",
            "content_type": content_type,
            "size": len(data),
            "data_b64": base64.b64encode(data).decode("ascii"),
        }

    def download(self, raw_ref: dict[str, Any]) -> bytes:
        """Retrieve inline base64 data."""
        if raw_ref.get("storage") != "inline":
            raise ValueError(f"Not an inline ref: {raw_ref.get('storage')}")

        data_b64 = raw_ref.get("data_b64")
        if not data_b64:
            return b""

        return base64.b64decode(data_b64)


