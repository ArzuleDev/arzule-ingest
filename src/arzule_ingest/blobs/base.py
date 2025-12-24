"""Base class for blob storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BlobStorage(ABC):
    """Abstract base class for blob storage backends."""

    @abstractmethod
    def upload(self, data: bytes, content_type: str = "application/octet-stream") -> dict[str, Any]:
        """
        Upload data and return a raw_ref dict.

        Args:
            data: The bytes to upload
            content_type: MIME type of the content

        Returns:
            A dict suitable for the raw_ref field of a TraceEvent
        """
        ...

    @abstractmethod
    def download(self, raw_ref: dict[str, Any]) -> bytes:
        """
        Download data from a raw_ref.

        Args:
            raw_ref: The raw_ref dict from a TraceEvent

        Returns:
            The blob data as bytes
        """
        ...

