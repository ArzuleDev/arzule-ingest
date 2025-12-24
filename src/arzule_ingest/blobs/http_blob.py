"""HTTP blob storage for large payloads."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Optional


from .base import BlobStorage


class HttpBlobStorage(BlobStorage):
    """
    HTTP-based blob storage for large payloads.

    Uploads to a configured endpoint and stores a reference URL.
    """

    def __init__(
        self,
        upload_url: str,
        api_key: str,
        download_base_url: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Initialize HTTP blob storage.

        Args:
            upload_url: URL for uploading blobs
            api_key: API key for authentication
            download_base_url: Base URL for downloads (defaults to upload_url)
            timeout_seconds: Request timeout
        """
        self.upload_url = upload_url.rstrip("/")
        self.api_key = api_key
        self.download_base_url = (download_base_url or upload_url).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def upload(self, data: bytes, content_type: str = "application/octet-stream") -> dict[str, Any]:
        """Upload data to HTTP endpoint."""
        import httpx

        # Generate unique blob ID
        blob_id = str(uuid.uuid4())
        checksum = hashlib.sha256(data).hexdigest()

        response = httpx.post(
            f"{self.upload_url}/{blob_id}",
            content=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
                "X-Content-SHA256": checksum,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        return {
            "storage": "http",
            "url": f"{self.download_base_url}/{blob_id}",
            "size": len(data),
            "content_type": content_type,
            "sha256": checksum,
        }

    def download(self, raw_ref: dict[str, Any]) -> bytes:
        """Download data from HTTP endpoint."""
        import httpx

        if raw_ref.get("storage") != "http":
            raise ValueError(f"Not an HTTP ref: {raw_ref.get('storage')}")

        url = raw_ref.get("url")
        if not url:
            raise ValueError("Missing URL in raw_ref")

        response = httpx.get(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        # Verify checksum if available
        expected_sha256 = raw_ref.get("sha256")
        if expected_sha256:
            actual_sha256 = hashlib.sha256(response.content).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
                )

        return response.content

