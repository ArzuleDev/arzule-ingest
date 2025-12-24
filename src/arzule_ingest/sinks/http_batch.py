"""HTTP batch sink for sending trace events to Arzule backend."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

from .base import TelemetrySink


class HttpBatchSink(TelemetrySink):
    """
    Batch and POST trace events to an HTTP endpoint.

    Batches events and sends them periodically or when buffer is full.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Initialize the HTTP batch sink.

        Args:
            endpoint_url: The ingest endpoint URL
            api_key: API key for authentication
            batch_size: Max events per batch
            flush_interval_seconds: Auto-flush interval
            timeout_seconds: HTTP request timeout
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.timeout_seconds = timeout_seconds

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        # Start background flush thread
        self._start_flush_thread()

    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval_seconds)
            self.flush()

    def write(self, event: dict[str, Any]) -> None:
        """Buffer a trace event."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.batch_size:
                self._send_batch()

    def flush(self) -> None:
        """Flush all buffered events."""
        with self._lock:
            if self._buffer:
                self._send_batch()

    def _send_batch(self) -> None:
        """Send current buffer to the endpoint (must hold lock)."""
        if not self._buffer:
            return

        batch = self._buffer.copy()
        self._buffer.clear()

        try:
            import httpx

            # Build JSONL payload
            payload = "\n".join(
                json.dumps(evt, separators=(",", ":"), default=str) for evt in batch
            )

            response = httpx.post(
                self.endpoint_url,
                content=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/x-ndjson",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

        except Exception as e:
            # Log error but don't crash - telemetry should be non-blocking
            # In production, implement retry logic or dead-letter queue
            import sys

            print(f"[arzule] Failed to send batch: {e}", file=sys.stderr)

    def close(self) -> None:
        """Stop the flush thread and send remaining events."""
        self._stop_event.set()
        self.flush()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)

    def __enter__(self) -> "HttpBatchSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

