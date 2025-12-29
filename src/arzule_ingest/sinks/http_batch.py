"""HTTP batch sink for sending trace events to Arzule backend."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

from .base import TelemetrySink


class TLSRequiredError(ValueError):
    """Raised when TLS is required but endpoint uses HTTP."""

    pass


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
        require_tls: bool = True,
    ) -> None:
        """
        Initialize the HTTP batch sink.

        Args:
            endpoint_url: The ingest endpoint URL
            api_key: API key for authentication
            batch_size: Max events per batch
            flush_interval_seconds: Auto-flush interval
            timeout_seconds: HTTP request timeout
            require_tls: SOC2 - Require HTTPS (default: True)
        """
        # SOC2: Enforce TLS for data in transit
        if require_tls:
            parsed = urlparse(endpoint_url)
            if parsed.scheme.lower() != "https":
                raise TLSRequiredError(
                    f"SOC2 compliance requires HTTPS. Got scheme '{parsed.scheme}'. "
                    "Set require_tls=False to disable (not recommended)."
                )

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.require_tls = require_tls

        # SOC2: Pre-compute safe endpoint for error logging (no query params/credentials)
        parsed = urlparse(endpoint_url)
        self._safe_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

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
        """Buffer a trace event.
        
        If the event has a different run_id than existing buffer events,
        the buffer is flushed first to prevent mixing runs in the same batch.
        """
        import sys
        with self._lock:
            event_run_id = event.get("run_id")
            
            # If buffer has events from a different run, flush first
            if self._buffer:
                first_run_id = self._buffer[0].get("run_id")
                if event_run_id != first_run_id:
                    print(
                        f"[arzule] Run boundary detected, flushing {len(self._buffer)} events from previous run",
                        file=sys.stderr,
                    )
                    self._send_batch()
            
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
        # NOTE: Don't clear buffer until send succeeds - prevents losing run.end events

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

            # SUCCESS: Only now clear the buffer (events are safely delivered)
            self._buffer.clear()

            # SOC2: Audit log the successful write
            from ..audit import audit_log
            audit_log().log_data_write(
                sink_type="http_batch",
                destination=self._safe_endpoint,
                event_count=len(batch),
                encrypted=self.require_tls,
            )

        except Exception as e:
            # Log error but don't crash - telemetry should be non-blocking
            import sys

            error_type = type(e).__name__
            status_info = ""
            server_error = ""
            status_code = None
            
            # Try to get response info from httpx exception
            if hasattr(e, "response") and e.response is not None:
                resp = e.response
                status_code = getattr(resp, "status_code", None)
                if status_code:
                    status_info = f", status={status_code}"
                    
                    if status_code in (401, 403):
                        status_info += " (check API key)"
                    
                    # Extract server error message
                    try:
                        # httpx stores response body - try multiple ways to get it
                        response_text = None
                        if hasattr(resp, "text"):
                            response_text = resp.text
                        elif hasattr(resp, "content"):
                            response_text = resp.content.decode("utf-8")
                        elif hasattr(resp, "read"):
                            response_text = resp.read().decode("utf-8")
                        
                        if response_text:
                            error_data = json.loads(response_text)
                            err_code = error_data.get("error", "")
                            err_msg = error_data.get("message", "")
                            server_error = f" - {err_code}: {err_msg}"
                    except Exception as parse_err:
                        server_error = f" - (could not parse response: {parse_err})"
            
            # Always log first event info on error for debugging
            if batch:
                first = batch[0]
                print(
                    f"[arzule:debug] First event in failed batch: "
                    f"run_id={first.get('run_id')}, tenant_id={first.get('tenant_id')}, "
                    f"project_id={first.get('project_id')}, event_type={first.get('event_type')}, "
                    f"span_id={first.get('span_id')}, trace_id={first.get('trace_id')}",
                    file=sys.stderr,
                )
            
            print(
                f"[arzule] Failed to send batch: {error_type}{status_info}{server_error} "
                f"(batch_size={len(batch)}, endpoint={self._safe_endpoint})",
                file=sys.stderr,
            )
            
            # For 4xx errors (except retryable ones), clear buffer to prevent infinite loops
            if status_code and 400 <= status_code < 500 and status_code not in (401, 403, 429):
                self._buffer.clear()

    def clear_buffer(self) -> int:
        """
        Clear all buffered events without sending.
        
        Returns the number of events that were cleared.
        Used when starting a new run to prevent mixing events from different runs.
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count

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

