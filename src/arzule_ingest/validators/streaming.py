"""
Validator Result Streaming

Listens for validator results via Server-Sent Events (SSE) from the
Arzule backend and dispatches them to registered callbacks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

ValidatorResultCallback = Callable[[Dict[str, Any]], None]


class ValidatorResultStream:
    """
    Streams validator results from the backend.

    Uses SSE (Server-Sent Events) to receive real-time validation results
    without blocking the main event ingestion flow.

    Example:
        >>> stream = ValidatorResultStream(
        ...     endpoint="https://api.arzule.com",
        ...     api_key="your-api-key",
        ...     session_id="session-123",
        ...     on_result=lambda r: print(f"Got result: {r}"),
        ... )
        >>> stream.start()
        >>> # ... do work ...
        >>> stream.stop()
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        session_id: str,
        on_result: Optional[ValidatorResultCallback] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        """
        Initialize the validator result stream.

        Args:
            endpoint: Base URL of the Arzule backend (e.g., "https://api.arzule.com")
            api_key: API key for authentication
            session_id: Session ID to subscribe to for validator results
            on_result: Optional callback invoked for each validator result
            reconnect_delay: Base delay between reconnection attempts (seconds)
            max_reconnect_attempts: Maximum number of reconnection attempts before giving up
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.session_id = session_id
        self._on_result = on_result
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._result_queue: Queue[Dict[str, Any]] = Queue()

    def start(self) -> None:
        """Start the streaming listener in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name=f"validator-stream-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"Started validator result stream for session {self.session_id}")

    def stop(self) -> None:
        """Stop the streaming listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug(f"Stopped validator result stream for session {self.session_id}")

    def _listen_loop(self) -> None:
        """Main listening loop - runs in background thread."""
        reconnect_attempts = 0

        while self._running and reconnect_attempts < self._max_reconnect_attempts:
            try:
                self._connect_and_stream()
                reconnect_attempts = 0  # Reset on successful connection
            except httpx.HTTPError as e:
                logger.warning(f"Validator stream connection error: {e}")
                reconnect_attempts += 1
                if self._running:
                    time.sleep(self._reconnect_delay * reconnect_attempts)
            except Exception as e:
                logger.error(f"Validator stream error: {e}")
                break

        if reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Validator stream exceeded max reconnect attempts "
                f"({self._max_reconnect_attempts}), giving up"
            )

    def _connect_and_stream(self) -> None:
        """Connect to SSE endpoint and process events."""
        url = f"{self.endpoint}/v1/validators/stream/{self.session_id}"

        with httpx.stream(
            "GET",
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            timeout=None,  # SSE connections are long-lived
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not self._running:
                    break

                if line.startswith("data: "):
                    self._handle_event(line[6:])

    def _handle_event(self, data: str) -> None:
        """Process a single SSE event."""
        try:
            result = json.loads(data)

            # Put in queue for synchronous access
            self._result_queue.put(result)

            # Call callback if registered
            if self._on_result:
                try:
                    self._on_result(result)
                except Exception as e:
                    logger.error(f"Error in validator result callback: {e}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid validator result JSON: {e}")

    def get_result(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Get next result from queue.

        Args:
            timeout: How long to wait for a result (0.0 = non-blocking)

        Returns:
            The next validator result dict, or None if no result available
        """
        try:
            return self._result_queue.get(block=timeout > 0, timeout=timeout if timeout > 0 else None)
        except Empty:
            return None

    def get_all_results(self) -> list[Dict[str, Any]]:
        """
        Get all currently queued results without blocking.

        Returns:
            List of all validator results currently in the queue
        """
        results: list[Dict[str, Any]] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except Empty:
                break
        return results

    @property
    def is_running(self) -> bool:
        """Check if the stream listener is currently running."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Get the number of results waiting in the queue."""
        return self._result_queue.qsize()

    def __enter__(self) -> "ValidatorResultStream":
        """Context manager entry - starts the stream."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Context manager exit - stops the stream."""
        self.stop()
