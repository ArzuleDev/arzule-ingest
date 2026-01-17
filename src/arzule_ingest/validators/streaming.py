"""Streaming support for validators.

This module provides utilities for streaming validation results in real-time
using Server-Sent Events (SSE) or WebSocket connections. This enables
progressive UI updates as validators complete.

Currently implements SSE client for receiving streamed results.
WebSocket support planned for bidirectional communication.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .types import ValidationResult, ValidatorResult


@dataclass
class StreamEvent:
    """A single event from the validation stream.

    Attributes:
        event_type: Type of event (validator_started, validator_completed,
                    aggregate_ready, error, heartbeat).
        data: Event payload data.
        timestamp_ms: Event timestamp in milliseconds.
    """

    event_type: str
    data: dict[str, Any]
    timestamp_ms: int

    @classmethod
    def from_sse_line(cls, line: str) -> Optional[StreamEvent]:
        """Parse an SSE data line into a StreamEvent.

        Args:
            line: SSE data line (e.g., 'data: {"type": "..."}'.

        Returns:
            StreamEvent if parsing succeeds, None otherwise.
        """
        if not line.startswith("data: "):
            return None

        try:
            data = json.loads(line[6:])
            return cls(
                event_type=data.get("type", "unknown"),
                data=data.get("data", {}),
                timestamp_ms=data.get("timestamp_ms", int(time.time() * 1000)),
            )
        except json.JSONDecodeError:
            return None


class ValidationStreamClient:
    """Client for streaming validation results.

    This client connects to the validators API streaming endpoint and
    yields events as validators start, complete, and aggregate results
    become available.

    The streaming approach provides several benefits over polling:
    - Lower latency: Events arrive immediately as they occur
    - More efficient: Single connection vs repeated requests
    - Progressive UI: Update UI as each validator completes

    Example:
        >>> client = ValidationStreamClient()
        >>> for event in client.stream_results("sess_abc", "agent_xyz"):
        ...     if event.event_type == "validator_completed":
        ...         print(f"Validator {event.data['validator_type']} done")
        ...     elif event.event_type == "aggregate_ready":
        ...         result = ValidationResult.from_dict(event.data)
        ...         print(f"Final score: {result.aggregate_score}")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the streaming client.

        Args:
            base_url: Override for validators API URL.
            api_key: Override for API key.
            debug: Enable debug logging.
        """
        self.base_url = (
            base_url
            or os.environ.get("ARZULE_VALIDATORS_URL")
            or "https://validators.arzule.com"
        ).rstrip("/")

        self.api_key = api_key or os.environ.get("ARZULE_API_KEY")
        self.debug = debug

    def _log(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(f"[validator-stream] {message}", file=sys.stderr)

    def stream_results(
        self,
        session_id: str,
        agent_id: str,
        timeout_seconds: float = 30.0,
    ) -> Generator[StreamEvent, None, None]:
        """Stream validation events for a subagent.

        This generator yields events as validators complete. The stream
        ends when all validators finish or the connection times out.

        Args:
            session_id: Claude Code session ID.
            agent_id: Subagent ID to stream results for.
            timeout_seconds: Maximum time to stream before closing.

        Yields:
            StreamEvent objects for each validation event.

        Example:
            >>> for event in client.stream_results("sess_abc", "agent_xyz"):
            ...     match event.event_type:
            ...         case "validator_started":
            ...             print(f"Started: {event.data['validator_type']}")
            ...         case "validator_completed":
            ...             print(f"Completed: {event.data['validator_type']}")
            ...         case "aggregate_ready":
            ...             print("All validators done!")
        """
        url = f"{self.base_url}/v1/validators/stream/{session_id}/{agent_id}"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._log(f"Opening stream: {url}")

        request = Request(url, headers=headers, method="GET")

        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                for line in self._read_sse_lines(response, timeout_seconds):
                    event = StreamEvent.from_sse_line(line)
                    if event:
                        self._log(f"Event: {event.event_type}")
                        yield event

                        # Stop streaming on aggregate_ready
                        if event.event_type == "aggregate_ready":
                            return

        except HTTPError as e:
            self._log(f"HTTP error: {e.code}")
            yield StreamEvent(
                event_type="error",
                data={"error": f"HTTP {e.code}", "reason": e.reason},
                timestamp_ms=int(time.time() * 1000),
            )

        except URLError as e:
            self._log(f"Network error: {e.reason}")
            yield StreamEvent(
                event_type="error",
                data={"error": "network_error", "reason": str(e.reason)},
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            self._log(f"Unexpected error: {e}")
            yield StreamEvent(
                event_type="error",
                data={"error": "unexpected_error", "reason": str(e)},
                timestamp_ms=int(time.time() * 1000),
            )

    def _read_sse_lines(
        self, response: Any, timeout_seconds: float
    ) -> Iterator[str]:
        """Read lines from SSE response with timeout.

        Args:
            response: HTTP response object.
            timeout_seconds: Maximum time to read.

        Yields:
            SSE data lines.
        """
        start_time = time.time()
        buffer = ""

        while True:
            if time.time() - start_time > timeout_seconds:
                self._log("Stream timeout")
                break

            try:
                chunk = response.read(1024)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        yield line

            except Exception as e:
                self._log(f"Read error: {e}")
                break


class ProgressiveResultCollector:
    """Collects streaming results and provides aggregate on demand.

    This class wraps the streaming client to provide a more convenient
    interface for collecting results with progress callbacks.

    Example:
        >>> def on_progress(completed, total, latest_result):
        ...     print(f"Progress: {completed}/{total}")
        ...
        >>> collector = ProgressiveResultCollector(on_progress=on_progress)
        >>> result = collector.collect("sess_abc", "agent_xyz")
        >>> print(f"Final: {result.aggregate_score}")
    """

    def __init__(
        self,
        client: Optional[ValidationStreamClient] = None,
        on_progress: Optional[
            Callable[[int, int, Optional[ValidatorResult]], None]
        ] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize the collector.

        Args:
            client: ValidationStreamClient instance.
            on_progress: Callback for progress updates (completed, total, latest).
            on_error: Callback for errors (error_type, message).
        """
        self.client = client or ValidationStreamClient()
        self.on_progress = on_progress
        self.on_error = on_error

    def collect(
        self,
        session_id: str,
        agent_id: str,
        timeout_seconds: float = 30.0,
    ) -> Optional[ValidationResult]:
        """Collect validation results with progress updates.

        Args:
            session_id: Claude Code session ID.
            agent_id: Subagent ID to collect results for.
            timeout_seconds: Maximum time to wait.

        Returns:
            Final ValidationResult, or None if collection failed.
        """
        completed_count = 0
        total_count = 0
        validator_results: list[ValidatorResult] = []
        final_result: Optional[ValidationResult] = None

        for event in self.client.stream_results(
            session_id=session_id,
            agent_id=agent_id,
            timeout_seconds=timeout_seconds,
        ):
            if event.event_type == "validator_started":
                total_count = max(total_count, event.data.get("total", 0))
                if self.on_progress:
                    self.on_progress(completed_count, total_count, None)

            elif event.event_type == "validator_completed":
                completed_count += 1
                result = ValidatorResult.from_dict(event.data)
                validator_results.append(result)
                if self.on_progress:
                    self.on_progress(completed_count, total_count, result)

            elif event.event_type == "aggregate_ready":
                final_result = ValidationResult.from_dict(event.data)
                break

            elif event.event_type == "error":
                if self.on_error:
                    self.on_error(
                        event.data.get("error", "unknown"),
                        event.data.get("reason", "Unknown error"),
                    )
                break

        return final_result


def stream_validation_progress(
    session_id: str,
    agent_id: str,
    on_event: Callable[[StreamEvent], None],
    timeout_seconds: float = 30.0,
) -> Optional[ValidationResult]:
    """Convenience function for streaming with event callback.

    This function provides a simple interface for streaming validation
    results with a callback for each event.

    Args:
        session_id: Claude Code session ID.
        agent_id: Subagent ID.
        on_event: Callback invoked for each stream event.
        timeout_seconds: Maximum stream duration.

    Returns:
        Final ValidationResult if aggregate_ready received.

    Example:
        >>> def handle_event(event):
        ...     print(f"Event: {event.event_type}")
        ...
        >>> result = stream_validation_progress(
        ...     "sess_abc", "agent_xyz",
        ...     on_event=handle_event,
        ... )
    """
    client = ValidationStreamClient()
    final_result = None

    for event in client.stream_results(
        session_id=session_id,
        agent_id=agent_id,
        timeout_seconds=timeout_seconds,
    ):
        on_event(event)

        if event.event_type == "aggregate_ready":
            final_result = ValidationResult.from_dict(event.data)
            break

    return final_result
