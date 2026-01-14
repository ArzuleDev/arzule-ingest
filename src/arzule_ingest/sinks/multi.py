"""Multi-sink that combines multiple sinks for simultaneous output."""

from __future__ import annotations

from typing import Any

from .base import TelemetrySink


class MultiSink(TelemetrySink):
    """
    Combines multiple sinks to send events to multiple destinations.
    
    Example use cases:
    - Send to both Arzule backend and local streaming server
    - Write to file and send to remote endpoint
    - Replicate events to multiple backends
    
    Events are sent to all sinks; failures in one sink don't affect others.
    """
    
    def __init__(self, sinks: list[TelemetrySink]):
        """
        Initialize with multiple sinks.
        
        Args:
            sinks: List of TelemetrySink objects
        """
        self.sinks = sinks
    
    def write(self, event: dict[str, Any]) -> None:
        """Write event to all sinks."""
        for sink in self.sinks:
            try:
                sink.write(event)
            except Exception:
                pass  # Best effort - don't let one sink failure block others
    
    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self.sinks:
            try:
                sink.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass


