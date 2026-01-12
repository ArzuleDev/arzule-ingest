"""Base class for telemetry sinks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TelemetrySink(ABC):
    """Abstract base class for trace event sinks."""

    @abstractmethod
    def write(self, event: dict[str, Any]) -> None:
        """Write a single trace event."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered events."""
        ...

    def close(self) -> None:
        """Close the sink and release resources."""
        self.flush()





