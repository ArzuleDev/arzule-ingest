"""ArzuleRun - Core runtime context manager for trace collection."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .ids import new_run_id, new_span_id, new_trace_id

if TYPE_CHECKING:
    from .sinks.base import TelemetrySink

_active_run: ContextVar[Optional["ArzuleRun"]] = ContextVar("_active_run", default=None)


def current_run() -> Optional["ArzuleRun"]:
    """Get the currently active ArzuleRun from context."""
    return _active_run.get()


@dataclass
class ArzuleRun:
    """
    Context manager for a single observability run.

    Manages trace lifecycle, sequence numbering, and event emission.
    """

    tenant_id: str
    project_id: str
    sink: "TelemetrySink"
    run_id: str = field(default_factory=new_run_id)
    trace_id: str = field(default_factory=new_trace_id)

    # Internal state
    _seq: int = field(default=0, repr=False)
    _root_span_id: Optional[str] = field(default=None, repr=False)

    # Maps for correlating start/end hooks
    _inflight: dict[str, str] = field(default_factory=dict, repr=False)

    # Pending handoffs awaiting ack/complete
    _handoff_pending: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # Span stack for parent/child relationships
    _span_stack: list[str] = field(default_factory=list, repr=False)

    def next_seq(self) -> int:
        """Get the next monotonically increasing sequence number."""
        self._seq += 1
        return self._seq

    def now(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def current_parent_span_id(self) -> Optional[str]:
        """Get the current parent span ID from the stack."""
        return self._span_stack[-1] if self._span_stack else self._root_span_id

    def push_span(self, span_id: str) -> None:
        """Push a span onto the stack."""
        self._span_stack.append(span_id)

    def pop_span(self) -> Optional[str]:
        """Pop a span from the stack."""
        return self._span_stack.pop() if self._span_stack else None

    def emit(self, evt: dict[str, Any]) -> None:
        """Emit a trace event to the configured sink."""
        self.sink.write(evt)

    def _make_event(
        self,
        event_type: str,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        status: str = "ok",
        summary: str = "",
        agent: Optional[dict[str, Any]] = None,
        workstream_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attrs_compact: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
        raw_ref: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a fully-formed TraceEvent dict."""
        return {
            "schema_version": "trace_event.v0_1",
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "trace_id": self.trace_id,
            "span_id": span_id or new_span_id(),
            "parent_span_id": parent_span_id,
            "seq": self.next_seq(),
            "ts": self.now(),
            "agent": agent,
            "workstream_id": workstream_id,
            "task_id": task_id,
            "event_type": event_type,
            "status": status,
            "summary": summary,
            "attrs_compact": attrs_compact or {},
            "payload": payload or {},
            "raw_ref": raw_ref or {"storage": "inline"},
        }

    def __enter__(self) -> "ArzuleRun":
        """Start the run context."""
        _active_run.set(self)
        self._root_span_id = new_span_id()

        self.emit(
            self._make_event(
                event_type="run.start",
                span_id=self._root_span_id,
                parent_span_id=None,
                status="ok",
                summary="run started",
            )
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """End the run context and flush events."""
        status = "error" if exc else "ok"
        attrs = {}
        if exc:
            attrs["exc_type"] = exc_type.__name__ if exc_type else None
            attrs["exc_msg"] = str(exc)[:200] if exc else None

        self.emit(
            self._make_event(
                event_type="run.end",
                span_id=new_span_id(),
                parent_span_id=self._root_span_id,
                status=status,
                summary="run ended",
                attrs_compact=attrs,
            )
        )

        try:
            self.sink.flush()
        finally:
            _active_run.set(None)





