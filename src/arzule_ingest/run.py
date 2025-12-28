"""ArzuleRun - Core runtime context manager for trace collection."""

from __future__ import annotations

import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .ids import new_run_id, new_span_id, new_trace_id

if TYPE_CHECKING:
    from .sinks.base import TelemetrySink

# =============================================================================
# Global Run Registry (for thread-safe fallback when ContextVar fails)
# =============================================================================

_run_registry: dict[str, "ArzuleRun"] = {}
_run_registry_lock = threading.Lock()


def register_run(run: "ArzuleRun") -> None:
    """Register a run in the global registry for thread-safe access.
    
    This allows spawned threads (where ContextVar doesn't propagate) to
    look up the run by ID.
    
    Args:
        run: The ArzuleRun instance to register
    """
    from .logger import log_run_registered
    with _run_registry_lock:
        _run_registry[run.run_id] = run
    log_run_registered(run.run_id)


def unregister_run(run_id: str) -> None:
    """Remove a run from the global registry.
    
    Args:
        run_id: The run ID to unregister
    """
    from .logger import log_run_unregistered
    with _run_registry_lock:
        _run_registry.pop(run_id, None)
    log_run_unregistered(run_id)


def get_run_by_id(run_id: str) -> Optional["ArzuleRun"]:
    """Get a run from the global registry by ID.
    
    Args:
        run_id: The run ID to look up
        
    Returns:
        The ArzuleRun instance if found, None otherwise
    """
    with _run_registry_lock:
        return _run_registry.get(run_id)


# =============================================================================
# ContextVar for Current Run
# =============================================================================

_active_run: ContextVar[Optional["ArzuleRun"]] = ContextVar("_active_run", default=None)


def current_run(run_id_hint: Optional[str] = None) -> Optional["ArzuleRun"]:
    """Get the currently active ArzuleRun from context.
    
    Falls back to global registry if ContextVar returns None and run_id_hint
    is provided. This handles the case where CrewAI spawns threads that don't
    inherit the ContextVar.
    
    Args:
        run_id_hint: Optional run ID to use for registry fallback lookup
        
    Returns:
        The active ArzuleRun instance, or None if not found
    """
    run = _active_run.get()
    if run is not None:
        return run
    
    # Fallback: try global registry if hint provided
    if run_id_hint:
        from .logger import log_context_fallback
        run = get_run_by_id(run_id_hint)
        if run:
            log_context_fallback(run_id_hint, "current_run")
        return run
    
    return None


@dataclass
class ArzuleRun:
    """
    Context manager for a single observability run.

    Manages trace lifecycle, sequence numbering, and event emission.
    Supports concurrent task execution with per-task span tracking.
    """

    tenant_id: str
    project_id: str
    sink: "TelemetrySink"
    run_id: str = field(default_factory=new_run_id)
    trace_id: str = field(default_factory=new_trace_id)

    # Internal state
    _seq: int = field(default=0, repr=False)
    _seq_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _root_span_id: Optional[str] = field(default=None, repr=False)

    # Maps for correlating start/end hooks
    _inflight: dict[str, str] = field(default_factory=dict, repr=False)

    # Pending handoffs awaiting ack/complete
    _handoff_pending: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # Span stack for parent/child relationships (legacy, for sequential execution)
    _span_stack: list[str] = field(default_factory=list, repr=False)

    # Per-task span tracking for concurrent execution
    # Maps task_key -> {"root_span_id": str, "current_span_id": str, "stack": list[str]}
    _task_spans: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _task_spans_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Maps agent_key -> current task_key (for correlating agent events to tasks)
    _agent_task_map: dict[str, str] = field(default_factory=dict, repr=False)

    # Crew-level span ID (parent for all concurrent task branches)
    _crew_span_id: Optional[str] = field(default=None, repr=False)

    # Async context tracking for concurrent/async task execution
    # Maps task_key -> {"async_id": str, "started_at": str, "parent_async_id": str | None}
    _async_contexts: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _async_contexts_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Pending async task tracking (for waiting before run.end)
    _pending_async_tasks: set[str] = field(default_factory=set, repr=False)
    _pending_async_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _async_complete_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def next_seq(self) -> int:
        """Get the next monotonically increasing sequence number (thread-safe)."""
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def now(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def current_parent_span_id(self) -> Optional[str]:
        """Get the current parent span ID from the stack (legacy sequential mode)."""
        return self._span_stack[-1] if self._span_stack else self._root_span_id

    def push_span(self, span_id: str) -> None:
        """Push a span onto the stack (legacy sequential mode)."""
        self._span_stack.append(span_id)

    def pop_span(self) -> Optional[str]:
        """Pop a span from the stack (legacy sequential mode)."""
        return self._span_stack.pop() if self._span_stack else None

    # =========================================================================
    # Per-Task Span Management (for concurrent/async execution)
    # =========================================================================

    def set_crew_span(self, span_id: str) -> None:
        """Set the crew-level span ID (parent for concurrent task branches)."""
        self._crew_span_id = span_id

    def get_crew_span(self) -> Optional[str]:
        """Get the crew-level span ID."""
        return self._crew_span_id or self._root_span_id

    def start_task_span(self, task_key: str, agent_key: Optional[str] = None) -> str:
        """
        Start a new span tree for a task (used for concurrent execution).

        Args:
            task_key: Unique identifier for the task (e.g., task.id or generated)
            agent_key: Optional agent key to associate with this task

        Returns:
            The root span ID for this task
        """
        span_id = new_span_id()
        with self._task_spans_lock:
            self._task_spans[task_key] = {
                "root_span_id": span_id,
                "current_span_id": span_id,
                "stack": [span_id],
                "agent_key": agent_key,
            }
            if agent_key:
                self._agent_task_map[agent_key] = task_key
        return span_id

    def end_task_span(self, task_key: str) -> Optional[str]:
        """
        End a task's span tree.

        Args:
            task_key: The task identifier

        Returns:
            The root span ID that was ended, or None
        """
        with self._task_spans_lock:
            task_info = self._task_spans.pop(task_key, None)
            if task_info:
                agent_key = task_info.get("agent_key")
                if agent_key and self._agent_task_map.get(agent_key) == task_key:
                    self._agent_task_map.pop(agent_key, None)
                return task_info["root_span_id"]
        return None

    def get_task_parent_span(self, task_key: Optional[str] = None, agent_key: Optional[str] = None) -> Optional[str]:
        """
        Get the current parent span for a task.

        Args:
            task_key: The task identifier
            agent_key: Alternative - look up task by agent key

        Returns:
            The current parent span ID for the task, or crew/root span if not found
        """
        with self._task_spans_lock:
            # Try task_key first
            if task_key and task_key in self._task_spans:
                return self._task_spans[task_key]["current_span_id"]

            # Fall back to agent_key lookup
            if agent_key and agent_key in self._agent_task_map:
                task_key = self._agent_task_map[agent_key]
                if task_key in self._task_spans:
                    return self._task_spans[task_key]["current_span_id"]

        # Fall back to crew span or root span
        return self._crew_span_id or self._root_span_id

    def get_task_root_span(self, task_key: str) -> Optional[str]:
        """Get the root span ID for a task."""
        with self._task_spans_lock:
            task_info = self._task_spans.get(task_key)
            return task_info["root_span_id"] if task_info else None

    def push_task_span(self, task_key: str, span_id: str) -> None:
        """Push a child span onto a task's span stack."""
        with self._task_spans_lock:
            if task_key in self._task_spans:
                self._task_spans[task_key]["stack"].append(span_id)
                self._task_spans[task_key]["current_span_id"] = span_id

    def pop_task_span(self, task_key: str) -> Optional[str]:
        """Pop a span from a task's span stack."""
        with self._task_spans_lock:
            if task_key in self._task_spans:
                stack = self._task_spans[task_key]["stack"]
                if len(stack) > 1:  # Keep at least the root
                    popped = stack.pop()
                    self._task_spans[task_key]["current_span_id"] = stack[-1]
                    return popped
        return None

    def get_task_key_for_agent(self, agent_key: str) -> Optional[str]:
        """Get the current task key associated with an agent."""
        with self._task_spans_lock:
            return self._agent_task_map.get(agent_key)

    def associate_agent_with_task(self, agent_key: str, task_key: str) -> None:
        """Associate an agent with a task for event correlation."""
        with self._task_spans_lock:
            self._agent_task_map[agent_key] = task_key
            if task_key in self._task_spans:
                self._task_spans[task_key]["agent_key"] = agent_key

    def has_concurrent_tasks(self) -> bool:
        """Check if there are multiple concurrent tasks being tracked."""
        with self._task_spans_lock:
            return len(self._task_spans) > 1

    # =========================================================================
    # Async Context Management (for async_execution=True tasks)
    # =========================================================================

    def start_async_context(self, task_key: str, parent_task_key: Optional[str] = None) -> str:
        """
        Create an async correlation ID for a concurrent task.

        Args:
            task_key: Unique identifier for the task
            parent_task_key: Optional parent task key for nested async

        Returns:
            The async_id (UUID string) for this context
        """
        async_id = str(uuid.uuid4())
        parent_async_id = None
        
        if parent_task_key:
            parent_async_id = self.get_async_id(parent_task_key)
        
        with self._async_contexts_lock:
            self._async_contexts[task_key] = {
                "async_id": async_id,
                "started_at": self.now(),
                "parent_async_id": parent_async_id,
            }
        
        return async_id

    def get_async_id(self, task_key: str) -> Optional[str]:
        """
        Get the async correlation ID for a task.

        Args:
            task_key: The task identifier

        Returns:
            The async_id if found, None otherwise
        """
        with self._async_contexts_lock:
            ctx = self._async_contexts.get(task_key)
            return ctx["async_id"] if ctx else None

    def end_async_context(self, task_key: str) -> Optional[str]:
        """
        End an async context and return its async_id.

        Args:
            task_key: The task identifier

        Returns:
            The async_id that was ended, or None if not found
        """
        with self._async_contexts_lock:
            ctx = self._async_contexts.pop(task_key, None)
            return ctx["async_id"] if ctx else None

    def get_async_context_info(self, task_key: str) -> Optional[dict[str, Any]]:
        """
        Get full async context info for a task.

        Args:
            task_key: The task identifier

        Returns:
            Dict with async_id, started_at, parent_async_id or None
        """
        with self._async_contexts_lock:
            return self._async_contexts.get(task_key)

    def has_async_contexts(self) -> bool:
        """Check if there are any active async contexts."""
        with self._async_contexts_lock:
            return len(self._async_contexts) > 0

    # =========================================================================
    # Pending Async Task Tracking (for waiting before run.end)
    # =========================================================================

    def register_async_task(self, task_key: str) -> None:
        """Register an async task as pending.
        
        Called when an async task starts. The run will wait for all pending
        async tasks to complete before emitting run.end.
        
        Args:
            task_key: Unique identifier for the async task
        """
        from .logger import log_async_task_registered
        with self._pending_async_lock:
            # Only clear event when transitioning from 0 to 1+ pending tasks.
            # This prevents a race where:
            # 1. Last task completes and sets event
            # 2. New task registers and clears event
            # 3. wait_for_async_tasks() may have already returned based on brief event set
            was_empty = len(self._pending_async_tasks) == 0
            self._pending_async_tasks.add(task_key)
            if was_empty:
                self._async_complete_event.clear()
        log_async_task_registered(self.run_id, task_key)

    def complete_async_task(self, task_key: str) -> None:
        """Mark an async task as complete.
        
        Called when an async task finishes. If this was the last pending task,
        signals the completion event.
        
        Args:
            task_key: The task identifier that completed
        """
        from .logger import log_async_task_completed
        with self._pending_async_lock:
            self._pending_async_tasks.discard(task_key)
            if not self._pending_async_tasks:
                self._async_complete_event.set()
        log_async_task_completed(self.run_id, task_key)

    def has_pending_async_tasks(self) -> bool:
        """Check if there are any pending async tasks."""
        with self._pending_async_lock:
            return len(self._pending_async_tasks) > 0

    def get_pending_async_tasks(self) -> set[str]:
        """Get a copy of the pending async task keys."""
        with self._pending_async_lock:
            return self._pending_async_tasks.copy()

    def wait_for_async_tasks(self, timeout: float = 30.0) -> bool:
        """Wait for all pending async tasks to complete.
        
        Uses a loop to handle the race condition where a new task registers
        after the completion event is signaled but before we return.
        
        Args:
            timeout: Maximum seconds to wait (default 30)
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        import time
        deadline = time.monotonic() + timeout
        
        while True:
            # Check under lock if we have pending tasks
            with self._pending_async_lock:
                if not self._pending_async_tasks:
                    return True
            
            # Calculate remaining time
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            
            # Wait for event or timeout
            # If a new task registers while we wait, the event stays cleared
            # and we'll loop back to check again
            self._async_complete_event.wait(timeout=remaining)

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
        register_run(self)  # Register in global registry for thread fallback
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
        from .logger import log_async_wait_start, log_async_timeout
        
        # Wait for pending async tasks before finalizing
        if self.has_pending_async_tasks():
            pending = self.get_pending_async_tasks()
            log_async_wait_start(self.run_id, pending)
            if not self.wait_for_async_tasks(timeout=30.0):
                log_async_timeout(self.run_id, self.get_pending_async_tasks())
        
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
            unregister_run(self.run_id)  # Clean up global registry
            _active_run.set(None)





