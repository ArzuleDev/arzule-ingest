"""ArzuleRun - Core runtime context manager for trace collection."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .ids import new_run_id, new_span_id, new_trace_id
from .run_managers import SpanManager, TaskManager, AsyncTracker, AgentContext
from .run_managers.agent_context import clear_current_agent_context

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
    if run is None and run_id_hint:
        run = get_run_by_id(run_id_hint)
    return run


# =============================================================================
# Main ArzuleRun Class
# =============================================================================

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

    # Closed flag to prevent emissions after __exit__
    _closed: bool = field(default=False, repr=False)

    # Helper managers (composition)
    _span_manager: Optional[SpanManager] = field(default=None, repr=False, init=False)
    _task_manager: Optional[TaskManager] = field(default=None, repr=False, init=False)
    _async_tracker: Optional[AsyncTracker] = field(default=None, repr=False, init=False)
    _agent_context: Optional[AgentContext] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Initialize helper managers after dataclass init."""
        # Will be properly initialized in __enter__ when root_span_id is set
        self._span_manager = None
        self._task_manager = None
        self._async_tracker = AsyncTracker()
        self._agent_context = AgentContext(self.run_id)

    # =========================================================================
    # Core State Methods
    # =========================================================================

    def next_seq(self) -> int:
        """Thread-safe sequence number generator."""
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    # =========================================================================
    # Span Stack Management (delegates to SpanManager)
    # =========================================================================

    def current_parent_span_id(self) -> Optional[str]:
        """Get the current parent span ID."""
        return self._span_manager.current_parent_span_id()

    def push_span(self, span_id: str) -> None:
        """Push a span onto the stack."""
        self._span_manager.push_span(span_id)

    def pop_span(self) -> Optional[str]:
        """Pop a span from the stack."""
        return self._span_manager.pop_span()

    def set_crew_span(self, span_id: str) -> None:
        """Set the crew-level span ID."""
        self._span_manager.set_crew_span(span_id)

    def get_crew_span(self) -> Optional[str]:
        """Get the crew-level span ID."""
        return self._span_manager.get_crew_span()

    def clear_crew_span(self) -> None:
        """Clear the crew-level span ID when a crew ends."""
        self._span_manager.clear_crew_span()

    # =========================================================================
    # Flow Span Management (for multi-crew orchestration)
    # =========================================================================

    def set_flow_span(self, span_id: str) -> None:
        """Set the flow-level span ID."""
        self._span_manager.set_flow_span(span_id)

    def get_flow_span(self) -> Optional[str]:
        """Get the flow-level span ID."""
        return self._span_manager.get_flow_span()

    def clear_flow_span(self) -> None:
        """Clear the flow-level span ID when a flow ends."""
        self._span_manager.clear_flow_span()

    def has_flow_context(self) -> bool:
        """Check if we are currently inside a flow."""
        return self._span_manager.has_flow_context()

    def set_method_span(self, span_id: str) -> None:
        """Set the current flow method span ID."""
        self._span_manager.set_method_span(span_id)

    def get_method_span(self) -> Optional[str]:
        """Get the current flow method span ID."""
        return self._span_manager.get_method_span()

    def clear_method_span(self) -> None:
        """Clear the method-level span ID when a method ends."""
        self._span_manager.clear_method_span()

    # =========================================================================
    # Task Management (delegates to TaskManager)
    # =========================================================================

    def start_task_span(self, task_key: str, agent_key: Optional[str] = None) -> str:
        """Start a new span tree for a task."""
        return self._task_manager.start_task_span(task_key, agent_key)

    def end_task_span(self, task_key: str) -> Optional[str]:
        """End a task's span tree."""
        return self._task_manager.end_task_span(task_key)

    def get_task_parent_span(self, task_key: Optional[str] = None, agent_key: Optional[str] = None) -> Optional[str]:
        """Get the current parent span for a task."""
        return self._task_manager.get_task_parent_span(task_key, agent_key)

    def get_task_root_span(self, task_key: str) -> Optional[str]:
        """Get the root span ID for a task."""
        return self._task_manager.get_task_root_span(task_key)

    def push_task_span(self, task_key: str, span_id: str) -> None:
        """Push a child span onto a task's span stack."""
        self._task_manager.push_task_span(task_key, span_id)

    def pop_task_span(self, task_key: str) -> Optional[str]:
        """Pop a span from a task's span stack."""
        return self._task_manager.pop_task_span(task_key)

    def get_task_key_for_agent(self, agent_key: str) -> Optional[str]:
        """Get the current task key associated with an agent."""
        return self._task_manager.get_task_key_for_agent(agent_key)

    def associate_agent_with_task(self, agent_key: str, task_key: str) -> None:
        """Associate an agent with a task."""
        self._task_manager.associate_agent_with_task(agent_key, task_key)

    def has_concurrent_tasks(self) -> bool:
        """Check if there are multiple concurrent tasks active."""
        return self._task_manager.has_concurrent_tasks()

    # =========================================================================
    # Async Context Tracking (delegates to AsyncTracker)
    # =========================================================================

    def start_async_context(self, task_key: str, parent_task_key: Optional[str] = None) -> str:
        """Create an async correlation ID for a concurrent task."""
        return self._async_tracker.start_async_context(task_key, parent_task_key, now_fn=self.now)

    def get_async_id(self, task_key: str) -> Optional[str]:
        """Get the async correlation ID for a task."""
        return self._async_tracker.get_async_id(task_key)

    def end_async_context(self, task_key: str) -> Optional[str]:
        """End an async context and return its async_id."""
        return self._async_tracker.end_async_context(task_key)

    def get_async_context_info(self, task_key: str) -> Optional[dict[str, Any]]:
        """Get full async context info for a task."""
        return self._async_tracker.get_async_context_info(task_key)

    def has_async_contexts(self) -> bool:
        """Check if there are any active async contexts."""
        return self._async_tracker.has_async_contexts()

    def register_async_task(self, task_key: str) -> None:
        """Register an async task as pending."""
        self._async_tracker.register_async_task(task_key)

    def complete_async_task(self, task_key: str) -> None:
        """Mark an async task as complete."""
        self._async_tracker.complete_async_task(task_key)

    def has_pending_async_tasks(self) -> bool:
        """Check if there are any pending async tasks."""
        return self._async_tracker.has_pending_async_tasks()

    def get_pending_async_tasks(self) -> set[str]:
        """Get the set of pending async task keys."""
        return self._async_tracker.get_pending_async_tasks()

    def wait_for_async_tasks(self, timeout: float = 30.0) -> bool:
        """Wait for all pending async tasks to complete."""
        return self._async_tracker.wait_for_async_tasks(timeout)

    # =========================================================================
    # Agent Context (delegates to AgentContext)
    # =========================================================================

    def set_current_agent(self, agent_info: dict[str, Any]) -> str:
        """Set the currently active agent for this thread.

        Args:
            agent_info: Dict with agent details (id, role, instance_id, etc.)

        Returns:
            The instance_id of the agent for tracking purposes
        """
        return self._agent_context.set_current_agent(agent_info)

    def clear_current_agent(self) -> Optional[str]:
        """Clear the thread-local agent context for this thread.

        Returns:
            The instance_id of the cleared agent, or None if no agent was active
        """
        return self._agent_context.clear_current_agent()

    def get_current_agent(self) -> Optional[dict[str, Any]]:
        """Get the currently active agent for this thread."""
        return self._agent_context.get_current_agent()

    def get_current_agent_instance_id(self) -> Optional[str]:
        """Get the instance_id of the currently active agent."""
        return self._agent_context.get_current_agent_instance_id()

    def get_parent_agent_id(self) -> Optional[str]:
        """Get the ID of the parent agent that spawned the current agent."""
        return self._agent_context.get_parent_agent_id()

    def get_agent_by_instance_id(self, instance_id: str) -> Optional[dict[str, Any]]:
        """Get agent info by instance_id."""
        return self._agent_context.get_agent_by_instance_id(instance_id)

    # =========================================================================
    # Agent Span Management (for concurrent agent execution)
    # =========================================================================

    def start_agent_span(self, agent_instance_id: str, span_id: str) -> None:
        """Start tracking spans for an agent instance."""
        self._span_manager.start_agent_span(agent_instance_id, span_id)

    def end_agent_span(self, agent_instance_id: str) -> Optional[str]:
        """End tracking spans for an agent instance."""
        return self._span_manager.end_agent_span(agent_instance_id)

    def push_agent_span(self, agent_instance_id: str, span_id: str) -> None:
        """Push a span onto an agent's span stack."""
        self._span_manager.push_agent_span(agent_instance_id, span_id)

    def pop_agent_span(self, agent_instance_id: str) -> Optional[str]:
        """Pop a span from an agent's span stack."""
        return self._span_manager.pop_agent_span(agent_instance_id)

    def get_agent_parent_span(self, agent_instance_id: str) -> Optional[str]:
        """Get the current parent span for an agent instance."""
        return self._span_manager.get_agent_parent_span(agent_instance_id)

    def get_agent_root_span(self, agent_instance_id: str) -> Optional[str]:
        """Get the root span for an agent instance."""
        return self._span_manager.get_agent_root_span(agent_instance_id)

    def has_agent_span(self, agent_instance_id: str) -> bool:
        """Check if an agent instance is being tracked."""
        return self._span_manager.has_agent_span(agent_instance_id)

    # =========================================================================
    # Event Emission
    # =========================================================================

    def emit(self, evt: dict[str, Any]) -> None:
        """Emit a trace event to the configured sink.
        
        Silently drops events if the run has been closed to prevent
        race conditions with background threads during run transitions.
        """
        if self._closed:
            return
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

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "ArzuleRun":
        """Start the run context."""
        _active_run.set(self)
        register_run(self)  # Register in global registry for thread fallback
        self._root_span_id = new_span_id()

        # Initialize managers now that root_span_id is set
        self._span_manager = SpanManager(root_span_id=self._root_span_id)
        self._task_manager = TaskManager(root_span_id=self._root_span_id)

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

        # Mark run as closed to prevent background threads from emitting
        # stale events after this point (race condition fix)
        self._closed = True

        try:
            self.sink.flush()
        finally:
            unregister_run(self.run_id)  # Clean up global registry
            _active_run.set(None)
